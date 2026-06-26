import logging
import asyncio
import json
import base64
import io as _io
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
import av
from PIL import Image

import os
import platform
import folder_paths
import comfy.model_management
from server import PromptServer
from aiohttp import web

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)

# Setup global event loop exception handler to silence ConnectionResetError (WinError 10054/10053) on Windows
try:
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            pass

    if loop is not None:
        old_handler = loop.get_exception_handler()

        def silence_connection_reset_handler(loop, context):
            exception = context.get('exception')
            if (isinstance(exception, (ConnectionResetError, ConnectionAbortedError)) or
                (isinstance(exception, OSError) and getattr(exception, 'winerror', None) in (10054, 10053))):
                # Suppress WinError 10054 and WinError 10053 tracebacks in logging
                return
            if old_handler:
                old_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(silence_connection_reset_handler)
except Exception:
    pass

# Custom socket type shared with CSLTXSequencer
GuideData = io.Custom("GUIDE_DATA")
MotionGuideData = io.Custom("MOTION_GUIDE_DATA")

# --- File Check Endpoint for Deduplication ---
@PromptServer.instance.routes.get("/cs_ltx_director_check_file")
async def ltx_director_check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})

    upload_dir = folder_paths.get_input_directory()
    temp_dir = os.path.join(upload_dir, "cs_whatdreamscost")

    # 1. Check if the exact filename exists in whatdreamscost or root input dir
    possible_paths = [
        os.path.join(temp_dir, filename),
        os.path.join(upload_dir, filename)
    ]

    found_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            if file_size:
                try:
                    if os.path.getsize(p) == int(file_size):
                        found_path = p
                        break
                except ValueError:
                    found_path = p
                    break
            else:
                found_path = p
                break

    if found_path:
        rel_name = os.path.relpath(found_path, upload_dir).replace('\\', '/')
        return web.json_response({"exists": True, "name": rel_name})

    # 2. Suffix search if exact match not found
    base_name = os.path.basename(filename)
    suffix = f"_{base_name}"
    try:
        for search_dir in [temp_dir, upload_dir]:
            if os.path.exists(search_dir):
                for f_name in os.listdir(search_dir):
                    if f_name.endswith(suffix) or f_name == base_name:
                        pot_path = os.path.join(search_dir, f_name)
                        if os.path.isfile(pot_path):
                            if file_size:
                                try:
                                    if os.path.getsize(pot_path) == int(file_size):
                                        rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                        return web.json_response({"exists": True, "name": rel_name})
                                except ValueError:
                                    pass
                            else:
                                rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                return web.json_response({"exists": True, "name": rel_name})
    except Exception as e:
        log.warning(f"[CSLTXDirector] Error listing input directory: {e}")

    return web.json_response({"exists": False})


def read_wav_peaks(wav_path):
    import wave
    peaks = []
    with wave.open(wav_path, 'rb') as w:
        n_frames = w.getnframes()
        if n_frames > 0:
            frames_bytes = w.readframes(n_frames)
            samples = np.frombuffer(frames_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step : (i + 1) * step]
                if len(chunk) > 0:
                    max_val = np.max(np.abs(chunk)) / 32767.0
                    peaks.append(float(max_val))
                else:
                    peaks.append(0.0)
        else:
            peaks = [0.0] * 200
    return peaks


def extract_audio_from_video(video_path):
    import wave
    try:
        base, _ = os.path.splitext(video_path)
        output_wav = base + "_extracted_audio.wav"

        # Check if already exists, is not empty, and has the correct 44100Hz sample rate
        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 44:
            try:
                with wave.open(output_wav, 'rb') as w_check:
                    if w_check.getframerate() == 44100:
                        peaks = read_wav_peaks(output_wav)
                        input_dir = folder_paths.get_input_directory()
                        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
                        return rel_output, peaks
            except Exception:
                pass

        # Decode the video using PyAV
        with av.open(video_path) as container:
            if not container.streams.audio:
                return None, None
            stream = container.streams.audio[0]

            # Setup resampler to 44100Hz, Mono, signed 16-bit integer (s16)
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=44100,
            )

            audio_bytes = bytearray()

            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())

            # Flush resampler
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())

            if not audio_bytes:
                return None, None

            # Write WAV file
            with wave.open(output_wav, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2) # 16-bit
                w.setframerate(44100)
                w.writeframes(audio_bytes)

        # Calculate peaks
        peaks = []
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        num_peaks = 200
        step = max(1, len(samples) // num_peaks)
        for i in range(num_peaks):
            chunk = samples[i * step : (i + 1) * step]
            if len(chunk) > 0:
                max_val = np.max(np.abs(chunk)) / 32767.0
                peaks.append(float(max_val))
            else:
                peaks.append(0.0)

        input_dir = folder_paths.get_input_directory()
        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
        return rel_output, peaks
    except Exception as e:
        print(f"[CSLTXDirector] Server audio extraction failed: {e}")
        return None, None


def get_audio_peaks(audio_path):
    import wave
    # If it is already a WAV file, read peaks directly
    _, ext = os.path.splitext(audio_path)
    if ext.lower() == ".wav":
        try:
            return read_wav_peaks(audio_path)
        except Exception:
            pass # fallback to PyAV

    # Use PyAV to decode and resample the audio file
    try:
        with av.open(audio_path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=8000,
            )
            audio_bytes = bytearray()
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())

            if not audio_bytes:
                return None

            peaks = []
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step : (i + 1) * step]
                if len(chunk) > 0:
                    max_val = np.max(np.abs(chunk)) / 32767.0
                    peaks.append(float(max_val))
                else:
                    peaks.append(0.0)
            return peaks
    except Exception as e:
        print(f"[CSLTXDirector] Failed to get audio peaks via PyAV: {e}")
        return None


@PromptServer.instance.routes.get("/cs_ltx_director_get_audio")
async def ltx_director_get_audio(request):
    filename = request.query.get("filename")
    if not filename:
        return web.json_response({"error": "Missing filename"}, status=400)

    upload_dir = folder_paths.get_input_directory()

    clean_filename = filename.replace('\\', '/')
    file_path = os.path.join(upload_dir, clean_filename)
    if not os.path.exists(file_path):
        basename = os.path.basename(clean_filename)
        temp_path = os.path.join(upload_dir, "cs_whatdreamscost", basename)
        if os.path.exists(temp_path):
            file_path = temp_path
        else:
            file_path = os.path.join(upload_dir, basename)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return web.json_response({"error": "File not found"}, status=404)

    _, ext = os.path.splitext(file_path)
    is_audio = ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

    if is_audio:
        peaks = None
        try:
            peaks = get_audio_peaks(file_path)
        except Exception as e:
            print(f"[CSLTXDirector] Failed to get audio peaks for audio file: {e}")

        rel_path = os.path.relpath(file_path, upload_dir).replace('\\', '/')
        return web.json_response({
            "audio_file": rel_path,
            "peaks": peaks
        })

    audio_file, peaks = None, None
    try:
        loop = asyncio.get_event_loop()
        audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
    except Exception as e:
        print(f"[CSLTXDirector] Error extracting audio: {e}")

    return web.json_response({
        "audio_file": audio_file,
        "peaks": peaks
    })


@PromptServer.instance.routes.get("/cs_ltx_director_open_folder")
async def ltx_director_open_folder(request):
    upload_dir = os.path.join(folder_paths.get_input_directory(), "cs_whatdreamscost")
    os.makedirs(upload_dir, exist_ok=True)
    try:
        if hasattr(os, "startfile"):
            os.startfile(upload_dir)
        else:
            import webbrowser
            webbrowser.open(os.path.abspath(upload_dir))
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[CSLTXDirector] Failed to open workspace folder: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


def _read_and_write_file_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


# --- LTX Director Chunked Video Upload Endpoint ---
# Bypasses the 413 Payload Too Large error for large video files.
# This endpoint is self-contained and independent of any other node.
@PromptServer.instance.routes.post("/cs_ltx_director_upload_chunk")
async def ltx_director_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = os.path.join(folder_paths.get_input_directory(), "cs_whatdreamscost")
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename to prevent path traversal attacks (e.g. ../../etc/passwd)
    filename = os.path.basename(filename)
    file_path = os.path.join(upload_dir, filename)

    # Belt-and-suspenders: confirm the resolved path is still inside the upload directory
    if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
        return web.json_response({"error": "Invalid filename"}, status=400)

    # Append chunk to file (write fresh on first chunk, append on subsequent)
    mode = "ab" if chunk_index > 0 else "wb"

    # Offload the blocking read/write disk I/O to a thread executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_file_chunk, file, file_path, mode)

    if chunk_index == total_chunks - 1:
        audio_file, peaks = None, None
        try:
            audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
        except Exception as e:
            print(f"[CSLTXDirector] Error in final chunk audio extraction: {e}")

        return web.json_response({
            "name": f"cs_whatdreamscost/{filename}",
            "audio_file": audio_file,
            "peaks": peaks
        })
    return web.json_response({"status": "ok"})



def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image from the ComfyUI input folder (if imageFile provided) or fallback to base64
    to a ComfyUI-style image tensor of shape [1, H, W, 3], float32 in [0, 1]."""
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _grid_dims_from_layout(layout: str) -> tuple[int, int]:
    text = str(layout or "")
    match = re.search(r"(\d+)\s*x\s*(\d+)", text, re.I)
    if match:
        return max(1, int(match.group(1))), max(1, int(match.group(2)))
    if "九" in text:
        return 3, 3
    if "四" in text:
        return 2, 2
    return 3, 2


def _aspect_value(grid_aspect: str) -> float | None:
    text = str(grid_aspect or "")
    if "16:9" in text:
        return 16 / 9
    if "9:16" in text:
        return 9 / 16
    if "1:1" in text or "方" in text:
        return 1.0
    return None


def _center_crop_aspect(tensor: torch.Tensor, aspect: float | None) -> torch.Tensor:
    if tensor is None or aspect is None:
        return tensor
    _, height, width, _ = tensor.shape
    if height <= 1 or width <= 1:
        return tensor
    current = width / height
    if abs(current - aspect) < 1e-3:
        return tensor
    if current > aspect:
        new_w = max(1, int(round(height * aspect)))
        x0 = max(0, (width - new_w) // 2)
        return tensor[:, :, x0:x0 + new_w, :]
    new_h = max(1, int(round(width / aspect)))
    y0 = max(0, (height - new_h) // 2)
    return tensor[:, y0:y0 + new_h, :, :]


def _split_grid_image_tensor(grid_images: torch.Tensor | None, layout: str, border_crop: float = 1.0, grid_aspect: str = "") -> list[torch.Tensor]:
    if grid_images is None:
        return []

    cols, rows = _grid_dims_from_layout(layout)
    count = max(1, cols * rows)
    aspect = _aspect_value(grid_aspect)
    crop_strength = max(0.0, float(border_crop or 0.0))

    batch = int(grid_images.shape[0])
    if batch >= count and batch != 1:
        return [_center_crop_aspect(grid_images[i:i + 1], aspect) for i in range(count)]

    source = grid_images[:1]
    _, height, width, _ = source.shape
    cells = []
    for idx in range(count):
        col = idx % cols
        row = idx // cols
        x0 = int(round(col * width / cols))
        x1 = int(round((col + 1) * width / cols))
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))

        cell_w = max(1, x1 - x0)
        cell_h = max(1, y1 - y0)
        pad_x = int(round(cell_w * 0.012 * crop_strength))
        pad_y = int(round(cell_h * 0.012 * crop_strength))
        if cell_w > pad_x * 2 + 8:
            x0 += pad_x
            x1 -= pad_x
        if cell_h > pad_y * 2 + 8:
            y0 += pad_y
            y1 -= pad_y

        cell = source[:, y0:y1, x0:x1, :]
        cells.append(_center_crop_aspect(cell, aspect))
    return cells


def _load_grid_segment_tensor(seg: dict, grid_images: torch.Tensor | None, layout: str, border_crop: float, grid_aspect: str) -> torch.Tensor:
    cells = _split_grid_image_tensor(grid_images, layout, border_crop, grid_aspect)
    if not cells:
        return _load_image_tensor(seg)
    idx = int(seg.get("grid_index", seg.get("batch_index", 0)) or 0)
    idx = max(0, min(idx, len(cells) - 1))
    return cells[idx]


def _strip_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_grid_prompt(prompt: str) -> str:
    marker = r"^\s*(?:分镜|镜头|画面|shot)\s*[一二三四五六七八九十\d]+\s*[:：.\-、]\s*"
    return re.sub(marker, "", str(prompt or ""), flags=re.I).strip()


def _parse_grid_prompts(text: str) -> list[str]:
    text = _strip_fence(text)
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("segments", "shots", "scenes", "storyboard", "分镜", "镜头"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            if isinstance(data, dict):
                numbered = []
                for key, value in data.items():
                    match = re.search(r"\d+", str(key))
                    if match:
                        numbered.append((int(match.group(0)), value))
                if numbered:
                    data = [value for _, value in sorted(numbered)]
                else:
                    data = list(data.values())
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append(_clean_grid_prompt(item))
                elif isinstance(item, dict):
                    for key in ("prompt", "description", "text", "content", "scene", "action", "动态描述", "描述", "提示词", "画面"):
                        if item.get(key):
                            out.append(_clean_grid_prompt(item[key]))
                            break
            return [item for item in out if item]
    except Exception:
        pass

    marker = re.compile(r"(?:^|\n)\s*(?:分镜|镜头|画面|shot)\s*[一二三四五六七八九十\d]+\s*[:：.\-、]\s*", re.I)
    matches = list(marker.finditer(text))
    if matches:
        prompts = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            prompts.append(_clean_grid_prompt(text[start:end]))
        return [item for item in prompts if item]
    return [_clean_grid_prompt(part) for part in re.split(r"\n+", text) if _clean_grid_prompt(part)]


def _even_grid_lengths(total_frames: int, count: int) -> list[int]:
    total_frames = max(count, int(total_frames or count))
    base = total_frames // count
    lengths = [max(1, base)] * count
    for i in range(total_frames - base * count):
        lengths[i % count] += 1
    return lengths


def _build_grid_timeline_if_empty(tdata: dict, grid_text: str, duration_frames: int, layout: str) -> dict:
    if tdata.get("segments"):
        return tdata
    cols, rows = _grid_dims_from_layout(layout)
    count = cols * rows
    prompts = _parse_grid_prompts(grid_text)
    lengths = _even_grid_lengths(duration_frames, count)
    cursor = 0
    segments = []
    for idx in range(count):
        segments.append({
            "id": f"grid_{idx}_{os.urandom(4).hex()}",
            "start": int(cursor),
            "length": int(lengths[idx]),
            "prompt": prompts[idx] if idx < len(prompts) else "",
            "type": "image",
            "source": "grid_images",
            "grid_index": idx,
            "batch_index": idx,
            "guideStrength": 1.0,
        })
        cursor += int(lengths[idx])
    tdata["segments"] = segments
    tdata.setdefault("motionSegments", [])
    tdata.setdefault("audioSegments", [])
    return tdata


def _load_video_tensor(seg: dict, frame_rate: float) -> torch.Tensor:
    """Extracts a sequence of frames from a video file based on the segment's trim parameters,
    and returns them as an [N, H, W, 3] float32 tensor."""
    file_path = os.path.join(folder_paths.get_input_directory(), seg.get("imageFile", ""))

    if not os.path.exists(file_path):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    trim_start_frames = float(seg.get("trimStart", 0))
    length_frames = float(seg.get("length", 1))
    start_sec = trim_start_frames / frame_rate

    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Seek slightly before target to hit a keyframe
            if stream.time_base:
                seek_pts = int((max(0, start_sec - 0.5)) / float(stream.time_base))
            else:
                seek_pts = int((max(0, start_sec - 0.5)) * av.time_base)

            container.seek(seek_pts, stream=stream, backward=True)

            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)

                if frame_time is None:
                    frame_time = 0.0

                if frame_time < start_sec - 0.01:
                    continue

                frames.append(frame.to_ndarray(format='rgb24'))

                if len(frames) >= int(length_frames):
                    break
    except Exception as e:
        log.warning(f"[PromptRelay] Video extract error: {e}")

    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)

def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    """Resize an [N, H, W, 3] float32 tensor to target dimensions using the given method,
    then snap the final dimensions to be divisible by `divisible_by`."""

    def snap(val, div):
        return max(div, (val // div) * div)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor

    t_nchw = tensor.permute(0, 3, 1, 2)

    if method == "stretch to fit":
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)

    elif method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        resized = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

    elif method == "pad" or method == "pad green":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_l = (tw - new_w) // 2
        pad_t = (th - new_h) // 2

        if method == "pad green":
            resized = torch.zeros((N, C, th, tw), dtype=t_nchw.dtype, device=t_nchw.device)
            # #66FF00 is roughly R: 102/255, G: 255/255, B: 0
            resized[:, 0, :, :] = 102 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, 2, :, :] = 0.0
            resized[:, :, pad_t:pad_t+new_h, pad_l:pad_l+new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t), mode="constant", value=0)

    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w = int(W * ratio)
        new_h = int(H * ratio)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner[:, :, top:top+th, left:left+tw]

    else:
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)

    return resized.permute(0, 2, 3, 1)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Apply H.264 compression artefacts to an [N, H, W, 3] float32 tensor (ComfyUI image format).
    crf=0 means no compression. Uses PyAV to encode/decode frames in-memory."""
    if crf == 0:
        return tensor

    N, H, W, C = tensor.shape

    # Dimensions must be even for H.264
    h = (H // 2) * 2
    w = (W // 2) * 2

    # uint8 [N, H, W, 3]
    tensor_bytes = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()

    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}

        for i in range(N):
            frame = av.VideoFrame.from_ndarray(tensor_bytes[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)

        for pkt in stream.encode(None):
            container.mux(pkt)

        container.close()

        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = [frame_r.to_ndarray(format="rgb24") for frame_r in container_r.decode(video=0)]
        container_r.close()

        if not decoded:
            return tensor

        decoded_np = np.stack(decoded).astype(np.float32) / 255.0

        # Re-embed into original tensor shape (may have been cropped by even-rounding)
        out = tensor.clone()
        dec_N = min(N, len(decoded))
        out[:dec_N, :h, :w] = torch.from_numpy(decoded_np[:dec_N]).to(tensor.device, tensor.dtype)

        return out

    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


def _build_combined_audio(timeline_data_str: str, start_frame: int, duration_frames: int, frame_rate: float, override_audio: bool = False) -> dict:
    """Parses timeline JSON, loads/trims audio directly from memory using PyAV,
    and aligns to a global timeline yielding ComfyUI's format.
    Output length explicitly mimics the timeline's duration_frames length."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    if not timeline_data_str:
        return empty_audio

    try:
        data = json.loads(timeline_data_str)
        is_retake = data.get("retakeMode", False)
        if is_retake and data.get("retakeVideo"):
            retake_vid = data.get("retakeVideo")
            audio_segs = [{
                "videoFile": retake_vid.get("imageFile") or retake_vid.get("fileName"),
                "audioFile": retake_vid.get("imageFile") or retake_vid.get("fileName"),
                "start": 0,
                "length": retake_vid.get("videoDurationFrames", duration_frames),
                "trimStart": 0
            }]
            override_audio = True
        elif override_audio:
            audio_segs = data.get("motionSegments", [])
        else:
            audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio

    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        file_key = "videoFile" if override_audio else "audioFile"
        if seg.get(file_key):
            file_path = os.path.join(folder_paths.get_input_directory(), seg[file_key])
            if not os.path.exists(file_path):
                # Try fallback under whatdreamscost subfolder
                basename = os.path.basename(seg[file_key])
                fallback_path = os.path.join(folder_paths.get_input_directory(), "cs_whatdreamscost", basename)
                if os.path.exists(fallback_path):
                    file_path = fallback_path

            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())

        if not override_audio and not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except:
                pass

        if not buffer:
            continue

        try:
            clip_frames = []

            # Use PyAV to decode directly from memory buffer
            with av.open(buffer) as container:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]

                # Setup resampler to ensure output is 44.1kHz, Stereo, Float32 Planar
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )

                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        # to_ndarray() on fltp gives shape (channels, samples)
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))

                # Flush the resampler to get any remaining samples
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            # Concatenate all frame blocks along the samples dimension (dim 1)
            waveform = torch.cat(clip_frames, dim=1) # Shape: [2, total_clip_samples]

            # Calculate interactive trim boundaries
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))

            if start_frames + length_frames <= start_frame:
                continue

            offset = max(0, start_frame - start_frames)
            trim_start_frames += offset
            length_frames = max(1, length_frames - offset)
            start_frames = max(0, start_frames - start_frame)

            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples

            if start_sample_src < 0: start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]

            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0: continue

            # Extract the correct segment of the audio
            clip_waveform = waveform[:, start_sample_src:end_sample_src]

            # Position onto the timeline
            start_sample_dst = int(start_frames / frame_rate * target_sr)

            if start_sample_dst >= out_waveform.shape[1]:
                continue

            end_sample_dst = start_sample_dst + actual_length

            # Clip any trailing overflow so we don't index past the timeline bounds
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length

            if actual_length <= 0:
                continue

            # Additive composite (allows clips overlapping to sum together naturally)
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    # Split prompts but do NOT filter out empty ones yet, so we can detect them
    locals_list = [p.strip() for p in local_prompts.split("|")]

    # If there are no visual segments on the timeline (e.g., only using IC-LoRA motion track),
    # bypass the local prompt chunking entirely and just use the global prompt.
    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        log.info("[PromptRelay] No local segments found. Using global prompt exclusively.")
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(global_prompt))
        return model.clone(), conditioning

    # Check if any specific segment is empty and apply fallbacks
    for i, p in enumerate(locals_list):
        if not p:
            fallback = global_prompt.strip() if global_prompt else "video"
            if not fallback:
                fallback = "video"
            locals_list[i] = fallback

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


class CSLTXDirector(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CSLTXDirector",
            display_name="CS-LTX 2.0 宫格导演台",
            category="CS-WhatDreamsCost",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The duration_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model", display_name="模型"),
                io.Clip.Input("clip", display_name="文本编码器"),
                io.Vae.Input("audio_vae", display_name="音频 VAE", optional=True, tooltip="可选。连接音频 VAE 后可以生成或处理音频潜空间。"),
                io.Latent.Input("optional_latent", display_name="可选潜空间", optional=True, tooltip="可选。连接已有视频潜空间后，将替代自动生成的空潜空间。"),
                io.String.Input(
                    "global_prompt", display_name="全局提示词", multiline=True, default="", force_input=True, optional=True,
                    tooltip="作用于整段视频，用来固定人物、主体、场景和整体风格。",
                ),
                io.Float.Input(
                    "start_second", display_name="开始秒数", default=0.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="本次生成在时间线中的开始时间，单位为秒。",
                ),
                io.Float.Input(
                    "end_second", display_name="结束秒数", default=5.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="本次生成在时间线中的结束时间，单位为秒。",
                ),
                io.Float.Input(
                    "duration_seconds", display_name="总秒数", default=5.0, min=0.1, max=1000.0, step=0.01,
                    tooltip="时间线总时长，单位为秒，会和帧数自动同步。",
                ),
                io.Int.Input(
                    "start_frame", display_name="开始帧", default=0, min=0, max=10000, step=1,
                    tooltip="本次生成在时间线中的开始帧。",
                ),
                io.Int.Input(
                    "end_frame", display_name="结束帧", default=120, min=1, max=10000, step=1,
                    tooltip="本次生成在时间线中的结束帧。",
                ),
                io.Int.Input(
                    "duration_frames", display_name="总帧数", default=120, min=1, max=10000, step=1,
                    tooltip="时间线总帧数，用于前端时间线比例和生成长度。",
                ),
                io.String.Input(
                    "timeline_data", display_name="时间线数据", default="",
                    tooltip="导演台前端自动管理的时间线 JSON，一般不需要手动编辑。",
                ),
                io.Boolean.Input(
                    "use_custom_audio", display_name="使用时间线音频", default=False, optional=True,
                    tooltip="开启后使用时间线中的音频轨；关闭后由模型自行生成音频。",
                ),
                io.Boolean.Input(
                    "use_custom_motion", display_name="使用运动视频引导", default=True, optional=True,
                    tooltip="开启后使用时间线中的运动视频轨；关闭后忽略运动视频分段。",
                ),
                io.Boolean.Input(
                    "inpaint_audio", display_name="补全音频空隙", default=True, optional=True,
                    tooltip="开启后，音频轨中的空白区域会尝试用生成音频补齐。",
                ),
                io.String.Input(
                    "local_prompts", display_name="分段提示词", multiline=True, default="",
                    tooltip="由导演台时间线自动写入。",
                ),
                io.String.Input(
                    "segment_lengths", display_name="分段长度", default="",
                    tooltip="由导演台时间线自动写入，表示每段的帧数。",
                ),
                io.Float.Input(
                    "epsilon", display_name="分段边界强度", default=0.001, min=0.0001, max=0.99, step=0.0001,
                    tooltip="提示词分段边界控制。数值越小边界越清晰，数值越大过渡越柔和。",
                ),
                io.Float.Input(
                    "frame_rate", display_name="帧率", default=24, min=1, max=240, step=1, optional=True,
                    tooltip="每秒帧数，用于时间线显示、音频同步和帧数换算。",
                ),
                io.Combo.Input(
                    "display_mode", display_name="时间显示方式", options=["frames", "seconds"], default="seconds", optional=True,
                    tooltip="时间线可以按帧或秒显示，内部仍按帧保存。",
                ),
                io.String.Input(
                    "guide_strength", display_name="引导强度", default="",
                    tooltip="由导演台时间线自动写入，表示各图像分段的引导强度。",
                ),
                io.Int.Input(
                    "custom_width", display_name="输出宽度", default=0, min=0, max=8192, step=1, optional=True,
                    tooltip="所有图像分段的目标宽度。设为 0 时尽量使用原图宽度。",
                ),
                io.Int.Input(
                    "custom_height", display_name="输出高度", default=0, min=0, max=8192, step=1, optional=True,
                    tooltip="所有图像分段的目标高度。设为 0 时尽量使用原图高度。",
                ),
                io.Combo.Input(
                    "resize_method",
                    options=["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"],
                    default="maintain aspect ratio",
                    display_name="图像适配方式",
                    optional=True,
                    tooltip="图像分段适配目标尺寸的方式。",
                ),
                io.Int.Input(
                    "divisible_by", display_name="尺寸整除", default=32, min=1, max=256, step=1, optional=True,
                    tooltip="输出尺寸会对齐到该数值的整数倍，例如 LTX 常用 32。",
                ),
                io.Int.Input(
                    "img_compression", display_name="图像压缩", default=18, min=0, max=100, step=1, optional=True,
                    tooltip="对引导图应用的 H.264 CRF 压缩。0 表示不压缩，数值越高压缩痕迹越重。",
                ),
                io.Boolean.Input(
                    "override_audio", display_name="使用 IC 视频音频", default=False, optional=True,
                    tooltip="使用 IC-LoRA 视频里的音频，而不是使用独立音频轨。",
                ),
                io.Boolean.Input(
                    "grid_mode", display_name="宫格模式", default=False, optional=True,
                    tooltip="开启后，导演台会监听宫格图像来源，并可自动把宫格拆成主轨分镜。",
                ),
                io.Image.Input(
                    "grid_images", display_name="宫格图像", optional=True,
                    tooltip="输入 2x2、3x2 或 3x3 宫格图像。运行时会从这里重新拆分高质量引导图。",
                ),
                io.String.Input(
                    "grid_text", display_name="宫格分镜文本", multiline=True, default="", force_input=True, optional=True,
                    tooltip="输入或连接四段、六段、九段中文分镜文本，用于自动写入主轨分镜提示词。",
                ),
                io.Combo.Input(
                    "grid_layout",
                    display_name="宫格模式选择",
                    options=["2x2 四宫格", "3x2 六宫格", "3x3 九宫格"],
                    default="3x2 六宫格",
                    optional=True,
                    tooltip="选择宫格读取顺序：从左到右、从上到下。",
                ),
                io.Combo.Input(
                    "grid_aspect",
                    display_name="分镜比例",
                    options=["自动 / 保持单格比例", "16:9 横屏", "9:16 竖屏", "1:1 方图"],
                    default="自动 / 保持单格比例",
                    optional=True,
                    tooltip="拆分后的单格分镜比例。自动模式会保持原始单格比例。",
                ),
                io.Float.Input(
                    "grid_border_crop", display_name="白边裁剪强度", default=1.0, min=0.0, max=3.0, step=0.05, optional=True,
                    tooltip="拆宫格时向内裁掉格线和白边。1.0 为常规，1.5 裁更多，0.5 裁更少。",
                ),
                io.Boolean.Input(
                    "grid_auto_refresh", display_name="宫格自动刷新", default=True, optional=True,
                    tooltip="开启后，宫格图像来源变化时自动刷新主轨分镜。",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="模型"),
                io.Conditioning.Output(display_name="正向条件"),
                io.Latent.Output(display_name="视频潜空间", tooltip="自动生成的 LTXV 空潜空间；连接可选潜空间时不会使用。"),
                io.Latent.Output(display_name="音频潜空间", tooltip="自动生成的音频潜空间，开启时间线音频时会使用自定义音频。"),
                GuideData.Output(display_name="引导数据"),
                MotionGuideData.Output(display_name="运动引导数据"),
                io.Float.Output(display_name="帧率", tooltip="导演台时间线使用的帧率。"),
                io.Audio.Output(display_name="合成音频", tooltip="按时间线合成后的音频布局。"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, start_second, end_second, duration_seconds, start_frame, end_frame, duration_frames,
                timeline_data, local_prompts, segment_lengths, global_prompt="", guide_strength="", epsilon=1e-3,
                frame_rate=24, display_mode="seconds",
                custom_width=768, custom_height=512, resize_method="maintain aspect ratio",
                divisible_by=32, img_compression=0, audio_vae=None, optional_latent=None,
                use_custom_audio=False, inpaint_audio=True, use_custom_motion=True, override_audio=False,
                grid_mode=False, grid_images=None, grid_text="", grid_layout="3x2 六宫格",
                grid_aspect="自动 / 保持单格比例", grid_border_crop=1.0, grid_auto_refresh=True) -> io.NodeOutput:

        # Parse timeline data
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
        except Exception as e:
            log.error(f"[CSLTXDirector] execute timeline_data parse error: {e}")
            tdata = {}

        is_retake_mode = tdata.get("retakeMode", False)
        is_retake_active = is_retake_mode and tdata.get("retakeVideo") is not None

        if bool(grid_mode) and grid_images is not None:
            tdata = _build_grid_timeline_if_empty(tdata, grid_text, duration_frames, grid_layout)
            timeline_data = json.dumps(tdata, ensure_ascii=False)

        # Extract global_prompt from timeline_data if not connected/empty
        if not global_prompt:
            if is_retake_mode:
                global_prompt = tdata.get("retake_global_prompt", "")
            else:
                global_prompt = tdata.get("global_prompt", "")

        log.info(f"[CSLTXDirector] execute RECEIVED global_prompt: {repr(global_prompt)}")

        # --- Build guide_data from image segments FIRST (to derive output dimensions) ---
        guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": frame_rate}
        derived_w, derived_h = custom_width, custom_height
        try:
            img_segs = [
                s for s in tdata.get("segments", [])
                if s.get("type", "image") in ("image", "video")
                and (s.get("imageFile") or s.get("imageB64") or s.get("source") in ("grid_images", "storyboard_images"))
                and int(s.get("start", 0)) < start_frame + duration_frames
                and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
            ]
            img_segs.sort(key=lambda s: s["start"])

            strengths = []
            if guide_strength.strip():
                strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]

            for idx, seg in enumerate(img_segs):
                seg_start = int(seg.get("start", 0))
                offset = max(0, start_frame - seg_start)

                if seg.get("type") == "video":
                    if offset > 0:
                        seg["trimStart"] = float(seg.get("trimStart", 0)) + offset
                        seg["length"] = max(1, int(seg.get("length", 1)) - offset)
                    tensor = _load_video_tensor(seg, float(frame_rate))
                elif seg.get("source") in ("grid_images", "storyboard_images") and grid_images is not None:
                    tensor = _load_grid_segment_tensor(seg, grid_images, grid_layout, grid_border_crop, grid_aspect)
                else:
                    tensor = _load_image_tensor(seg)

                # Apply resize
                src_h, src_w = tensor.shape[1], tensor.shape[2]

                def snap(val, div):
                    return max(div, (val // div) * div)

                if custom_width > 0 and custom_height > 0:
                    # Both dimensions set — apply selected resize_method (pad, crop, stretch, maintain AR)
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
                elif custom_width > 0:
                    # Width only — scale height from AR, snap both, then resize to exact dimensions
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                elif custom_height > 0:
                    # Height only — scale width from AR, snap both, then resize to exact dimensions
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                else:
                    # Both zero — keep original dimensions, just snap to divisible_by
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)


                # Apply compression
                if img_compression > 0:
                    tensor = _compress_image(tensor, img_compression)

                # Record dimensions of the first processed image for latent generation
                if idx == 0:
                    derived_h = tensor.shape[1]
                    derived_w = tensor.shape[2]

                if seg.get("isEndFrame"):
                    insert_frame = max(0, seg_start + int(seg.get("length", 1)) - 1 - start_frame)
                else:
                    insert_frame = max(0, seg_start - start_frame)
                strength = strengths[idx] if idx < len(strengths) else 1.0
                guide_data["images"].append(tensor)
                guide_data["insert_frames"].append(insert_frame)
                guide_data["strengths"].append(float(strength))

            # If no images were loaded from the timeline, create a dummy image at strength 0
            # to prevent artifacts in text-to-video mode.
            if not guide_data["images"] and optional_latent is None:
                src_w = derived_w if derived_w > 0 else 768
                src_h = derived_h if derived_h > 0 else 512

                # If there's an IC-LoRA video or retake base video on the timeline, extract its dimensions for accurate aspect ratio scaling
                tdata_motion = json.loads(timeline_data) if timeline_data else {}
                found_dims = False

                # Check for retake base video first
                is_retake = tdata_motion.get("retakeMode", False)
                retake_vid = tdata_motion.get("retakeVideo") or {}
                retake_file = retake_vid.get("imageFile", "") if isinstance(retake_vid, dict) else ""
                if is_retake and retake_file:
                    r_path = os.path.join(folder_paths.get_input_directory(), retake_file)
                    if not os.path.exists(r_path):
                        basename = os.path.basename(retake_file)
                        fallback_path = os.path.join(folder_paths.get_input_directory(), "cs_whatdreamscost", basename)
                        if os.path.exists(fallback_path):
                            r_path = fallback_path
                    if os.path.exists(r_path):
                        try:
                            with av.open(r_path) as container:
                                stream = container.streams.video[0]
                                src_w = stream.width or stream.codec_context.width
                                src_h = stream.height or stream.codec_context.height
                                found_dims = True
                        except:
                            pass

                # Fallback to normal motion segments
                if not found_dims:
                    for mseg in tdata_motion.get("motionSegments", []):
                        v_file = mseg.get("videoFile")
                        if v_file:
                            v_path = os.path.join(folder_paths.get_input_directory(), v_file)
                            if not os.path.exists(v_path):
                                basename = os.path.basename(v_file)
                                fallback_path = os.path.join(folder_paths.get_input_directory(), "cs_whatdreamscost", basename)
                                if os.path.exists(fallback_path):
                                    v_path = fallback_path
                            if os.path.exists(v_path):
                                try:
                                    with av.open(v_path) as container:
                                        stream = container.streams.video[0]
                                        src_w = stream.width or stream.codec_context.width
                                        src_h = stream.height or stream.codec_context.height
                                        found_dims = True
                                        break
                                except:
                                    pass

                # Create a dummy tensor of the exact source dimensions
                tensor = torch.zeros((1, src_h, src_w, 3), dtype=torch.float32)

                def snap(val, div):
                    return max(div, (val // div) * div)

                # Route the dummy tensor through the exact same resizing pipeline
                if custom_width > 0 and custom_height > 0:
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
                elif custom_width > 0:
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                elif custom_height > 0:
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                else:
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)

                guide_data["images"].append(tensor)
                guide_data["insert_frames"].append(0)
                guide_data["strengths"].append(0.0)

                derived_w = tensor.shape[2]
                derived_h = tensor.shape[1]

        except Exception as e:
            log.warning("[PromptRelay] Could not build guide_data: %s", e)

        # --- Auto-generate LTXV latent if none was provided ---
        # Apply the community 8n+1 rule directly to the timeline's duration_frames:
        # int(ceil(((duration_frames) - 1) / 8) * 8) + 1
        # This ensures we get AT LEAST the requested frames, snapped to LTXV's requirements.
        ltxv_length = int(math.ceil((duration_frames - 1) / 8.0) * 8) + 1

        if optional_latent is None:
            latent_w = max(32, (derived_w // 32) * 32)
            latent_h = max(32, (derived_h // 32) * 32)
            # LTXV temporal: ((length - 1) // 8) + 1 latent frames; invert to get pixel frames -> length
            latent_t = ((ltxv_length - 1) // 8) + 1
            samples = torch.zeros(
                [1, 128, latent_t, latent_h // 32, latent_w // 32],
                device=comfy.model_management.intermediate_device(),
            )
            latent = {"samples": samples}
            log.info(
                "[PromptRelay] Auto-generated LTXV latent: %dx%d, %d pixel frames (%d latent frames)",
                latent_w, latent_h, ltxv_length, latent_t,
            )
        else:
            latent = optional_latent

        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )

        # --- Build Audio Output ---
        audio_out = _build_combined_audio(timeline_data, start_frame, ltxv_length, float(frame_rate), override_audio=override_audio)

        # --- Audio Latent Generation ---
        audio_latent = {}

        if audio_vae is not None:
            # Helper to generate empty latent
            def get_empty_latent():
                # Support both raw AudioVAE objects and ComfyUI VAE wrappers.
                inner = getattr(audio_vae, "first_stage_model", audio_vae)
                z_channels = audio_vae.latent_channels
                audio_freq = inner.latent_frequency_bins
                num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, float(frame_rate))
                audio_latents = torch.zeros(
                    (1, z_channels, num_audio_latents, audio_freq),
                    device=comfy.model_management.intermediate_device(),
                )
                return {"samples": audio_latents, "type": "audio"}

            if use_custom_audio or override_audio or is_retake_active:
                try:
                    if audio_out is not None:
                        # 1. Encode audio waveform into latent space
                        waveform = audio_out["waveform"]
                        if waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        if waveform.ndim != 3:
                            raise ValueError(
                                f"Expected custom audio waveform with 2 or 3 dims, got shape {tuple(waveform.shape)}"
                            )

                        # Wrapped ComfyUI VAE expects (batch, samples, channels);
                        # raw AudioVAE expects a dict with waveform in (batch, channels, samples).
                        if hasattr(audio_vae, "first_stage_model"):
                            latent_samples = audio_vae.encode(waveform.movedim(1, -1))
                        else:
                            latent_samples = audio_vae.encode({
                                "waveform": waveform,
                                "sample_rate": audio_out["sample_rate"],
                            })

                        if latent_samples.numel() == 0:
                            raise ValueError("Encoded audio latent is empty (0 elements).")

                        # 2. Create a 3D gap mask [B, F, H] to avoid accidental broadcasting to the 5D video latent
                        # which also has 128 channels. A 4D audio mask [1, 128, F, H] confuses ComfyUI's KSampler
                        # into masking the video latent as well, causing black frames.
                        B, C, F_len, H_len = latent_samples.shape

                        if is_retake_active:
                            gap_mask = torch.zeros((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)

                            retake_start = float(tdata.get("retakeStart", 0))
                            retake_len = float(tdata.get("retakeLength", 0))

                            overlap_start = max(start_frame, retake_start)
                            overlap_end = min(start_frame + ltxv_length, retake_start + retake_len)

                            if overlap_end > overlap_start:
                                rel_start = overlap_start - start_frame
                                rel_len = overlap_end - overlap_start

                                start_sec = rel_start / float(frame_rate)
                                len_sec = rel_len / float(frame_rate)
                                total_sec = ltxv_length / float(frame_rate)

                                start_idx = int((start_sec / total_sec) * F_len)
                                end_idx = int(((start_sec + len_sec) / total_sec) * F_len)

                                start_idx = max(0, min(F_len, start_idx))
                                end_idx = max(0, min(F_len, end_idx))

                                gap_mask[:, start_idx:end_idx, :] = 1.0
                        else:
                            gap_mask = torch.ones((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)

                            audio_segs_key = "motionSegments" if override_audio else "audioSegments"
                            file_key = "videoFile" if override_audio else "audioFile"
                            for seg in tdata.get(audio_segs_key, []):
                                if not seg.get(file_key):
                                    continue

                                seg_start = float(seg.get("start", 0))
                                seg_len = float(seg.get("length", 1))

                                if seg_start + seg_len <= start_frame or seg_start >= start_frame + ltxv_length:
                                    continue

                                offset = max(0, start_frame - seg_start)
                                seg_len = max(1.0, seg_len - offset)
                                seg_start = max(0, seg_start - start_frame)

                                start_sec = seg_start / float(frame_rate)
                                len_sec = seg_len / float(frame_rate)
                                total_sec = ltxv_length / float(frame_rate)

                                start_idx = int((start_sec / total_sec) * F_len)
                                end_idx = int(((start_sec + len_sec) / total_sec) * F_len)
                                gap_mask[:, start_idx:end_idx, :] = 0.0

                        if inpaint_audio:
                            # Generate new audio in the gaps, preserve custom audio segments
                            mask = gap_mask
                        else:
                            # Preserve the entire audio latent (no generation).
                            # We use a 3D zeros mask to prevent video blackouts.
                            mask = torch.zeros((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)

                        audio_latent = {
                            "samples": latent_samples,
                            "type": "audio",
                            "noise_mask": mask
                        }
                        log.info("[PromptRelay] Generated custom audio latent with dynamic noise mask.")
                    else:
                        raise ValueError("No audio waveform to encode.")
                except Exception as e:
                    log.error("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                # Generate empty latent
                try:
                    audio_latent = get_empty_latent()
                    log.info("[PromptRelay] Auto-generated empty audio latent.")
                except Exception as e:
                    log.error("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

        # --- Motion guide output from timeline video segments ---
        motion_guide_data = {"segments": [], "frame_rate": float(frame_rate), "duration_frames": int(duration_frames), "resize_method": resize_method}
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            if use_custom_motion:
                motion_segments = tdata.get("motionSegments", [])
            else:
                motion_segments = []
            for seg in motion_segments:
                seg_start = int(seg.get("start", 0))
                length = int(seg.get("length", 1))
                if seg_start >= start_frame + duration_frames or seg_start + length <= start_frame:
                    continue
                if not seg.get("videoFile"):
                    continue

                offset = max(0, start_frame - seg_start)
                new_start = max(0, seg_start - start_frame)

                # Trim length so it doesn't extend beyond duration_frames
                clipped_len = min(length - offset, duration_frames - new_start)
                if clipped_len <= 0:
                    continue

                clean = dict(seg)
                clean["start"] = new_start
                clean["length"] = clipped_len
                clean["trimStart"] = float(seg.get("trimStart", 0)) + offset
                motion_guide_data["segments"].append(clean)
        except Exception as e:
            log.warning("[CSLTXDirector] Could not build motion_guide_data: %s", e)

        # Inject raw timeline details for downstream masking in Retake Mode
        guide_data["timeline_data"] = timeline_data
        guide_data["start_frame"] = start_frame
        guide_data["duration_frames"] = duration_frames
        guide_data["resize_method"] = resize_method

        return io.NodeOutput(patched, conditioning, latent, audio_latent, guide_data, motion_guide_data, float(frame_rate), audio_out)


NODE_CLASS_MAPPINGS = {
    "CSLTXDirector": CSLTXDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSLTXDirector": "CS-LTX 2.0 宫格导演台",
}
