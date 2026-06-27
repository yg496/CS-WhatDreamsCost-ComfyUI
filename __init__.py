from .ltx_keyframer import CSLTXKeyframer
from .multi_image_loader import CSMultiImageLoader
from .ltx_sequencer import CSLTXSequencer
from .speech_length_calculator import CSSpeechLengthCalculator
from .load_audio_ui import CSLoadAudioUI
from .load_video_ui import CSLoadVideoUI
from .ltx_director import CSLTXDirector
from .ltx_director_guide import CSLTXDirectorGuide, CSLTXDirectorCropGuides

NODE_CLASS_MAPPINGS = {
    "CSLTXKeyframer": CSLTXKeyframer,
    "CSMultiImageLoader": CSMultiImageLoader,
    "CSLTXSequencer": CSLTXSequencer,
    "CSSpeechLengthCalculator": CSSpeechLengthCalculator,
    "CSLoadAudioUI": CSLoadAudioUI,
    "CSLoadVideoUI": CSLoadVideoUI,
    "CSLTXDirector": CSLTXDirector,
    "CSLTXDirectorGuide": CSLTXDirectorGuide,
    "CSLTXDirectorCropGuides": CSLTXDirectorCropGuides,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSLTXKeyframer": "CS-LTX 关键帧",
    "CSMultiImageLoader": "CS 多图加载器",
    "CSLTXSequencer": "CS-LTX 序列引导",
    "CSSpeechLengthCalculator": "CS 语音时长计算",
    "CSLoadAudioUI": "CS 音频加载器",
    "CSLoadVideoUI": "CS 视频加载器",
    "CSLTXDirector": "CS-LTX 2.0 宫格导演台",
    "CSLTXDirectorGuide": "CS-LTX 导演台引导",
    "CSLTXDirectorCropGuides": "CS-LTX 裁剪引导帧",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
