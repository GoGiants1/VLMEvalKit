import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .aki import AKI
from .aria import Aria
from .aya_vision import AyaVision
from .base import BaseModel
from .bunnyllama3 import BunnyLLama3
from .cambrian import Cambrian
from .chameleon import Chameleon
from .cogvlm import CogVlm, GLM4v
from .deepseek_vl import DeepSeekVL
from .deepseek_vl2 import DeepSeekVL2
from .eagle_x import Eagle
from .emu import Emu, Emu3_chat, Emu3_gen
from .falcon_vlm import Falcon2VLM
from .gemma import Gemma3, PaliGemma
from .h2ovl_mississippi import H2OVLChat
from .idefics import IDEFICS, IDEFICS2
from .instructblip import InstructBLIP
from .internvl import InternVLChat
from .janus import Janus
from .kimi_vl import KimiVL
from .kosmos import Kosmos2
from .llama4 import llama4
from .llama_vision import llama_vision
from .llava import (
    LLaVA,
    LLaVA_Next,
    LLaVA_Next2,
    LLaVA_OneVision,
    LLaVA_OneVision_HF,
    LLaVA_XTuner,
)
from .long_vita import LongVITA
from .mantis import Mantis
from .mgm import Mini_Gemini
from .minicpm_v import MiniCPM_Llama3_V, MiniCPM_o_2_6, MiniCPM_V, MiniCPM_V_2_6
from .minigpt4 import MiniGPT4
from .minimonkey import MiniMonkey
from .mixsense import LLama3Mixsense
from .mmalaya import MMAlaya, MMAlaya2
from .molmo import molmo
from .monkey import Monkey, MonkeyChat
from .moondream import Moondream1, Moondream2
from .mplug_owl2 import mPLUG_Owl2
from .mplug_owl3 import mPLUG_Owl3
from .nvlm import NVLM
from .ola import Ola
from .omchat import OmChat
from .omnilmm import OmniLMM12B
from .open_flamingo import OpenFlamingo
from .ovis import Ovis, Ovis1_6, Ovis1_6_Plus, Ovis2
from .pandagpt import PandaGPT
from .parrot import Parrot
from .phi3_vision import Phi3_5Vision, Phi3Vision
from .phi4_multimodal import Phi4Multimodal
from .pixtral import Pixtral
from .points import POINTS, POINTSV15
from .qh_360vl import QH_360VL
from .qwen2_vl import Qwen2VLChat
from .qwen_vl import QwenVL, QwenVLChat
from .rbdash import RBDash
from .ristretto import Ristretto
from .ross import Ross
from .sail_vl import SailVL
from .slime import SliME
from .smolvlm import SmolVLM, SmolVLM2
from .transcore_m import TransCoreM
from .ursa import UrsaChat
from .valley import ValleyEagleChat
from .video_llm import (
    Chatunivi,
    LLaMAVID,
    PLLaVA,
    VideoChat2_HD,
    VideoChatGPT,
    VideoLLaVA,
    VideoLLaVA_HF,
)
from .vila import NVILA, VILA
from .vintern_chat import VinternChat
from .visualglm import VisualGLM
from .vita import VITA, VITAQwen2
from .vlaa_thinker import VLAAThinkerChat
from .vlm_r1 import VLMR1Chat
from .vxverse import VXVERSE
from .wemm import WeMM
from .xcomposer import (
    ShareCaptioner,
    XComposer,
    XComposer2,
    XComposer2_4KHD,
    XComposer2d5,
)
from .xgen_mm import XGenMM
from .yi_vl import Yi_VL
