import os
import re
import sys
import logging
import random
import tempfile
import threading
from typing import Callable, Optional, Tuple
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
# Librosa imports Numba during Parakeet ASR setup. Keep Numba's cache rooted in
# the project so launching app.py from another directory cannot stall startup.
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import gradio as gr
from funasr import AutoModel
import voxcpm
from voxcpm.model.utils import resolve_runtime_device

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    "`Target Text`: *\"I never asked you to stay… It's not like I care or anything. "
    "But… why does it still hurt so much now that you're gone?\"*  \n\n"
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "seed_label": "Seed",
        "seed_info": "Seed used for reproducible generation. Updated with the actual successful seed after generation.",
        "random_seed_label": "Random Seed",
        "random_seed_info": "Generate a new seed before each inference run.",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,  # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, " "designed to generate highly realistic speech."
)

ASR_BACKENDS = {"auto", "sensevoice", "parakeet"}
PARAKEET_ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_LOCAL_MODEL_DIRNAME = PARAKEET_ASR_MODEL_ID.replace("/", "__")
ProgressCallback = Optional[Callable[[float, str], None]]
GenerationProgressCallback = Optional[Callable[[int, int], None]]

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------


def _coerce_audio_filepath(audio_input) -> Optional[str]:
    if audio_input is None or audio_input == "":
        return None
    if isinstance(audio_input, (str, os.PathLike)):
        return os.fspath(audio_input)
    if isinstance(audio_input, dict):
        path = audio_input.get("path")
        return os.fspath(path) if path else None
    path = getattr(audio_input, "path", None)
    if path:
        return os.fspath(path)
    return str(audio_input)


def _extract_asr_text(result) -> str:
    if not result:
        return ""
    first = result[0] if isinstance(result, list) else result
    raw_text = first.get("text", "") if isinstance(first, dict) else str(first)
    return re.sub(r"<\|.*?\|>", "", raw_text).strip()


def _extract_parakeet_asr_text(result) -> str:
    if not result:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (list, tuple)):
        return " ".join(str(item).strip() for item in result if str(item).strip()).strip()
    return str(result).strip()


def _prepare_asr_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[str, Optional[str]]:
    """Return an ASR-friendly 16 kHz mono file and optional temp path to remove."""
    import librosa
    import soundfile as sf

    source_path = os.fspath(audio_path)
    try:
        info = sf.info(source_path)
        suffix = Path(source_path).suffix.lower()
        if info.samplerate == sample_rate and info.channels == 1 and suffix in {".wav", ".flac"}:
            return source_path, None
    except Exception:
        pass

    audio, _ = librosa.load(source_path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError("Reference audio contains no readable samples.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
    sf.write(temp_path, audio, sample_rate, subtype="PCM_16")
    return temp_path, temp_path


def _normalize_asr_backend(asr_backend: str) -> str:
    backend = (asr_backend or "auto").strip().lower()
    if backend not in ASR_BACKENDS:
        raise ValueError(f"Unknown ASR backend: {asr_backend!r}. Expected one of: {', '.join(sorted(ASR_BACKENDS))}.")
    return backend


def _emit_progress(callback: ProgressCallback, fraction: float, message: str) -> None:
    if callback is None:
        return
    callback(max(0.0, min(1.0, fraction)), message)


def _resolve_generation_inputs(
    demo,
    ref_wav,
    use_prompt_text: bool,
    prompt_text_value: str,
    control_instruction: str,
    progress_callback: ProgressCallback = None,
) -> Tuple[Optional[str], str, str]:
    audio_path = _coerce_audio_filepath(ref_wav)
    actual_prompt_text = (prompt_text_value or "").strip() if use_prompt_text else ""
    if use_prompt_text:
        if not audio_path:
            raise gr.Error("Upload reference audio before using Ultimate Cloning Mode.")
        if not actual_prompt_text:
            logger.info("Auto-transcribing reference audio before generation...")
            actual_prompt_text = demo.prompt_wav_recognition(audio_path, progress_callback=progress_callback).strip()
        if not actual_prompt_text:
            raise gr.Error(
                "Auto-transcription returned no text. Enter the reference transcript or disable Ultimate Cloning Mode."
            )
        return audio_path, actual_prompt_text, ""
    return audio_path, "", control_instruction


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto", asr_backend: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")
        self.asr_backend = _normalize_asr_backend(os.environ.get("VOXCPM_ASR_BACKEND", asr_backend))

        project_root = Path(__file__).parent
        local_asr_model = project_root / "models" / "iic__SenseVoiceSmall"
        local_parakeet_model = project_root / "models" / PARAKEET_LOCAL_MODEL_DIRNAME
        local_zipenhancer_model = project_root / "models" / "iic__speech_zipenhancer_ans_multiloss_16k_base"

        self.asr_model_id = str(local_asr_model) if local_asr_model.exists() else "iic/SenseVoiceSmall"
        self.parakeet_model_id = str(local_parakeet_model) if local_parakeet_model.exists() else None
        self.zipenhancer_model_id = (
            str(local_zipenhancer_model)
            if local_zipenhancer_model.exists()
            else "iic/speech_zipenhancer_ans_multiloss_16k_base"
        )
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None
        self.parakeet_processor = None
        self.parakeet_model = None
        self._voxcpm_load_lock = threading.RLock()
        self._asr_load_lock = threading.RLock()
        self._parakeet_load_lock = threading.RLock()
        logger.info("ASR backend: %s", self._resolved_asr_backend_name())

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def _get_load_lock(self, attr_name: str):
        lock = getattr(self, attr_name, None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, attr_name, lock)
        return lock

    def asr_status_text(self) -> str:
        backend = self._resolved_asr_backend_name()
        if backend == "parakeet":
            model_name = "NVIDIA Parakeet TDT 0.6B v3"
            model_path = self.parakeet_model_id or "not installed"
            device = "cuda" if self.device.startswith("cuda") else "cpu"
        else:
            model_name = "SenseVoiceSmall"
            model_path = self.asr_model_id
            device = self.asr_device
        return f"ASR: {model_name} | language: auto-detect | device: {device} | model: {model_path}"

    def preload_models(
        self, *, preload_asr: bool = True, preload_tts: bool = True, preload_denoiser: bool = True
    ) -> None:
        logger.info("Preloading models...")
        if preload_tts:
            logger.info("Preloading VoxCPM TTS model...")
            current_model = self.get_or_load_voxcpm()
            if preload_denoiser:
                logger.info("Preloading ZipEnhancer denoiser...")
                current_model._get_or_load_denoiser()
        if preload_asr:
            if self._should_use_parakeet_asr():
                logger.info("Preloading Parakeet ASR model (language=auto-detect)...")
                self.get_or_load_parakeet_asr_model()
                if self.asr_backend == "auto":
                    logger.info("Preloading SenseVoice ASR fallback (language=auto)...")
                    self.get_or_load_asr_model()
            else:
                logger.info("Preloading SenseVoice ASR model (language=auto)...")
                self.get_or_load_asr_model()
        logger.info("Preload complete.")

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        with self._get_load_lock("_voxcpm_load_lock"):
            if self.voxcpm_model is not None:
                return self.voxcpm_model
            logger.info(f"Loading model: {self._model_id}")
            self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
                self._model_id,
                zipenhancer_model_id=self.zipenhancer_model_id,
                optimize=self.optimize,
                device=self.device,
            )
            logger.info("Model loaded successfully.")
            return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        with self._get_load_lock("_asr_load_lock"):
            if self.asr_model is not None:
                return self.asr_model
            logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
            self.asr_model = AutoModel(
                model=self.asr_model_id,
                disable_update=True,
                log_level="DEBUG",
                device=self.asr_device,
            )
            logger.info("ASR model loaded successfully.")
            return self.asr_model

    def _should_use_parakeet_asr(self) -> bool:
        if self.asr_backend == "sensevoice":
            return False
        if self.asr_backend == "parakeet":
            return True
        return self.device.startswith("cuda") and self.parakeet_model_id is not None

    def _resolved_asr_backend_name(self) -> str:
        if self._should_use_parakeet_asr():
            return "parakeet"
        return "sensevoice"

    def get_or_load_parakeet_asr_model(self):
        if self.parakeet_processor is not None and self.parakeet_model is not None:
            return self.parakeet_processor, self.parakeet_model
        with self._get_load_lock("_parakeet_load_lock"):
            if self.parakeet_processor is not None and self.parakeet_model is not None:
                return self.parakeet_processor, self.parakeet_model
            if self.parakeet_model_id is None:
                raise RuntimeError(
                    "NVIDIA Parakeet ASR is not installed locally. Run install.bat to pre-download it, "
                    "or start app.py with --asr-backend sensevoice."
                )
            try:
                import torch
                from transformers import AutoModelForTDT, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "NVIDIA Parakeet ASR requires a Transformers build with AutoModelForTDT support."
                ) from exc

            target_device = "cuda" if self.device.startswith("cuda") else "cpu"
            logger.info("Loading Parakeet ASR model: %s on device: %s", self.parakeet_model_id, target_device)
            self.parakeet_processor = AutoProcessor.from_pretrained(self.parakeet_model_id, local_files_only=True)
            self.parakeet_model = AutoModelForTDT.from_pretrained(
                self.parakeet_model_id,
                dtype="auto",
                local_files_only=True,
            )
            self.parakeet_model.to(target_device)
            self.parakeet_model.eval()
            logger.info("Parakeet ASR model loaded successfully.")
            return self.parakeet_processor, self.parakeet_model

    def _recognize_with_sensevoice(self, asr_audio_path: str, progress_callback: ProgressCallback = None) -> str:
        _emit_progress(progress_callback, 0.45, "Transcribing reference audio with SenseVoice, language auto, 45%")
        logger.info("Running SenseVoice ASR with language=auto on device: %s", self.asr_device)
        res = self.get_or_load_asr_model().generate(
            input=asr_audio_path,
            language="auto",
            use_itn=True,
        )
        _emit_progress(progress_callback, 0.95, "Transcribing reference audio, 95%")
        return _extract_asr_text(res)

    def _recognize_with_parakeet(self, asr_audio_path: str, progress_callback: ProgressCallback = None) -> str:
        import librosa
        import torch

        _emit_progress(progress_callback, 0.25, "Loading Parakeet ASR, language auto-detect, 25%")
        processor, model = self.get_or_load_parakeet_asr_model()
        sample_rate = getattr(processor.feature_extractor, "sampling_rate", 16000)
        _emit_progress(progress_callback, 0.40, "Preparing Parakeet audio features, 40%")
        audio, _ = librosa.load(asr_audio_path, sr=sample_rate, mono=True)
        if audio.size == 0:
            return ""
        inputs = processor([audio], sampling_rate=sample_rate)
        inputs.to(model.device, dtype=model.dtype)
        logger.info("Running Parakeet ASR with language=auto-detect on device: %s", model.device)
        _emit_progress(progress_callback, 0.55, "Transcribing reference audio with Parakeet, 55%")
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        sequences = getattr(output, "sequences", output)
        _emit_progress(progress_callback, 0.95, "Transcribing reference audio, 95%")
        return _extract_parakeet_asr_text(processor.decode(sequences, skip_special_tokens=True))

    def prompt_wav_recognition(self, prompt_wav: Optional[str], progress_callback: ProgressCallback = None) -> str:
        prompt_wav_path = _coerce_audio_filepath(prompt_wav)
        if prompt_wav_path is None:
            return ""
        _emit_progress(progress_callback, 0.05, "Transcribing reference audio, 5%")
        asr_audio_path, temp_path = _prepare_asr_audio(prompt_wav_path)
        _emit_progress(progress_callback, 0.15, "Prepared 16 kHz mono ASR audio, 15%")
        try:
            if self._should_use_parakeet_asr():
                try:
                    return self._recognize_with_parakeet(asr_audio_path, progress_callback)
                except Exception:
                    if self.asr_backend == "parakeet":
                        raise
                    logger.warning("Parakeet ASR failed; falling back to SenseVoice.", exc_info=True)
            return self._recognize_with_sensevoice(asr_audio_path, progress_callback)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
        progress_callback: GenerationProgressCallback = None,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if progress_callback is not None:
            generate_kwargs["progress_callback"] = progress_callback
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
        progress_callback: GenerationProgressCallback = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = _coerce_audio_filepath(reference_wav_path_input)
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
            progress_callback=progress_callback,
        )
        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[PROJECT_ROOT / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _gradio_progress_callback(progress):
        return lambda fraction, message: progress(fraction, desc=message)

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
        progress=gr.Progress(track_tqdm=True),
    ):
        progress(0.02, desc="Preparing generation, 2%")

        def asr_progress(fraction: float, message: str) -> None:
            mapped = 0.03 + (0.24 * max(0.0, min(1.0, fraction)))
            progress(mapped, desc=f"{message} / preparing generation, {int(mapped * 100)}%")

        audio_path, actual_prompt_text, actual_control = _resolve_generation_inputs(
            demo,
            ref_wav,
            use_prompt_text,
            prompt_text_value,
            control_instruction,
            progress_callback=asr_progress,
        )
        seed = _coerce_seed(seed_value)

        def tts_progress(step: int, total: int) -> None:
            if total <= 0:
                return
            fraction = min(1.0, max(0.0, (step + 1) / total))
            mapped = 0.35 + (0.55 * fraction)
            progress(mapped, desc=f"Synthesising speech, {int(mapped * 100)}%")

        progress(0.30, desc="Preparing voice prompt, 30%")
        sr, wav_np, last_successful_seed = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=audio_path,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            seed=seed,
            progress_callback=tts_progress,
        )
        progress(0.95, desc="Finalising audio, 95%")
        progress(1.0, desc="Complete, 100%")
        return (sr, wav_np), last_successful_seed, actual_prompt_text if use_prompt_text else gr.update()

    def _on_toggle_instant(checked, current_prompt_text, audio_path):
        """Instant UI toggle — no ASR, no blocking."""
        current_prompt_text = current_prompt_text or ""
        if checked:
            placeholder = (
                "Recognizing reference audio..."
                if _coerce_audio_filepath(audio_path) and not current_prompt_text.strip()
                else I18N("prompt_text_placeholder")
            )
            return (
                gr.update(visible=True, value=current_prompt_text, placeholder=placeholder),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _on_reference_audio_change(checked, current_prompt_text, audio_path):
        if not checked or not _coerce_audio_filepath(audio_path):
            return gr.update()
        return gr.update(
            visible=True,
            value=current_prompt_text or "",
            placeholder="Recognizing reference audio...",
        )

    def _run_asr_if_needed(checked, audio_path, progress=gr.Progress(track_tqdm=True)):
        """Run ASR after the UI has updated. Only when toggled ON."""
        audio_file = _coerce_audio_filepath(audio_path)
        if not checked or not audio_file:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio using %s...", demo.asr_status_text())
            asr_text = demo.prompt_wav_recognition(
                audio_file,
                progress_callback=_gradio_progress_callback(progress),
            )
            logger.info("ASR result: %r", asr_text[:60])
            if not asr_text:
                progress(1.0, desc="Transcribing reference audio complete, 100%")
                return gr.update(
                    value="",
                    placeholder="No speech was recognized. Enter the reference transcript manually.",
                )
            progress(1.0, desc="Transcribing reference audio complete, 100%")
            return gr.update(value=asr_text, placeholder=I18N("prompt_text_placeholder"))
        except Exception as e:
            logger.warning("ASR recognition failed: %s", e, exc_info=True)
            return gr.update(value="", placeholder=f"ASR failed: {e}")

    def _ensure_prompt_text_before_generate(
        ref_wav, use_prompt_text, prompt_text_value, progress=gr.Progress(track_tqdm=True)
    ):
        if not use_prompt_text:
            return gr.update()
        progress(0.02, desc="Preparing reference transcript, 2%")
        audio_path, actual_prompt_text, _ = _resolve_generation_inputs(
            demo,
            ref_wav,
            True,
            prompt_text_value,
            "",
            progress_callback=_gradio_progress_callback(progress),
        )
        if not audio_path:
            raise gr.Error("Upload reference audio before using Ultimate Cloning Mode.")
        progress(1.0, desc="Reference transcript ready, 100%")
        return gr.update(value=actual_prompt_text, placeholder=I18N("prompt_text_placeholder"))

    with gr.Blocks(theme=_APP_THEME, css=_CUSTOM_CSS) as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                gr.Markdown(demo.asr_status_text())
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )
                    with gr.Row():
                        seed_value = gr.Number(
                            value=random.randint(0, 2**32 - 1),
                            precision=0,
                            label=I18N("seed_label"),
                            info=I18N("seed_info"),
                            interactive=False,
                        )
                        random_seed = gr.Checkbox(
                            value=True,
                            label=I18N("random_seed_label"),
                            elem_classes=["switch-toggle"],
                            info=I18N("random_seed_info"),
                        )

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text, prompt_text, reference_wav],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        reference_wav.change(
            fn=_on_reference_audio_change,
            inputs=[show_prompt_text, prompt_text, reference_wav],
            outputs=[prompt_text],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        random_seed.change(
            fn=_on_random_seed_toggle,
            inputs=[random_seed],
            outputs=[seed_value],
        )

        run_btn.click(
            fn=_prepare_seed,
            inputs=[random_seed, seed_value],
            outputs=[seed_value],
            show_progress=False,
        ).then(
            fn=_ensure_prompt_text_before_generate,
            inputs=[reference_wav, show_prompt_text, prompt_text],
            outputs=[prompt_text],
        ).then(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                seed_value,
            ],
            outputs=[audio_output, seed_value, prompt_text],
            show_progress=True,
            api_name="generate",
        )

    return interface


def run_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
    asr_backend: str = "auto",
    preload: bool = True,
    preload_denoiser: bool = True,
    open_browser: bool = True,
):
    demo = VoxCPMDemo(model_id=model_id, device=device, asr_backend=asr_backend)
    if preload:
        demo.preload_models(preload_asr=True, preload_tts=True, preload_denoiser=preload_denoiser)
    interface = create_demo_interface(demo)
    logger.info("Launching web UI at http://%s:%s", server_name, server_port)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        inbrowser=open_browser,
        i18n=I18N,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host/interface (default: 127.0.0.1; use 0.0.0.0 for LAN access)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--asr-backend",
        type=str,
        default="auto",
        choices=sorted(ASR_BACKENDS),
        help="Reference audio transcription backend: auto, sensevoice, or parakeet (default: auto)",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip loading models before launching the web UI.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the web UI without opening a browser window.",
    )
    parser.add_argument(
        "--preload-denoiser",
        action="store_true",
        default=True,
        help="Deprecated: ZipEnhancer is loaded before launch by default.",
    )
    parser.add_argument(
        "--no-preload-denoiser",
        action="store_false",
        dest="preload_denoiser",
        help="Skip loading ZipEnhancer before launching the web UI.",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
        asr_backend=args.asr_backend,
        preload=not args.no_preload,
        preload_denoiser=args.preload_denoiser,
        open_browser=not args.no_browser,
    )
