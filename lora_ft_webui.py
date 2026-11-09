import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import torch
import yaml

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Default pretrained model path: prefer VoxCPM2 if it exists, fallback to VoxCPM1.5
_v2_path = project_root / "models" / "openbmb__VoxCPM2"
_v15_path = project_root / "models" / "openbmb__VoxCPM1.5"
default_pretrained_path = str(_v2_path) if _v2_path.exists() else (str(_v15_path) if _v15_path.exists() else "openbmb/VoxCPM2")

import numpy as np
from funasr import AutoModel

# --- Localization ---
from i18n import DEFAULT_LANG, STRINGS, t
from voxcpm.core import VoxCPM, resolve_model_path
from voxcpm.model.voxcpm import LoRAConfig

# Global variables
current_model: VoxCPM | None = None
asr_model: AutoModel | None = None
training_process: subprocess.Popen | None = None
training_log = ""


def get_timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_sample_rate(pretrained_path: str) -> int | None:
    """Read audio_vae_config.sample_rate from the model's config.json.

    This is the AudioVAE *encoder* input rate, which is the correct rate for
    resampling training data.  Returns None when detection fails.
    """
    config_file = os.path.join(pretrained_path, "config.json")
    if not os.path.isfile(config_file):
        return None
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["audio_vae_config"]["sample_rate"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Warning: failed to detect sample_rate from {config_file}: {e}", file=sys.stderr)
        return None


def get_or_load_asr_model():
    global asr_model
    if asr_model is None:
        print("Loading ASR model (SenseVoiceSmall)...", file=sys.stderr)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level="ERROR",
            device=device,
        )
    return asr_model


def recognize_audio(audio_path):
    if not audio_path:
        return ""
    try:
        model = get_or_load_asr_model()
        res = model.generate(input=audio_path, language="auto", use_itn=True)
        text = res[0]["text"].split("|>")[-1]
        return text
    except Exception as e:
        print(f"ASR Error: {e}", file=sys.stderr)
        return ""


def scan_lora_checkpoints(root_dir="lora", with_info=False):
    """
    Scans for LoRA checkpoints in the lora directory.

    Args:
        root_dir: Directory to scan for LoRA checkpoints
        with_info: If True, returns list of (path, base_model) tuples

    Returns:
        List of checkpoint paths, or list of (path, base_model) tuples if with_info=True
    """
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # Look for lora_weights.safetensors recursively
    for root, dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            # Use the relative path from root_dir as the ID
            rel_path = os.path.relpath(root, root_dir)

            if with_info:
                # Try to read base_model from lora_config.json
                base_model = None
                lora_config_file = os.path.join(root, "lora_config.json")
                if os.path.exists(lora_config_file):
                    try:
                        with open(lora_config_file, "r", encoding="utf-8") as f:
                            lora_info = json.load(f)
                        base_model = lora_info.get("base_model", "Unknown")
                    except (json.JSONDecodeError, OSError):
                        pass
                checkpoints.append((rel_path, base_model))
            else:
                checkpoints.append(rel_path)

    # Also check for checkpoints in the default location if they exist
    default_ckpt = "checkpoints/finetune_lora"
    if os.path.exists(os.path.join(root_dir, default_ckpt)):
        # This might be covered by the walk, but good to be sure
        pass

    return sorted(checkpoints, reverse=True)


def load_lora_config_from_checkpoint(lora_path):
    """Load LoRA config from lora_config.json if available."""
    lora_config_file = os.path.join(lora_path, "lora_config.json")
    if os.path.exists(lora_config_file):
        try:
            with open(lora_config_file, "r", encoding="utf-8") as f:
                lora_info = json.load(f)
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
        except Exception as e:
            print(f"Warning: Failed to load lora_config.json: {e}", file=sys.stderr)
    return None, None


def get_default_lora_config():
    """Return default LoRA config for hot-swapping support."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def load_model(pretrained_path, lora_path=None):
    global current_model
    print(f"Loading model from {pretrained_path}...", file=sys.stderr)

    lora_config = None
    lora_weights_path = None

    if lora_path:
        full_lora_path = os.path.join("lora", lora_path)
        if os.path.exists(full_lora_path):
            lora_weights_path = full_lora_path
            # Try to load LoRA config from lora_config.json
            lora_config, _ = load_lora_config_from_checkpoint(full_lora_path)
            if lora_config:
                print(f"Loaded LoRA config from {full_lora_path}/lora_config.json", file=sys.stderr)
            else:
                # Fallback to default config for old checkpoints
                lora_config = get_default_lora_config()
                print("Using default LoRA config (lora_config.json not found)", file=sys.stderr)

    # Always init with a default LoRA config to allow hot-swapping later
    if lora_config is None:
        lora_config = get_default_lora_config()

    current_model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=False,
        lora_config=lora_config,
        lora_weights_path=lora_weights_path,
    )
    return "Model loaded successfully!"


def run_inference(text, prompt_wav, prompt_text, lora_selection, cfg_scale, steps, seed, pretrained_path=None):
    # 如果选择了 LoRA 模型且当前模型未加载，尝试从 LoRA config 读取 base_model
    if current_model is None:
        # 优先使用用户指定的预训练模型路径
        base_model_path = pretrained_path if pretrained_path and pretrained_path.strip() else default_pretrained_path

        # 如果选择了 LoRA，尝试从其 config 读取 base_model
        if lora_selection and lora_selection != "None":
            full_lora_path = os.path.join("lora", lora_selection)
            lora_config_file = os.path.join(full_lora_path, "lora_config.json")

            if os.path.exists(lora_config_file):
                try:
                    with open(lora_config_file, "r", encoding="utf-8") as f:
                        lora_info = json.load(f)
                    saved_base_model = lora_info.get("base_model")

                    if saved_base_model:
                        # 优先使用保存的 base_model 路径
                        if os.path.exists(saved_base_model):
                            base_model_path = saved_base_model
                            print(f"Using base model from LoRA config: {base_model_path}", file=sys.stderr)
                        else:
                            print(f"Warning: Saved base_model path not found: {saved_base_model}", file=sys.stderr)
                            print(f"Falling back to default: {base_model_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Failed to read base_model from LoRA config: {e}", file=sys.stderr)

        # 加载模型
        lora_to_load = lora_selection if lora_selection and lora_selection != "None" else None
        try:
            print(f"Loading base model: {base_model_path}", file=sys.stderr)
            load_model(base_model_path, lora_to_load)
            if lora_to_load:
                print(f"Model loaded with LoRA: {lora_selection}", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to load model from {base_model_path}: {e!s}"
            print(error_msg, file=sys.stderr)
            return None, error_msg
        lora_just_loaded = lora_to_load
    else:
        lora_just_loaded = None

    # Handle LoRA hot-swapping
    assert current_model is not None, "Model must be loaded before inference"
    if lora_selection and lora_selection != "None":
        full_lora_path = os.path.join("lora", lora_selection)

        if lora_just_loaded != lora_selection:
            new_lora_config, new_base_model = load_lora_config_from_checkpoint(full_lora_path)
            current_r = current_model.tts_model.lora_config.r if current_model.tts_model.lora_config else None
            new_r = new_lora_config.r if new_lora_config else None

            if new_r is not None and current_r is not None and new_r != current_r:
                print(f"LoRA rank mismatch (model r={current_r}, checkpoint r={new_r}), reloading...", file=sys.stderr)
                reload_base = (
                    new_base_model
                    if new_base_model and os.path.exists(new_base_model)
                    else (pretrained_path if pretrained_path and pretrained_path.strip() else default_pretrained_path)
                )
                try:
                    load_model(reload_base, lora_selection)
                except Exception as e:
                    return None, f"Failed to reload model for LoRA rank change: {e}"
            else:
                print(f"Hot-loading LoRA: {full_lora_path}", file=sys.stderr)
                try:
                    current_model.load_lora(full_lora_path)
                except Exception as e:
                    print(f"Error loading LoRA: {e}", file=sys.stderr)
                    return None, f"Error loading LoRA: {e}"
        current_model.set_lora_enabled(True)
    else:
        print("Disabling LoRA", file=sys.stderr)
        current_model.set_lora_enabled(False)

    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 处理 prompt 参数：必须同时为 None 或同时有值
    final_prompt_wav = None
    final_prompt_text = None

    if prompt_wav and prompt_wav.strip():
        # 有参考音频
        final_prompt_wav = prompt_wav

        # 如果没有提供参考文本，尝试自动识别
        if not prompt_text or not prompt_text.strip():
            print(t("log_ref_audio_no_text"), file=sys.stderr)
            try:
                final_prompt_text = recognize_audio(prompt_wav)
                if final_prompt_text:
                    print(t("log_auto_recognized").format(text=final_prompt_text), file=sys.stderr)
                else:
                    return None, t("error_asr_failed")
            except Exception as e:
                return None, t("error_asr_exception").format(error=str(e))
        else:
            final_prompt_text = prompt_text.strip()
    # 如果没有参考音频，两个都设为 None（用于零样本 TTS）

    try:
        audio_np = current_model.generate(
            text=text,
            prompt_wav_path=final_prompt_wav,
            prompt_text=final_prompt_text,
            cfg_value=cfg_scale,
            inference_timesteps=steps,
            denoise=False,
        )
        return (current_model.tts_model.sample_rate, audio_np), "Generation Success"
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {e!s}"


def start_training(
    pretrained_path,
    train_manifest,
    val_manifest,
    learning_rate,
    num_iters,
    batch_size,
    lora_rank,
    lora_alpha,
    save_interval,
    output_name="",
    # Advanced options
    grad_accum_steps=1,
    num_workers=2,
    log_interval=10,
    valid_interval=1000,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=None,
    sample_rate=16000,
    max_grad_norm=1.0,
    # LoRA advanced
    enable_lm=True,
    enable_dit=True,
    enable_proj=False,
    dropout=0.0,
    tensorboard_path="",
    # Distribution options
    hf_model_id="",
    distribute=False,
):
    global training_log

    if training_process is not None and training_process.poll() is None:
        return "Training is already running!"

    if output_name and output_name.strip():
        timestamp = output_name.strip()
    else:
        timestamp = get_timestamp_str()

    save_dir = os.path.join("lora", timestamp)
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Resolve HuggingFace Hub IDs to local paths before reading config
    pretrained_path = resolve_model_path(pretrained_path)

    # Auto-detect sample_rate from model config.json to prevent mismatch
    detected_sr = detect_sample_rate(pretrained_path)
    if detected_sr is not None:
        if int(sample_rate) != detected_sr:
            training_log += (
                f"[Auto-fix] sample_rate changed from {int(sample_rate)} to {detected_sr} "
                f"(read from {pretrained_path}/config.json audio_vae_config.sample_rate)\n"
            )
        sample_rate = detected_sr

    # Create config dictionary
    # Resolve max_steps default
    resolved_max_steps = int(max_steps) if max_steps not in (None, "", 0) else int(num_iters)

    # Auto-detect out_sample_rate from model config
    out_sample_rate = 0
    config_file = os.path.join(pretrained_path, "config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out_sr = cfg.get("audio_vae_config", {}).get("out_sample_rate")
            if out_sr:
                out_sample_rate = int(out_sr)
        except Exception:
            pass

    config = {
        "pretrained_path": pretrained_path,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "sample_rate": int(sample_rate),
        "out_sample_rate": out_sample_rate,
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "num_workers": int(num_workers),
        "num_iters": int(num_iters),
        "log_interval": int(log_interval),
        "valid_interval": int(valid_interval),
        "save_interval": int(save_interval),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_steps": int(warmup_steps),
        "max_steps": resolved_max_steps,
        "max_grad_norm": float(max_grad_norm),
        "save_path": checkpoints_dir,
        "tensorboard": tensorboard_path if tensorboard_path else logs_dir,
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": bool(enable_lm),
            "enable_dit": bool(enable_dit),
            "enable_proj": bool(enable_proj),
            "r": int(lora_rank),
            "alpha": int(lora_alpha),
            "dropout": float(dropout),
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
    }

    # Add distribution options if provided
    if hf_model_id and hf_model_id.strip():
        config["hf_model_id"] = hf_model_id.strip()
    if distribute:
        config["distribute"] = True

    config_path = os.path.join(save_dir, "train_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    cmd = [sys.executable, "scripts/train_voxcpm_finetune.py", "--config_path", config_path]

    training_log = f"Starting training...\nConfig saved to {config_path}\nOutput dir: {save_dir}\n"

    def run_process():
        global training_process, training_log
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        assert training_process.stdout is not None
        for line in training_process.stdout:
            print(line, end="", flush=True)  # Stream to stdout (Docker logs)
            training_log += line
            # Keep log size manageable
            if len(training_log) > 100000:
                training_log = training_log[-100000:]

        training_process.wait()
        training_log += f"\nTraining finished with code {training_process.returncode}"

    threading.Thread(target=run_process, daemon=True).start()

    return f"Training started! Check 'lora/{timestamp}'"


def get_training_log():
    return training_log


def stop_training():
    global training_log
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        training_log += "\nTraining terminated by user."
        return "Training stopped."
    return "No training running."


# --- GUI Layout ---

# 自定义CSS样式
custom_css = """
/* 整体主题样式 */
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* 标题区域样式 - 扁平化设计 */
.title-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    padding: 15px 25px;
    margin-bottom: 15px;
    border: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.title-section h1 {
    color: white;
    text-shadow: none;
    font-weight: 600;
    margin: 0;
    font-size: 28px;
    line-height: 1.2;
}

.title-section h3 {
    color: rgba(255, 255, 255, 0.9);
    font-weight: 400;
    margin-top: 5px;
    font-size: 14px;
    line-height: 1.3;
}

.title-section p {
    color: rgba(255, 255, 255, 0.85);
    font-size: 13px;
    margin: 5px 0 0 0;
    line-height: 1.3;
}

/* 标签页样式 */
.tabs {
    background: white;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* 按钮样式增强 */
.button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
}

.button-stop {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
}

.button-stop:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(250, 112, 154, 0.4);
}

.button-refresh {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 500;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 2px 10px rgba(132, 250, 176, 0.3);
}

.button-refresh:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(132, 250, 176, 0.4);
}

/* 表单区域样式 */
.form-section {
    background: white;
    border-radius: 20px;
    padding: 30px;
    margin: 15px 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}

/* 输入框样式 */
.input-field {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    padding: 12px 16px;
    transition: all 0.3s ease;
    background: #fafafa;
}

.input-field:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    background: white;
}

/* 滑块样式 */
.slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    outline: none;
    opacity: 0.8;
    transition: opacity 0.2s;
}

.slider:hover {
    opacity: 1;
}

.slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    cursor: pointer;
    border: 3px solid #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

.slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    cursor: pointer;
    border: 3px solid #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

/* 折叠面板样式 */
.accordion {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    overflow: hidden;
    background: white;
}

.accordion-header {
    background: linear-gradient(135deg, #f5f7fa 0%, #e3e7ed 100%);
    padding: 15px 20px;
    font-weight: 600;
    color: #333;
}

/* 状态显示样式 */
.status-success {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(132, 250, 176, 0.3);
}

.status-error {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
}

/* 语言切换按钮样式 - 扁平化 */
.lang-selector {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, 0.4);
}

.lang-selector label.gr-box {
    color: white !important;
    font-weight: 600;
    margin-bottom: 8px !important;
}

/* 单选按钮组样式 */
.lang-selector fieldset,
.lang-selector .gr-form {
    gap: 10px !important;
    display: flex !important;
}

/* 单选按钮容器 - 扁平化 (未选中状态 - 较浅的深色) */
.lang-selector label.gr-radio-label {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6)) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    color: white !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    cursor: pointer !important;
    margin: 0 4px !important;
}

/* 选中的单选按钮 - 扁平化 (更深的深色背景) */
.lang-selector input[type="radio"]:checked + label,
.lang-selector label.gr-radio-label:has(input:checked) {
    background: linear-gradient(135deg, #5568d3, #6b4c9a) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2) !important;
    transform: none !important;
}

/* 未选中的单选按钮悬停效果 - 扁平化 */
.lang-selector label.gr-radio-label:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.75), rgba(118, 75, 162, 0.75)) !important;
    border-color: rgba(255, 255, 255, 0.7) !important;
    transform: none !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
}

/* 隐藏原始的单选按钮圆点 */
.lang-selector input[type="radio"] {
    opacity: 0;
    position: absolute;
}

/* Gradio Radio 特定样式 - 扁平化 */
.lang-selector .wrap {
    gap: 8px !important;
}

.lang-selector .wrap > label {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6)) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    color: white !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.lang-selector .wrap > label.selected {
    background: linear-gradient(135deg, #5568d3, #6b4c9a) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2) !important;
}

/* 标签样式优化 */
label {
    color: #333;
    font-weight: 500;
    margin-bottom: 8px;
}

/* Markdown 标题样式 */
.markdown-text h4 {
    color: #667eea;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 10px;
}

/* 参数组件间距优化 */
.form-section > div {
    margin-bottom: 15px;
}

/* Slider 组件样式优化 */
.gr-slider {
    padding: 10px 0;
}

/* Number 输入框优化 */
.gr-number {
    max-width: 100%;
}

/* 按钮容器优化 */
.gr-button {
    min-height: 45px;
    font-size: 16px;
}

/* 三栏布局优化 */
#component-0 .gr-row {
    gap: 20px;
}

/* 生成按钮特殊样式 */
.button-primary.gr-button-lg {
    min-height: 55px;
    font-size: 18px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* 刷新按钮小尺寸 */
.button-refresh.gr-button-sm {
    min-height: 38px;
    font-size: 14px;
    margin-top: 5px;
    margin-bottom: 15px;
}

/* 信息提示文字样式 */
.gr-info {
    font-size: 13px;
    color: #666;
    margin-top: 5px;
}

/* 区域标题样式优化 */
.form-section h4 {
    color: #667eea;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f0f0;
}

.form-section strong {
    color: #667eea;
    font-size: 15px;
    display: block;
    margin: 15px 0 10px 0;
}
"""

with gr.Blocks(title="VoxCPM LoRA WebUI", theme=gr.themes.Soft(), css=custom_css) as app:

    # State for language
    lang_state = gr.State(DEFAULT_LANG)

    # 标题区域
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=3):
            title_md = gr.Markdown(f"""
            # 🎵 VoxCPM LoRA WebUI
            {t('header_title')}

            {t('header_subtitle')}
            """)
        with gr.Column(scale=1):
            lang_btn = gr.Radio(
                choices=["en", "zh"], value=DEFAULT_LANG, label="🌐 Language / 语言", elem_classes="lang-selector"
            )

    with gr.Tabs(elem_classes="tabs") as tabs:
        # === Training Tab ===
        with gr.Tab("🚀 " + t("tab_train")) as tab_train:
            train_setup_md = gr.Markdown(f"""
            {t('train_setup_title')}
            {t('train_setup_desc')}
            """)

            with gr.Row():
                with gr.Column(scale=2, elem_classes="form-section"):
                    base_config_md = gr.Markdown(t("base_config_header"))

                    train_pretrained_path = gr.Textbox(
                        label="📂 " + t("pretrained_path"), value=default_pretrained_path, elem_classes="input-field"
                    )
                    train_manifest = gr.Textbox(
                        label="📋 " + t("train_manifest"),
                        value="examples/train_data_example.jsonl",
                        elem_classes="input-field",
                    )
                    val_manifest = gr.Textbox(label="📊 " + t("val_manifest"), value="", elem_classes="input-field")

                    train_params_md = gr.Markdown(t("train_params_header"))

                    with gr.Row():
                        lr = gr.Number(label="📈 " + t("lr"), value=1e-4, elem_classes="input-field")
                        num_iters = gr.Number(
                            label="🔄 " + t("max_iters"), value=2000, precision=0, elem_classes="input-field"
                        )
                        batch_size = gr.Number(
                            label="📦 " + t("batch_size"), value=1, precision=0, elem_classes="input-field"
                        )

                    with gr.Row():
                        lora_rank = gr.Number(label="🎯 " + t("lora_rank"), value=32, precision=0, elem_classes="input-field")
                        lora_alpha = gr.Number(label="⚖️ " + t("lora_alpha"), value=16, precision=0, elem_classes="input-field")
                        save_interval = gr.Number(
                            label="💾 " + t("save_interval"), value=1000, precision=0, elem_classes="input-field"
                        )

                    output_name = gr.Textbox(
                        label="📁 " + t("output_name"), value="", elem_classes="input-field"
                    )

                    with gr.Row():
                        start_btn = gr.Button("▶️ " + t("start_train"), variant="primary", elem_classes="button-primary")
                        stop_btn = gr.Button("⏹️ " + t("stop_train"), variant="stop", elem_classes="button-stop")

                    with gr.Accordion(t("advanced_accordion"), open=False, elem_classes="accordion") as advanced_acc:
                        with gr.Row():
                            grad_accum_steps = gr.Number(label=t("grad_accum_steps"), value=1, precision=0)
                            num_workers = gr.Number(label=t("num_workers"), value=2, precision=0)
                            log_interval = gr.Number(label=t("log_interval"), value=10, precision=0)
                        with gr.Row():
                            valid_interval = gr.Number(label=t("valid_interval"), value=1000, precision=0)
                            weight_decay = gr.Number(label=t("weight_decay"), value=0.01)
                            warmup_steps = gr.Number(label=t("warmup_steps"), value=100, precision=0)
                        with gr.Row():
                            max_steps = gr.Number(label=t("max_steps"), value=0, precision=0)
                            sample_rate = gr.Number(label=t("sample_rate"), value=16000, precision=0)
                            max_grad_norm = gr.Number(label=t("max_grad_norm"), value=1.0)
                        with gr.Row():
                            tensorboard_path = gr.Textbox(label=t("tensorboard_path"), value="")
                            enable_lm = gr.Checkbox(label=t("enable_lm"), value=True)
                            enable_dit = gr.Checkbox(label=t("enable_dit"), value=True)
                        with gr.Row():
                            enable_proj = gr.Checkbox(label=t("enable_proj"), value=False)
                            dropout = gr.Number(label=t("dropout"), value=0.0)

                        distribution_md = gr.Markdown(t("distribution_header"))
                        with gr.Row():
                            hf_model_id = gr.Textbox(label=t("hf_model_id"), value="")
                            distribute = gr.Checkbox(label=t("distribute"), value=False)

                with gr.Column(scale=2, elem_classes="form-section"):
                    train_logs_md = gr.Markdown(t("train_logs_header"))
                    logs_out = gr.TextArea(
                        label="",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        elem_classes="input-field",
                        show_label=False,
                    )

            def on_pretrained_path_change(path):
                """Auto-detect sample_rate when pretrained model path changes."""
                sr = detect_sample_rate(path)
                if sr is not None:
                    return gr.update(value=sr)
                return gr.update()

            train_pretrained_path.change(
                on_pretrained_path_change,
                inputs=[train_pretrained_path],
                outputs=[sample_rate],
            )

            start_btn.click(
                start_training,
                inputs=[
                    train_pretrained_path,
                    train_manifest,
                    val_manifest,
                    lr,
                    num_iters,
                    batch_size,
                    lora_rank,
                    lora_alpha,
                    save_interval,
                    output_name,
                    # advanced
                    grad_accum_steps,
                    num_workers,
                    log_interval,
                    valid_interval,
                    weight_decay,
                    warmup_steps,
                    max_steps,
                    sample_rate,
                    max_grad_norm,
                    enable_lm,
                    enable_dit,
                    enable_proj,
                    dropout,
                    tensorboard_path,
                    # distribution
                    hf_model_id,
                    distribute,
                ],
                outputs=[logs_out],  # Initial message
            )
            stop_btn.click(stop_training, outputs=[logs_out])

            # Log refresher
            timer = gr.Timer(1)
            timer.tick(get_training_log, outputs=logs_out)

        # === Inference Tab ===
        with gr.Tab("🎵 " + t("tab_infer")) as tab_infer:
            infer_header_md = gr.Markdown(f"""
            {t('infer_title')}
            {t('infer_desc')}
            """)

            with gr.Row():
                # 左栏：输入配置 (35%)
                with gr.Column(scale=35, elem_classes="form-section"):
                    input_config_md = gr.Markdown(t("input_config_header"))

                    infer_text = gr.TextArea(
                        label="💬 " + t("text_to_synth"),
                        value="Hello, this is a test of the VoxCPM LoRA model.",
                        elem_classes="input-field",
                        lines=4,
                        placeholder=t("placeholder_synth"),
                    )

                    voice_clone_md = gr.Markdown(t("voice_clone_header"))

                    prompt_wav = gr.Audio(label="🎵 " + t("ref_audio"), type="filepath", elem_classes="input-field")

                    prompt_text = gr.Textbox(
                        label="📝 " + t("ref_text"),
                        elem_classes="input-field",
                        placeholder=t("placeholder_ref_text"),
                    )

                # 中栏：模型选择和参数配置 (35%)
                with gr.Column(scale=35, elem_classes="form-section"):
                    model_select_md = gr.Markdown(t("model_select_header"))

                    lora_select = gr.Dropdown(
                        label="🎯 " + t("select_lora"),
                        choices=["None"] + scan_lora_checkpoints(),
                        value="None",
                        interactive=True,
                        elem_classes="input-field",
                        info=t("info_lora_select"),
                    )

                    refresh_lora_btn = gr.Button("🔄 " + t("refresh"), elem_classes="button-refresh", size="sm")

                    gen_params_md = gr.Markdown(t("gen_params_header"))

                    cfg_scale = gr.Slider(
                        label="🎛️ " + t("cfg_scale"),
                        minimum=1.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.1,
                        info=t("info_cfg"),
                    )

                    steps = gr.Slider(
                        label="🔢 " + t("infer_steps"),
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        info=t("info_steps"),
                    )

                    seed = gr.Number(
                        label="🎲 " + t("seed"),
                        value=-1,
                        precision=0,
                        elem_classes="input-field",
                        info=t("info_seed"),
                    )

                    generate_btn = gr.Button("🎵 " + t("gen_audio"), variant="primary", elem_classes="button-primary", size="lg")

                # 右栏：生成结果 (30%)
                with gr.Column(scale=30, elem_classes="form-section"):
                    gen_output_md = gr.Markdown(t("gen_output_header"))

                    audio_out = gr.Audio(label="", elem_classes="input-field", show_label=False)

                    status_header_md = gr.Markdown(t("status_header"))

                    status_out = gr.Textbox(
                        label="",
                        interactive=False,
                        elem_classes="input-field",
                        show_label=False,
                        lines=3,
                        placeholder=t("placeholder_status"),
                    )

            def refresh_loras():
                # 获取 LoRA checkpoints 及其 base model 信息
                checkpoints_with_info = scan_lora_checkpoints(with_info=True)
                choices = ["None"] + [ckpt[0] for ckpt in checkpoints_with_info]

                # 输出调试信息
                print(t("log_refresh_lora").format(count=len(checkpoints_with_info)), file=sys.stderr)
                for ckpt_path, base_model in checkpoints_with_info:
                    if base_model:
                        print(f"  - {ckpt_path} (Base Model: {base_model})", file=sys.stderr)
                    else:
                        print(f"  - {ckpt_path}", file=sys.stderr)

                return gr.update(choices=choices, value="None")

            refresh_lora_btn.click(refresh_loras, outputs=[lora_select])

            # Auto-recognize audio when uploaded
            prompt_wav.change(fn=recognize_audio, inputs=[prompt_wav], outputs=[prompt_text])

            generate_btn.click(
                run_inference,
                inputs=[
                    infer_text,
                    prompt_wav,
                    prompt_text,
                    lora_select,
                    cfg_scale,
                    steps,
                    seed,
                    train_pretrained_path,
                ],
                outputs=[audio_out, status_out],
            )

    # --- Language Switching Logic ---
    def change_language(lang):
        d = STRINGS[lang]
        return (
            # Title area
            gr.update(value=f"# 🎵 VoxCPM LoRA WebUI\n{d['header_title']}\n\n{d['header_subtitle']}"),
            # Tabs
            gr.update(label=d["tab_train"]),
            gr.update(label=d["tab_infer"]),
            # Training setup markdown
            gr.update(value=f"{d['train_setup_title']}\n{d['train_setup_desc']}"),
            # Base config header
            gr.update(value=d["base_config_header"]),
            # Form labels
            gr.update(label=d["pretrained_path"]),
            gr.update(label=d["train_manifest"]),
            gr.update(label=d["val_manifest"]),
            # Training params header
            gr.update(value=d["train_params_header"]),
            gr.update(label=d["lr"]),
            gr.update(label=d["max_iters"]),
            gr.update(label=d["batch_size"]),
            gr.update(label=d["lora_rank"]),
            gr.update(label=d["lora_alpha"]),
            gr.update(label=d["save_interval"]),
            gr.update(label=d["output_name"]),
            gr.update(value=d["start_train"]),
            gr.update(value=d["stop_train"]),
            # Advanced accordion title
            gr.update(label=d["advanced_accordion"]),
            # Advanced options
            gr.update(label=d["grad_accum_steps"]),
            gr.update(label=d["num_workers"]),
            gr.update(label=d["log_interval"]),
            gr.update(label=d["valid_interval"]),
            gr.update(label=d["weight_decay"]),
            gr.update(label=d["warmup_steps"]),
            gr.update(label=d["max_steps"]),
            gr.update(label=d["sample_rate"]),
            gr.update(label=d["max_grad_norm"]),
            gr.update(label=d["tensorboard_path"]),
            gr.update(label=d["enable_lm"]),
            gr.update(label=d["enable_dit"]),
            gr.update(label=d["enable_proj"]),
            gr.update(label=d["dropout"]),
            # Distribution
            gr.update(value=d["distribution_header"]),
            gr.update(label=d["hf_model_id"]),
            gr.update(label=d["distribute"]),
            # Training logs header
            gr.update(value=d["train_logs_header"]),
            gr.update(label=d["train_logs"]),
            # Inference section
            gr.update(value=f"{d['infer_title']}\n{d['infer_desc']}"),
            gr.update(value=d["input_config_header"]),
            gr.update(label=d["text_to_synth"], placeholder=d["placeholder_synth"]),
            gr.update(value=d["voice_clone_header"]),
            gr.update(label=d["ref_audio"]),
            gr.update(label=d["ref_text"], placeholder=d["placeholder_ref_text"]),
            # Model selection
            gr.update(value=d["model_select_header"]),
            gr.update(label=d["select_lora"], info=d["info_lora_select"]),
            gr.update(value=d["refresh"]),
            gr.update(value=d["gen_params_header"]),
            gr.update(label=d["cfg_scale"], info=d["info_cfg"]),
            gr.update(label=d["infer_steps"], info=d["info_steps"]),
            gr.update(label=d["seed"], info=d["info_seed"]),
            gr.update(value=d["gen_audio"]),
            # Output section
            gr.update(value=d["gen_output_header"]),
            gr.update(label=d["gen_output"]),
            gr.update(value=d["status_header"]),
            gr.update(label=d["status"], placeholder=d["placeholder_status"]),
        )

    lang_btn.change(
        change_language,
        inputs=[lang_btn],
        outputs=[
            # Title area
            title_md,
            # Tabs
            tab_train,
            tab_infer,
            # Training setup
            train_setup_md,
            base_config_md,
            # Form labels
            train_pretrained_path,
            train_manifest,
            val_manifest,
            # Training params
            train_params_md,
            lr,
            num_iters,
            batch_size,
            lora_rank,
            lora_alpha,
            save_interval,
            output_name,
            start_btn,
            stop_btn,
            # Advanced accordion title
            advanced_acc,
            # Advanced options
            grad_accum_steps,
            num_workers,
            log_interval,
            valid_interval,
            weight_decay,
            warmup_steps,
            max_steps,
            sample_rate,
            max_grad_norm,
            tensorboard_path,
            enable_lm,
            enable_dit,
            enable_proj,
            dropout,
            # Distribution
            distribution_md,
            hf_model_id,
            distribute,
            # Training logs
            train_logs_md,
            logs_out,
            # Inference section
            infer_header_md,
            input_config_md,
            infer_text,
            voice_clone_md,
            prompt_wav,
            prompt_text,
            # Model selection
            model_select_md,
            lora_select,
            refresh_lora_btn,
            gen_params_md,
            cfg_scale,
            steps,
            seed,
            generate_btn,
            # Output section
            gen_output_md,
            audio_out,
            status_header_md,
            status_out,
        ],
    )

if __name__ == "__main__":
    os.makedirs("lora", exist_ok=True)
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    root_path = os.environ.get("GRADIO_ROOT_PATH", "")

    print(f"\U0001f399\ufe0f  VoxCPM Training WebUI: http://0.0.0.0:{port}{root_path}", flush=True)
    app.queue().launch(server_name="0.0.0.0", server_port=port, root_path=root_path)
