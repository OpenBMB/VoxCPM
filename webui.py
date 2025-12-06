import os
import sys
import time
import glob
import yaml
import shutil
import datetime
import subprocess
import threading
import gradio as gr
import torch
import soundfile as sf
from pathlib import Path
from typing import Optional, List

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Default pretrained model path relative to this repo
default_pretrained_path = str(project_root / "models" / "openbmb__VoxCPM1.5")

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
import numpy as np
from funasr import AutoModel

# --- Localization ---
LANG_DICT = {
    "en": {
        "title": "VoxCPM LoRA WebUI",
        "tab_train": "Training",
        "tab_infer": "Inference",
        "pretrained_path": "Pretrained Model Path",
        "train_manifest": "Train Manifest (jsonl)",
        "val_manifest": "Validation Manifest (Optional)",
        "lr": "Learning Rate",
        "max_iters": "Max Iterations",
        "batch_size": "Batch Size",
        "lora_rank": "LoRA Rank",
        "lora_alpha": "LoRA Alpha",
        "save_interval": "Save Interval",
        "start_train": "Start Training",
        "stop_train": "Stop Training",
        "train_logs": "Training Logs",
        "text_to_synth": "Text to Synthesize",
        "voice_cloning": "### Voice Cloning (Optional)",
        "ref_audio": "Reference Audio",
        "ref_text": "Reference Text (Optional)",
        "select_lora": "Select LoRA Checkpoint",
        "cfg_scale": "CFG Scale",
        "infer_steps": "Inference Steps",
        "seed": "Seed",
        "gen_audio": "Generate Audio",
        "gen_output": "Generated Audio",
        "status": "Status",
        "lang_select": "Language / 语言",
        "refresh": "Refresh",
        "output_name": "Output Name (Optional, resume if exists)",
    },
    "zh": {
        "title": "VoxCPM LoRA WebUI",
        "tab_train": "训练 (Training)",
        "tab_infer": "推理 (Inference)",
        "pretrained_path": "预训练模型路径",
        "train_manifest": "训练数据清单 (jsonl)",
        "val_manifest": "验证数据清单 (可选)",
        "lr": "学习率 (Learning Rate)",
        "max_iters": "最大迭代次数",
        "batch_size": "批次大小 (Batch Size)",
        "lora_rank": "LoRA Rank",
        "lora_alpha": "LoRA Alpha",
        "save_interval": "保存间隔 (Steps)",
        "start_train": "开始训练",
        "stop_train": "停止训练",
        "train_logs": "训练日志",
        "text_to_synth": "合成文本",
        "voice_cloning": "### 声音克隆 (可选)",
        "ref_audio": "参考音频",
        "ref_text": "参考文本 (可选)",
        "select_lora": "选择 LoRA 模型",
        "cfg_scale": "CFG Scale (引导系数)",
        "infer_steps": "推理步数",
        "seed": "随机种子 (Seed)",
        "gen_audio": "生成音频",
        "gen_output": "生成结果",
        "status": "状态",
        "lang_select": "Language / 语言",
        "refresh": "刷新",
        "output_name": "输出目录名称 (可选，若存在则继续训练)",
    }
}

# Global variables
current_model: Optional[VoxCPM] = None
asr_model: Optional[AutoModel] = None
training_process: Optional[subprocess.Popen] = None
training_log = ""

def get_timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_or_load_asr_model():
    global asr_model
    if asr_model is None:
        print("Loading ASR model (SenseVoiceSmall)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level='ERROR',
            device=device,
        )
    return asr_model

def recognize_audio(audio_path):
    if not audio_path:
        return ""
    try:
        model = get_or_load_asr_model()
        res = model.generate(input=audio_path, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text
    except Exception as e:
        print(f"ASR Error: {e}")
        return ""

def scan_lora_checkpoints(root_dir="lora"):
    """Scans for LoRA checkpoints in the lora directory."""
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    
    # Look for lora_weights.safetensors recursively
    for root, dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            # Use the relative path from root_dir as the ID
            rel_path = os.path.relpath(root, root_dir)
            checkpoints.append(rel_path)
            
    # Also check for checkpoints in the default location if they exist
    default_ckpt = "checkpoints/finetune_lora"
    if os.path.exists(os.path.join(root_dir, default_ckpt)):
         # This might be covered by the walk, but good to be sure
         pass

    return sorted(checkpoints, reverse=True)

def load_model(pretrained_path, lora_path=None):
    global current_model
    print(f"Loading model from {pretrained_path}...")
    
    lora_config = None
    lora_weights_path = None
    
    if lora_path:
        full_lora_path = os.path.join("lora", lora_path)
        if os.path.exists(full_lora_path):
            lora_weights_path = full_lora_path
            # We assume standard LoRA config if loading from weights
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                r=32,
                alpha=16,
                target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
                target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
    
    # Always init with a default LoRA config to allow hot-swapping later if not already set
    if lora_config is None:
        lora_config = LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            r=32, # Default rank
            alpha=16,
            target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
            target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

    current_model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=False,
        lora_config=lora_config,
        lora_weights_path=lora_weights_path,
    )
    return "Model loaded successfully!"

def run_inference(text, prompt_wav, prompt_text, lora_selection, cfg_scale, steps, seed):
    global current_model
    if current_model is None:
        # Try to load default
        try:
            load_model("models/openbmb__VoxCPM1.5")
        except:
            return None, "Model not loaded and default failed."

    # Handle LoRA hot-swapping
    if lora_selection and lora_selection != "None":
        full_lora_path = os.path.join("lora", lora_selection)
        print(f"Hot-loading LoRA: {full_lora_path}")
        try:
            current_model.load_lora(full_lora_path)
            current_model.set_lora_enabled(True)
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return None, f"Error loading LoRA: {e}"
    else:
        print("Disabling LoRA")
        current_model.set_lora_enabled(False)

    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
        audio_np = current_model.generate(
            text=text,
            prompt_wav_path=prompt_wav,
            prompt_text=prompt_text,
            cfg_value=cfg_scale,
            inference_timesteps=steps,
            denoise=False 
        )
        return (current_model.tts_model.sample_rate, audio_np), "Generation Success"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

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
    sample_rate=44100,
    # LoRA advanced
    enable_lm=True,
    enable_dit=True,
    enable_proj=False,
    dropout=0.0,
    tensorboard_path=""
):
    global training_process, training_log
    
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

    # Create config dictionary
    # Resolve max_steps default
    resolved_max_steps = int(max_steps) if max_steps not in (None, "", 0) else int(num_iters)

    config = {
        "pretrained_path": pretrained_path,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "sample_rate": int(sample_rate),
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
        "save_path": checkpoints_dir,
        "tensorboard": tensorboard_path if tensorboard_path else logs_dir,
        "lambdas": {
            "loss/diff": 1.0,
            "loss/stop": 1.0
        },
        "lora": {
            "enable_lm": bool(enable_lm),
            "enable_dit": bool(enable_dit),
            "enable_proj": bool(enable_proj),
            "r": int(lora_rank),
            "alpha": int(lora_alpha),
            "dropout": float(dropout),
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    }

    config_path = os.path.join(save_dir, "train_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable,
        "scripts/train_voxcpm_finetune.py",
        "--config_path",
        config_path
    ]

    training_log = f"Starting training...\nConfig saved to {config_path}\nOutput dir: {save_dir}\n"
    
    def run_process():
        global training_process, training_log
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in training_process.stdout:
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
    global training_process, training_log
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        training_log += "\nTraining terminated by user."
        return "Training stopped."
    return "No training running."

# --- GUI Layout ---

with gr.Blocks(title="VoxCPM LoRA WebUI") as app:
    
    # State for language
    lang_state = gr.State("zh") # Default to Chinese

    with gr.Row():
        title_md = gr.Markdown("# VoxCPM WebUI")
        lang_btn = gr.Radio(choices=["en", "zh"], value="zh", label="Language / 语言", scale=0, min_width=210,type="value")

    with gr.Tabs() as tabs:
        # === Training Tab ===
        with gr.Tab("训练 (Training)") as tab_train:
            with gr.Row():
                with gr.Column():
                    train_pretrained_path = gr.Textbox(
                        label="预训练模型路径", 
                        value=default_pretrained_path
                    )
                    train_manifest = gr.Textbox(
                        label="训练数据清单 (jsonl)", 
                        value="examples/train_data_example.jsonl"
                    )
                    val_manifest = gr.Textbox(
                        label="验证数据清单 (可选)", 
                        value=""
                    )
                    
                    with gr.Row():
                        lr = gr.Number(label="学习率 (Learning Rate)", value=1e-4)
                        num_iters = gr.Number(label="最大迭代次数", value=2000, precision=0)
                        batch_size = gr.Number(label="批次大小 (Batch Size)", value=1, precision=0)
                    
                    with gr.Row():
                        lora_rank = gr.Number(label="LoRA Rank", value=32, precision=0)
                        lora_alpha = gr.Number(label="LoRA Alpha", value=16, precision=0)
                        save_interval = gr.Number(label="保存间隔 (Steps)", value=1000, precision=0)
                    
                    output_name = gr.Textbox(label="输出目录名称 (可选，若存在则继续训练)", value="")

                    with gr.Row():
                        start_btn = gr.Button("开始训练", variant="primary")
                        stop_btn = gr.Button("停止训练", variant="stop")

                    with gr.Accordion("高级选项 (Advanced)", open=False):
                        with gr.Row():
                            grad_accum_steps = gr.Number(label="梯度累积 (grad_accum_steps)", value=1, precision=0)
                            num_workers = gr.Number(label="数据加载线程 (num_workers)", value=2, precision=0)
                            log_interval = gr.Number(label="日志间隔 (log_interval)", value=10, precision=0)
                        with gr.Row():
                            valid_interval = gr.Number(label="验证间隔 (valid_interval)", value=1000, precision=0)
                            weight_decay = gr.Number(label="权重衰减 (weight_decay)", value=0.01)
                            warmup_steps = gr.Number(label="warmup_steps", value=100, precision=0)
                        with gr.Row():
                            max_steps = gr.Number(label="最大步数 (max_steps, 0→默认num_iters)", value=0, precision=0)
                            sample_rate = gr.Number(label="采样率 (sample_rate)", value=44100, precision=0)
                            tensorboard_path = gr.Textbox(label="Tensorboard 路径 (可选)", value="")
                        with gr.Row():
                            enable_lm = gr.Checkbox(label="启用 LoRA LM (enable_lm)", value=True)
                            enable_dit = gr.Checkbox(label="启用 LoRA DIT (enable_dit)", value=True)
                            enable_proj = gr.Checkbox(label="启用投影 (enable_proj)", value=False)
                            dropout = gr.Number(label="LoRA Dropout", value=0.0)

                with gr.Column():
                    logs_out = gr.TextArea(label="训练日志", lines=20, max_lines=30, interactive=False)
                    
            start_btn.click(
                start_training,
                inputs=[
                    train_pretrained_path, train_manifest, val_manifest,
                    lr, num_iters, batch_size, lora_rank, lora_alpha, save_interval,
                    output_name,
                    # advanced
                    grad_accum_steps, num_workers, log_interval, valid_interval,
                    weight_decay, warmup_steps, max_steps, sample_rate,
                    enable_lm, enable_dit, enable_proj, dropout, tensorboard_path
                ],
                outputs=[logs_out] # Initial message
            )
            stop_btn.click(stop_training, outputs=[logs_out])
            
            # Log refresher
            timer = gr.Timer(1)
            timer.tick(get_training_log, outputs=logs_out)

        # === Inference Tab ===
        with gr.Tab("推理 (Inference)") as tab_infer:
            with gr.Row():
                with gr.Column():
                    infer_text = gr.TextArea(label="合成文本", value="Hello, this is a test of the VoxCPM LoRA model.")
                    
                    with gr.Group():
                        voice_cloning_md = gr.Markdown("### 声音克隆 (可选)")
                        prompt_wav = gr.Audio(label="参考音频", type="filepath")
                        prompt_text = gr.Textbox(label="参考文本 (可选)")
                    
                    with gr.Row():
                        lora_select = gr.Dropdown(
                            label="选择 LoRA 模型", 
                            choices=["None"] + scan_lora_checkpoints(),
                            value="None",
                            interactive=True
                        )
                        refresh_lora_btn = gr.Button("刷新", size="sm")
                    
                    with gr.Row():
                        cfg_scale = gr.Slider(label="CFG Scale (引导系数)", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
                        steps = gr.Slider(label="推理步数", minimum=1, maximum=50, value=10, step=1)
                        seed = gr.Number(label="随机种子 (Seed)", value=-1, precision=0)

                    generate_btn = gr.Button("生成音频", variant="primary")

                with gr.Column():
                    audio_out = gr.Audio(label="生成结果")
                    status_out = gr.Textbox(label="状态", interactive=False)

            def refresh_loras():
                return gr.Dropdown(choices=["None"] + scan_lora_checkpoints())

            refresh_lora_btn.click(refresh_loras, outputs=[lora_select])
            
            # Auto-recognize audio when uploaded
            prompt_wav.change(
                fn=recognize_audio,
                inputs=[prompt_wav],
                outputs=[prompt_text]
            )
            
            generate_btn.click(
                run_inference,
                inputs=[infer_text, prompt_wav, prompt_text, lora_select, cfg_scale, steps, seed],
                outputs=[audio_out, status_out]
            )

    # --- Language Switching Logic ---
    def change_language(lang):
        d = LANG_DICT[lang]
        # Labels for advanced options
        if lang == "zh":
            adv = {
                'grad_accum_steps': "梯度累积 (grad_accum_steps)",
                'num_workers': "数据加载线程 (num_workers)",
                'log_interval': "日志间隔 (log_interval)",
                'valid_interval': "验证间隔 (valid_interval)",
                'weight_decay': "权重衰减 (weight_decay)",
                'warmup_steps': "warmup_steps",
                'max_steps': "最大步数 (max_steps)",
                'sample_rate': "采样率 (sample_rate)",
                'enable_lm': "启用 LoRA LM (enable_lm)",
                'enable_dit': "启用 LoRA DIT (enable_dit)",
                'enable_proj': "启用投影 (enable_proj)",
                'dropout': "LoRA Dropout",
                'tensorboard_path': "Tensorboard 路径 (可选)"
            }
        else:
            adv = {
                'grad_accum_steps': "Grad Accum Steps",
                'num_workers': "Num Workers",
                'log_interval': "Log Interval",
                'valid_interval': "Valid Interval",
                'weight_decay': "Weight Decay",
                'warmup_steps': "Warmup Steps",
                'max_steps': "Max Steps",
                'sample_rate': "Sample Rate",
                'enable_lm': "Enable LoRA LM",
                'enable_dit': "Enable LoRA DIT",
                'enable_proj': "Enable Projection",
                'dropout': "LoRA Dropout",
                'tensorboard_path': "Tensorboard Path (Optional)"
            }

        return (
            gr.update(value=f"# {d['title']}"),
            gr.update(label=d['tab_train']),
            gr.update(label=d['tab_infer']),
            gr.update(label=d['pretrained_path']),
            gr.update(label=d['train_manifest']),
            gr.update(label=d['val_manifest']),
            gr.update(label=d['lr']),
            gr.update(label=d['max_iters']),
            gr.update(label=d['batch_size']),
            gr.update(label=d['lora_rank']),
            gr.update(label=d['lora_alpha']),
            gr.update(label=d['save_interval']),
            gr.update(label=d['output_name']),
            gr.update(value=d['start_train']),
            gr.update(value=d['stop_train']),
            gr.update(label=d['train_logs']),
            # Advanced options (must match outputs order)
            gr.update(label=adv['grad_accum_steps']),
            gr.update(label=adv['num_workers']),
            gr.update(label=adv['log_interval']),
            gr.update(label=adv['valid_interval']),
            gr.update(label=adv['weight_decay']),
            gr.update(label=adv['warmup_steps']),
            gr.update(label=adv['max_steps']),
            gr.update(label=adv['sample_rate']),
            gr.update(label=adv['enable_lm']),
            gr.update(label=adv['enable_dit']),
            gr.update(label=adv['enable_proj']),
            gr.update(label=adv['dropout']),
            gr.update(label=adv['tensorboard_path']),
            # Inference section
            gr.update(label=d['text_to_synth']),
            gr.update(value=d['voice_cloning']),
            gr.update(label=d['ref_audio']),
            gr.update(label=d['ref_text']),
            gr.update(label=d['select_lora']),
            gr.update(value=d['refresh']),
            gr.update(label=d['cfg_scale']),
            gr.update(label=d['infer_steps']),
            gr.update(label=d['seed']),
            gr.update(value=d['gen_audio']),
            gr.update(label=d['gen_output']),
            gr.update(label=d['status']),
        )

    lang_btn.change(
        change_language,
        inputs=[lang_btn],
        outputs=[
            title_md, tab_train, tab_infer,
            train_pretrained_path, train_manifest, val_manifest,
            lr, num_iters, batch_size, lora_rank, lora_alpha, save_interval,
            output_name,
            start_btn, stop_btn, logs_out,
            # advanced outputs
            grad_accum_steps, num_workers, log_interval, valid_interval,
            weight_decay, warmup_steps, max_steps, sample_rate,
            enable_lm, enable_dit, enable_proj, dropout, tensorboard_path,
            infer_text, voice_cloning_md, prompt_wav, prompt_text,
            lora_select, refresh_lora_btn, cfg_scale, steps, seed,
            generate_btn, audio_out, status_out
        ]
    )

if __name__ == "__main__":
    # Ensure lora directory exists
    os.makedirs("lora", exist_ok=True)
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
