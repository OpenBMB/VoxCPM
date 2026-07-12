from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
import requests

try:
    import torch
except ImportError:  # pragma: no cover - torch is required by the project
    torch = None

try:
    from funasr import AutoModel
except ImportError:  # Optional dependency for speech-to-text input / transcripts
    AutoModel = None

from voxcpm.core import VoxCPM
from voxcpm.model.utils import resolve_runtime_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_ASR_MODEL_ID = "iic/SenseVoiceSmall"
PROFILE_DIR = Path.cwd() / ".voxcpm-phone-profiles"
DEFAULT_BACKEND_URL = os.environ.get("PCA_BACKEND_URL", "").strip()
DEFAULT_BACKEND_TOKEN = os.environ.get("PCA_BACKEND_TOKEN", "").strip()
DEFAULT_BACKEND_CONTEXT = os.environ.get("PCA_ASSISTANT_CONTEXT", "").strip()
DEFAULT_BACKEND_MODE = os.environ.get("PCA_BACKEND_MODE", "auto").strip().lower()
DEFAULT_CAPTURE_URL = os.environ.get("PCA_CAPTURE_URL", "").strip()
DEFAULT_CAPTURE_TOKEN = os.environ.get("PCA_CAPTURE_TOKEN", "").strip()

CAPTURE_CLASSIFICATIONS = {"public", "internal", "confidential", "restricted"}
_capture_classification = os.environ.get("PCA_CAPTURE_CLASSIFICATION", "confidential").strip().lower()
DEFAULT_CAPTURE_CLASSIFICATION = (
    _capture_classification if _capture_classification in CAPTURE_CLASSIFICATIONS else "confidential"
)

APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "IBM Plex Sans", "sans-serif"],
)

APP_CSS = """
:root {
    --bg0: #07111e;
    --bg1: #0d1727;
    --bg2: rgba(12, 20, 34, 0.78);
    --stroke: rgba(170, 210, 255, 0.16);
    --text: #edf4ff;
    --muted: #9fb0c8;
    --accent: #65f0ff;
    --accent2: #ffba6e;
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
}

body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(101, 240, 255, 0.20), transparent 34%),
        radial-gradient(circle at 85% 15%, rgba(255, 186, 110, 0.14), transparent 24%),
        linear-gradient(180deg, #07111e 0%, #091422 50%, #050a12 100%) !important;
    color: var(--text) !important;
}

.assistant-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 18px 16px 28px;
}

.assistant-hero {
    border: 1px solid var(--stroke);
    border-radius: 28px;
    padding: 22px 22px 20px;
    background: linear-gradient(180deg, rgba(19, 29, 48, 0.95), rgba(11, 18, 31, 0.88));
    box-shadow: var(--shadow);
    margin-bottom: 16px;
}

.assistant-kicker {
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: var(--accent);
    margin-bottom: 10px;
}

.assistant-hero h1 {
    margin: 0;
    font-size: clamp(2rem, 5vw, 3.75rem);
    line-height: 0.98;
    letter-spacing: -0.05em;
}

.assistant-hero p {
    margin: 12px 0 0;
    color: var(--muted);
    max-width: 70ch;
    font-size: 1rem;
}

.assistant-grid {
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 16px;
    align-items: start;
}

.panel {
    border: 1px solid var(--stroke);
    border-radius: 24px;
    background: var(--bg2);
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow);
    padding: 16px;
}

.panel-title {
    color: var(--text);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 10px;
}

.gradio-container .gr-button.primary,
.gradio-container .gr-button.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), #7bdbff) !important;
    color: #03131b !important;
    border: 0 !important;
    font-weight: 700 !important;
    box-shadow: 0 12px 30px rgba(101, 240, 255, 0.22);
}

.gradio-container .gr-button.secondary {
    border: 1px solid var(--stroke) !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap {
    color: var(--text) !important;
}

.gradio-container .tab-nav {
    gap: 8px;
}

.gradio-container .chatbot {
    min-height: 460px;
}

.gradio-container .gr-box,
.gradio-container .gr-panel {
    border-color: var(--stroke) !important;
}

@media (max-width: 980px) {
    .assistant-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 640px) {
    .assistant-shell {
        padding: 10px 8px 22px;
    }

    .assistant-hero {
        padding: 18px 16px 16px;
        border-radius: 22px;
    }

    .panel {
        border-radius: 20px;
        padding: 14px;
    }
}
"""


def _clean_text(value: Optional[str]) -> str:
    return (value or "").strip()


def _normalize_spaces(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def _format_history(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return history or []


def _build_final_text(text: str, control: str) -> str:
    text = _normalize_spaces(_clean_text(text))
    control = _normalize_spaces(_clean_text(control))
    return f"({control}){text}" if control else text


def _extract_reply_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, dict):
        for key in ("reply", "text", "message", "content", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                content = first.get("text")
                if isinstance(content, str) and content.strip():
                    return content.strip()

        data = payload.get("data")
        if isinstance(data, str) and data.strip():
            return data.strip()

    return ""


def _build_capture_payload(
    *,
    user_message: str,
    assistant_reply: str,
    history: list[tuple[str, str]],
    profile_name: str,
    backend_mode: str,
    classification: str = DEFAULT_CAPTURE_CLASSIFICATION,
) -> dict[str, Any]:
    """Build a PCA capture event conforming to pca_capture_event.schema.json.

    The gateway validates against that schema with additionalProperties: false,
    so only the contract fields are emitted. `assistant_reply`, `backend_mode`
    and turn count are not part of the capture contract and are intentionally
    dropped — the capture records the user's phone message, not VoxCPM state.
    """
    classification = _clean_text(classification).lower()
    if classification not in CAPTURE_CLASSIFICATIONS:
        classification = "confidential"

    return {
        "source": "iphone_shortcut",
        "capture_type": "text",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": _clean_text(user_message),
        "classification": classification,
        "provenance": {
            "agent": "voxcpm-phone-assistant",
        },
        "tags": [tag for tag in ("voxcpm-phone-assistant", _clean_text(profile_name)) if tag],
    }


def _extract_capture_receipt(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, dict):
        for key in ("capture_id", "receipt", "id", "status", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        data = payload.get("data")
        if isinstance(data, str) and data.strip():
            return data.strip()

    return ""


def _build_backend_messages(
    user_message: str,
    history: list[tuple[str, str]],
    assistant_context: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    assistant_context = _clean_text(assistant_context)
    if assistant_context:
        messages.append({"role": "system", "content": assistant_context})

    for user_turn, assistant_turn in history:
        if _clean_text(user_turn):
            messages.append({"role": "user", "content": user_turn})
        if _clean_text(assistant_turn):
            messages.append({"role": "assistant", "content": assistant_turn})

    messages.append({"role": "user", "content": user_message})
    return messages


def _profile_path(profile_name: str) -> Path:
    safe_name = "".join(ch for ch in profile_name.strip().lower() if ch.isalnum() or ch in ("-", "_"))
    if not safe_name:
        safe_name = "default"
    return PROFILE_DIR / f"{safe_name}.json"


def _copy_reference_audio(source_path: str, profile_name: str) -> str:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    suffix = source.suffix or ".wav"
    target_name = "".join(ch for ch in profile_name.strip().lower() if ch.isalnum() or ch in ("-", "_")) or "default"
    target = PROFILE_DIR / f"{target_name}{suffix}"
    shutil.copy2(source, target)
    return str(target)


def _load_profile(profile_name: str) -> dict[str, Any]:
    path = _profile_path(profile_name)
    if not path.exists():
        raise FileNotFoundError(f"Voice profile not found: {profile_name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_profile(profile_name: str, payload: dict[str, Any]) -> str:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile_name)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


class PhoneAssistantRuntime:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        load_denoiser: bool = True,
        optimize: bool = True,
        zipenhancer_model_path: str | None = None,
    ) -> None:
        self._model_id = model_id
        self.device = resolve_runtime_device(device, "cuda")
        self.optimize = optimize and self.device.startswith("cuda")
        self.load_denoiser = load_denoiser
        self.zipenhancer_model_path = zipenhancer_model_path
        self._model: Optional[VoxCPM] = None
        self._asr_model: Optional[Any] = None
        self._asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"

    def model(self) -> VoxCPM:
        if self._model is not None:
            return self._model

        logger.info("Loading VoxCPM model: %s", self._model_id)
        self._model = VoxCPM.from_pretrained(
            hf_model_id=self._model_id,
            load_denoiser=self.load_denoiser,
            zipenhancer_model_id=self.zipenhancer_model_path
            if self.zipenhancer_model_path
            else None,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("VoxCPM loaded successfully.")
        return self._model

    def asr_model(self):
        if AutoModel is None:
            raise RuntimeError(
                "Speech-to-text is not installed. Type your message manually or install the optional ASR dependency."
            )
        if self._asr_model is not None:
            return self._asr_model

        logger.info("Loading ASR model: %s on %s", DEFAULT_ASR_MODEL_ID, self._asr_device)
        self._asr_model = AutoModel(
            model=DEFAULT_ASR_MODEL_ID,
            disable_update=True,
            log_level="ERROR",
            device=self._asr_device,
        )
        return self._asr_model

    def transcribe(self, audio_path: Optional[str]) -> str:
        if not audio_path:
            return ""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        result = self.asr_model().generate(
            input=audio_path,
            language="auto",
            use_itn=True,
        )

        if not result:
            return ""
        transcript = result[0].get("text", "")
        if not isinstance(transcript, str):
            return ""
        return transcript.split("|>")[-1].strip()

    def synthesize(
        self,
        *,
        text: str,
        reference_audio: Optional[str],
        reference_transcript: Optional[str],
        control: Optional[str],
        normalize: bool,
        denoise: bool,
        cfg_value: float,
        inference_timesteps: int,
        auto_transcribe_reference: bool,
    ) -> tuple[int, np.ndarray, str]:
        model = self.model()

        target_text = _build_final_text(text, control or "")
        reference_audio = _clean_text(reference_audio) or None
        reference_transcript = _clean_text(reference_transcript) or None

        if reference_audio and not reference_transcript and auto_transcribe_reference:
            try:
                reference_transcript = self.transcribe(reference_audio)
            except Exception as exc:  # pragma: no cover - depends on optional ASR
                logger.warning("Reference transcription failed: %s", exc)

        kwargs: dict[str, Any] = dict(
            text=target_text,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            normalize=bool(normalize),
            denoise=bool(denoise and reference_audio),
        )

        if reference_audio:
            kwargs["reference_wav_path"] = reference_audio
            if reference_transcript:
                kwargs["prompt_wav_path"] = reference_audio
                kwargs["prompt_text"] = reference_transcript

        logger.info("Generating audio: cfg=%s steps=%s reference=%s", cfg_value, inference_timesteps, bool(reference_audio))
        wav = model.generate(**kwargs)
        return model.tts_model.sample_rate, wav, reference_transcript or ""

    def request_reply(
        self,
        *,
        backend_url: str,
        backend_token: str,
        user_message: str,
        history: list[tuple[str, str]],
        assistant_context: str,
        backend_mode: str = "auto",
        timeout: int = 30,
    ) -> str:
        backend_url = _clean_text(backend_url)
        if not backend_url:
            raise ValueError("No backend URL configured.")

        backend_mode = _clean_text(backend_mode).lower() or "auto"
        openai_compatible = backend_mode == "openai" or (
            backend_mode == "auto" and "/v1/chat/completions" in backend_url
        )

        if openai_compatible:
            payload = {
                "model": "pca",
                "messages": _build_backend_messages(user_message, history, assistant_context),
                "temperature": 0.4,
                "stream": False,
            }
        else:
            history_payload: list[dict[str, str]] = []
            for user_turn, assistant_turn in history:
                history_payload.append({"role": "user", "content": user_turn})
                history_payload.append({"role": "assistant", "content": assistant_turn})

            payload = {
                "message": user_message,
                "history": history_payload,
                "assistant_context": assistant_context,
                "device": self.device,
            }

        headers = {"Content-Type": "application/json"}
        if backend_token:
            headers["Authorization"] = f"Bearer {backend_token.strip()}"

        response = requests.post(backend_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        reply = ""
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                reply = _extract_reply_text(response.json())
            except Exception:
                reply = response.text.strip()
        else:
            reply = response.text.strip()

        reply = _clean_text(reply)
        if not reply:
            raise ValueError("Webhook returned an empty reply.")
        return reply

    def request_capture(
        self,
        *,
        capture_url: str,
        capture_token: str,
        user_message: str,
        assistant_reply: str,
        history: list[tuple[str, str]],
        profile_name: str,
        backend_mode: str,
        timeout: int = 30,
    ) -> str:
        capture_url = _clean_text(capture_url)
        if not capture_url:
            raise ValueError("No capture URL configured.")

        payload = _build_capture_payload(
            user_message=user_message,
            assistant_reply=assistant_reply,
            history=history,
            profile_name=profile_name,
            backend_mode=backend_mode,
        )

        headers = {"Content-Type": "application/json"}
        if capture_token:
            headers["Authorization"] = f"Bearer {capture_token.strip()}"

        response = requests.post(capture_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        receipt = ""
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                receipt = _extract_capture_receipt(response.json())
            except Exception:
                receipt = response.text.strip()
        else:
            receipt = response.text.strip()

        return _clean_text(receipt)


def _history_to_chatbot(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return _format_history(history)


def build_interface(runtime: PhoneAssistantRuntime, default_profile_data: dict[str, Any] | None = None):
    default_profile_data = default_profile_data or {}
    backend_url_default = _clean_text(os.environ.get("PCA_BACKEND_URL", DEFAULT_BACKEND_URL))
    backend_token_default = _clean_text(os.environ.get("PCA_BACKEND_TOKEN", DEFAULT_BACKEND_TOKEN))
    backend_context_default = _clean_text(os.environ.get("PCA_ASSISTANT_CONTEXT", DEFAULT_BACKEND_CONTEXT))
    backend_mode_default = _clean_text(os.environ.get("PCA_BACKEND_MODE", DEFAULT_BACKEND_MODE)).lower() or "auto"
    capture_url_default = _clean_text(os.environ.get("PCA_CAPTURE_URL", DEFAULT_CAPTURE_URL))
    capture_token_default = _clean_text(os.environ.get("PCA_CAPTURE_TOKEN", DEFAULT_CAPTURE_TOKEN))
    if backend_mode_default not in {"auto", "openai", "custom"}:
        backend_mode_default = "auto"

    def _merge_profile_inputs(
        profile_data: dict[str, Any],
        reference_audio: Optional[str],
        reference_transcript: str,
        control_text: str,
    ) -> tuple[Optional[str], str, str]:
        profile_data = profile_data or {}

        effective_audio = _clean_text(reference_audio) or _clean_text(profile_data.get("reference_audio"))
        effective_transcript = _clean_text(reference_transcript) or _clean_text(profile_data.get("reference_transcript"))
        effective_control = _clean_text(control_text) or _clean_text(profile_data.get("control_text"))
        return (
            effective_audio or None,
            effective_transcript,
            effective_control,
        )

    def _profile_summary(profile_data: dict[str, Any]) -> str:
        if not profile_data:
            return "No saved profile loaded."
        name = profile_data.get("name", "default")
        audio = "yes" if profile_data.get("reference_audio") else "no"
        transcript = "yes" if profile_data.get("reference_transcript") else "no"
        return f"Loaded profile: {name} | audio: {audio} | transcript: {transcript}"

    def save_voice_profile(
        profile_name: str,
        reference_audio: Optional[str],
        reference_transcript: str,
        control_text: str,
        auto_transcribe_reference: bool,
    ):
        profile_name = _clean_text(profile_name) or "default"
        if not reference_audio:
            raise ValueError("Upload a reference voice sample before saving the profile.")

        transcript = _clean_text(reference_transcript)
        if not transcript and auto_transcribe_reference:
            transcript = runtime.transcribe(reference_audio)

        stored_audio = _copy_reference_audio(reference_audio, profile_name)
        payload = {
            "name": profile_name,
            "reference_audio": stored_audio,
            "reference_transcript": transcript,
            "control_text": _clean_text(control_text),
        }
        saved_path = _save_profile(profile_name, payload)
        payload["profile_path"] = saved_path
        return payload, _profile_summary(payload), stored_audio, transcript, payload.get("control_text", "")

    def load_voice_profile(profile_name: str):
        profile_name = _clean_text(profile_name) or "default"
        payload = _load_profile(profile_name)
        return (
            payload,
            payload.get("reference_audio", ""),
            payload.get("reference_transcript", ""),
            payload.get("control_text", ""),
            _profile_summary(payload),
        )

    def clear_voice_profile():
        return {}, "", "", "", "No saved profile loaded."

    def prepare_turn(
        user_audio: Optional[str],
        user_text: str,
        assistant_reply: str,
        backend_url: str,
        backend_token: str,
        capture_url: str,
        capture_token: str,
        assistant_context: str,
        backend_mode: str,
        reference_audio: Optional[str],
        reference_transcript: str,
        control_text: str,
        normalize_text: bool,
        denoise_reference: bool,
        auto_transcribe_reference: bool,
        cfg_value: float,
        inference_timesteps: int,
        history: list[tuple[str, str]],
        profile_data: dict[str, Any],
    ):
        history = _history_to_chatbot(history)
        user_text_clean = _clean_text(user_text)

        if not user_text_clean and user_audio:
            if AutoModel is None:
                raise RuntimeError(
                    "No text message provided and speech-to-text is unavailable. Type the message manually or install the optional ASR dependency."
                )
            user_text_clean = runtime.transcribe(user_audio)

        user_text_clean = _normalize_spaces(user_text_clean)
        if not user_text_clean:
            raise ValueError("Provide a text message or record one with the phone microphone.")

        if backend_url.strip():
            assistant_reply = runtime.request_reply(
                backend_url=backend_url,
                backend_token=backend_token,
                user_message=user_text_clean,
                history=history,
                assistant_context=assistant_context,
                backend_mode=backend_mode,
            )
        else:
            assistant_reply = _clean_text(assistant_reply)
            if not assistant_reply:
                raise ValueError("Type the assistant reply, or configure a backend URL.")

        capture_status = ""
        if capture_url.strip():
            try:
                capture_receipt = runtime.request_capture(
                    capture_url=capture_url,
                    capture_token=capture_token,
                    user_message=user_text_clean,
                    assistant_reply=assistant_reply,
                    history=history,
                    profile_name=_clean_text(profile_data.get("name", "default")) or "default",
                    backend_mode=backend_mode,
                )
                capture_status = "Capture staged in PCA."
                if capture_receipt:
                    capture_status = f"Capture staged in PCA ({capture_receipt})."
            except Exception as exc:
                logger.warning("Capture staging failed: %s", exc)
                capture_status = f"Capture staging failed: {exc}"

        reference_audio, reference_transcript, control_text = _merge_profile_inputs(
            profile_data,
            reference_audio,
            reference_transcript,
            control_text,
        )

        sample_rate, wav, transcript = runtime.synthesize(
            text=assistant_reply,
            reference_audio=reference_audio,
            reference_transcript=reference_transcript,
            control=control_text,
            normalize=normalize_text,
            denoise=denoise_reference,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            auto_transcribe_reference=auto_transcribe_reference,
        )

        if reference_audio and not reference_transcript and transcript:
            reference_transcript = transcript

        history = history + [(f"You: {user_text_clean}", f"PCA: {assistant_reply}")]
        status = "Generated cloned speech."
        if reference_audio and transcript:
            status = "Generated cloned speech with transcript-guided cloning."
        if capture_status:
            status = f"{status} {capture_status}"

        return (
            history,
            user_text_clean,
            assistant_reply,
            reference_transcript,
            (sample_rate, wav),
            status,
            profile_data,
        )

    def speak_reply(
        assistant_reply: str,
        reference_audio: Optional[str],
        reference_transcript: str,
        control_text: str,
        profile_data: dict[str, Any],
        normalize_text: bool,
        denoise_reference: bool,
        auto_transcribe_reference: bool,
        cfg_value: float,
        inference_timesteps: int,
    ):
        assistant_reply = _clean_text(assistant_reply)
        if not assistant_reply:
            raise ValueError("Type a reply to speak first.")

        reference_audio, reference_transcript, control_text = _merge_profile_inputs(
            profile_data,
            reference_audio,
            reference_transcript,
            control_text,
        )

        sample_rate, wav, transcript = runtime.synthesize(
            text=assistant_reply,
            reference_audio=reference_audio,
            reference_transcript=reference_transcript,
            control=control_text,
            normalize=normalize_text,
            denoise=denoise_reference,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            auto_transcribe_reference=auto_transcribe_reference,
        )

        return (sample_rate, wav), transcript or reference_transcript, "Playback updated."

    def clear_state(profile_data: dict[str, Any]):
        return [], "", "", "", None, "Ready.", profile_data

    with gr.Blocks(theme=APP_THEME, css=APP_CSS, fill_height=True) as demo:
        profile_state = gr.State(default_profile_data)
        with gr.Column(elem_classes=["assistant-shell"]):
            gr.HTML(
                """
                <div class="assistant-hero">
                  <div class="assistant-kicker">VoxCPM Phone Assistant</div>
                  <h1>Cloned voice for your mobile assistant.</h1>
                  <p>
                    Record a message from your phone, send it to your assistant or webhook,
                    and hear the reply in the cloned voice from your reference samples.
                    Best results come from a clean 10-30 second clip with an exact transcript.
                  </p>
                </div>
                """
            )

            with gr.Row(elem_classes=["assistant-grid"]):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Conversation")
                    conversation = gr.Chatbot(
                        label="Conversation",
                        height=460,
                        bubble_full_width=False,
                        show_copy_button=True,
                    )

                    user_text = gr.Textbox(
                        label="Your message",
                        placeholder="Type a message, or record one with the phone microphone below.",
                        lines=3,
                    )
                    user_audio = gr.Audio(
                        label="Voice input",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )

                    with gr.Row():
                        send_btn = gr.Button("Send & Speak", variant="primary")
                        speak_btn = gr.Button("Speak reply", variant="secondary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                    assistant_reply = gr.Textbox(
                        label="Assistant reply",
                        placeholder="Webhook reply appears here, or type the response you want VoxCPM to speak.",
                        lines=4,
                    )
                    audio_out = gr.Audio(label="Voiced reply")
                    status = gr.Textbox(label="Status", value="Ready.", interactive=False)

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Voice profile")
                    profile_name = gr.Textbox(
                        label="Profile name",
                        value=default_profile_data.get("name", "default"),
                        placeholder="A short name for this voice profile, like my-voice.",
                    )
                    reference_audio = gr.Audio(
                        label="Reference voice sample",
                        sources=["upload", "microphone"],
                        value=default_profile_data.get("reference_audio"),
                        type="filepath",
                    )
                    reference_transcript = gr.Textbox(
                        label="Reference transcript",
                        value=default_profile_data.get("reference_transcript", ""),
                        placeholder="Paste the exact transcript for best cloning quality. Leave blank to auto-transcribe if available.",
                        lines=3,
                    )
                    control_text = gr.Textbox(
                        label="Voice control",
                        value=default_profile_data.get("control_text", ""),
                        placeholder="Optional style guidance such as warm, calm, faster, younger, smiling.",
                        lines=2,
                    )

                    with gr.Row():
                        save_profile_btn = gr.Button("Save enrollment", variant="primary")
                        load_profile_btn = gr.Button("Load enrollment", variant="secondary")
                        clear_profile_btn = gr.Button("Clear enrollment", variant="secondary")
                    profile_status = gr.Textbox(
                        label="Profile status",
                        value=_profile_summary(default_profile_data),
                        interactive=False,
                    )

                    with gr.Accordion("Advanced", open=False):
                        assistant_webhook_url = gr.Textbox(
                            label="PCA backend URL",
                            value=backend_url_default,
                            placeholder="OpenAI-compatible /v1/chat/completions or a custom webhook endpoint.",
                        )
                        assistant_webhook_token = gr.Textbox(
                            label="PCA backend token",
                            value=backend_token_default,
                            placeholder="Optional bearer token.",
                        )
                        capture_webhook_url = gr.Textbox(
                            label="PCA capture URL",
                            value=capture_url_default,
                            placeholder="Optional capture webhook URL for staging the phone message in PCA.",
                        )
                        capture_webhook_token = gr.Textbox(
                            label="PCA capture token",
                            value=capture_token_default,
                            placeholder="Optional bearer token for the capture webhook.",
                        )
                        assistant_context = gr.Textbox(
                            label="Assistant context",
                            value=backend_context_default,
                            placeholder="Optional system prompt or assistant instructions.",
                            lines=3,
                        )
                        backend_mode = gr.Dropdown(
                            choices=["auto", "openai", "custom"],
                            value=backend_mode_default,
                            label="Backend mode",
                            info="Auto detects OpenAI-compatible chat endpoints; custom uses the webhook payload format.",
                        )
                        auto_transcribe_reference = gr.Checkbox(
                            value=True,
                            label="Auto-transcribe reference audio when ASR is installed",
                        )
                        normalize_text = gr.Checkbox(
                            value=False,
                            label="Normalize text",
                        )
                        denoise_reference = gr.Checkbox(
                            value=False,
                            label="Denoise reference audio",
                        )
                        cfg_value = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=2.0,
                            step=0.1,
                            label="CFG guidance scale",
                        )
                        inference_timesteps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Inference steps",
                        )

                    gr.Markdown(
                        """
                        **Backend payloads**

                        OpenAI-compatible mode sends `{"model":"pca","messages":[...],"temperature":0.4,"stream":false}`.

                        Custom mode sends `{"message": "...", "history": [...], "assistant_context": "..."}`.

                        If a PCA capture URL is configured, the phone message is also staged as a PCA capture event (`source: iphone_shortcut`, `capture_type: text`, `content`, `classification`, and `provenance`) conforming to the PCA capture schema.

                        The response can be plain text or JSON with `reply`, `text`, `content`, or OpenAI-style `choices[0].message.content`.
                        """
                    )

            send_btn.click(
                fn=prepare_turn,
                inputs=[
                    user_audio,
                    user_text,
                    assistant_reply,
                    assistant_webhook_url,
                    assistant_webhook_token,
                    capture_webhook_url,
                    capture_webhook_token,
                    assistant_context,
                    backend_mode,
                    reference_audio,
                    reference_transcript,
                    control_text,
                    normalize_text,
                    denoise_reference,
                    auto_transcribe_reference,
                    cfg_value,
                    inference_timesteps,
                    conversation,
                    profile_state,
                ],
                outputs=[
                    conversation,
                    user_text,
                    assistant_reply,
                    reference_transcript,
                    audio_out,
                    status,
                    profile_state,
                ],
                show_progress=True,
            )

            speak_btn.click(
                fn=speak_reply,
                inputs=[
                    assistant_reply,
                    reference_audio,
                    reference_transcript,
                    control_text,
                    profile_state,
                    normalize_text,
                    denoise_reference,
                    auto_transcribe_reference,
                    cfg_value,
                    inference_timesteps,
                ],
                outputs=[audio_out, reference_transcript, status],
                show_progress=True,
            )

            save_profile_btn.click(
                fn=save_voice_profile,
                inputs=[
                    profile_name,
                    reference_audio,
                    reference_transcript,
                    control_text,
                    auto_transcribe_reference,
                ],
                outputs=[
                    profile_state,
                    profile_status,
                    reference_audio,
                    reference_transcript,
                    control_text,
                ],
                show_progress=True,
            )

            load_profile_btn.click(
                fn=load_voice_profile,
                inputs=[profile_name],
                outputs=[
                    profile_state,
                    reference_audio,
                    reference_transcript,
                    control_text,
                    profile_status,
                ],
                show_progress=True,
            )

            clear_profile_btn.click(
                fn=clear_voice_profile,
                inputs=[],
                outputs=[profile_state, reference_audio, reference_transcript, control_text, profile_status],
            )

            clear_btn.click(
                fn=clear_state,
                inputs=[profile_state],
                outputs=[conversation, user_text, assistant_reply, reference_transcript, audio_out, status, profile_state],
            )

    return demo


def launch(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    port: int = 8809,
    host: str = "0.0.0.0",
    share: bool = False,
    no_denoiser: bool = False,
    optimize: bool = True,
    zipenhancer_path: Optional[str] = None,
    default_profile_name: str = "reddit-female",
    backend_url: str = DEFAULT_BACKEND_URL,
    backend_token: str = DEFAULT_BACKEND_TOKEN,
    capture_url: str = DEFAULT_CAPTURE_URL,
    capture_token: str = DEFAULT_CAPTURE_TOKEN,
    backend_context: str = DEFAULT_BACKEND_CONTEXT,
    backend_mode: str = DEFAULT_BACKEND_MODE if DEFAULT_BACKEND_MODE in {"auto", "openai", "custom"} else "auto",
):
    runtime = PhoneAssistantRuntime(
        model_id=model_id,
        device=device,
        load_denoiser=not no_denoiser,
        optimize=optimize,
        zipenhancer_model_path=zipenhancer_path,
    )
    default_profile_data: dict[str, Any] = {}
    if _clean_text(default_profile_name):
        try:
            default_profile_data = _load_profile(default_profile_name)
        except FileNotFoundError:
            logger.warning("Default profile not found: %s", default_profile_name)

    if _clean_text(backend_url):
        os.environ["PCA_BACKEND_URL"] = backend_url
    if _clean_text(backend_token):
        os.environ["PCA_BACKEND_TOKEN"] = backend_token
    if _clean_text(capture_url):
        os.environ["PCA_CAPTURE_URL"] = capture_url
    if _clean_text(capture_token):
        os.environ["PCA_CAPTURE_TOKEN"] = capture_token
    if _clean_text(backend_context):
        os.environ["PCA_ASSISTANT_CONTEXT"] = backend_context
    if _clean_text(backend_mode):
        os.environ["PCA_BACKEND_MODE"] = backend_mode

    demo = build_interface(runtime, default_profile_data=default_profile_data)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
    )


def main():
    parser = argparse.ArgumentParser(description="VoxCPM mobile assistant bridge")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--port", type=int, default=8809)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-denoiser", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--zipenhancer-path", type=str, default=None)
    parser.add_argument("--default-profile-name", type=str, default="reddit-female")
    parser.add_argument("--backend-url", type=str, default=DEFAULT_BACKEND_URL)
    parser.add_argument("--backend-token", type=str, default=DEFAULT_BACKEND_TOKEN)
    parser.add_argument("--capture-url", type=str, default=DEFAULT_CAPTURE_URL)
    parser.add_argument("--capture-token", type=str, default=DEFAULT_CAPTURE_TOKEN)
    parser.add_argument("--backend-context", type=str, default=DEFAULT_BACKEND_CONTEXT)
    parser.add_argument(
        "--backend-mode",
        type=str,
        default=DEFAULT_BACKEND_MODE if DEFAULT_BACKEND_MODE in {"auto", "openai", "custom"} else "auto",
        choices=["auto", "openai", "custom"],
    )
    args = parser.parse_args()

    launch(
        model_id=args.model_id,
        device=args.device,
        port=args.port,
        host=args.host,
        share=args.share,
        no_denoiser=args.no_denoiser,
        optimize=not args.no_optimize,
        zipenhancer_path=args.zipenhancer_path,
        default_profile_name=args.default_profile_name,
        backend_url=args.backend_url,
        backend_token=args.backend_token,
        capture_url=args.capture_url,
        capture_token=args.capture_token,
        backend_context=args.backend_context,
        backend_mode=args.backend_mode,
    )


if __name__ == "__main__":
    main()
