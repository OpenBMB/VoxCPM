# VoxCPM2 – Docker image with CUDA 13.0 (native sm_120 / Blackwell support)
# Pattern adapted from suite-redazione/tada-tts (proven on RTX 5070)
# Requires: nvidia-container-toolkit on the host
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/cache/huggingface
ENV MODELSCOPE_CACHE=/cache/modelscope
ENV TORCH_HOME=/cache/torch

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libgomp1 \
    # cuBLAS 12 compat — some deps (funasr, ctranslate2) ship CUDA 12 binaries
    libcublas-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv con Python 3.11 — isolato dal sistema
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# setuptools>=78 required for PEP 639 SPDX license strings
RUN pip install --no-cache-dir --upgrade "pip>=24" "setuptools>=78" wheel

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# Install voxcpm and all its deps (may pull CPU torch as transitive dep — OK)
RUN pip install --no-cache-dir -e .

# Force-reinstall the cu130 torch stack last so CUDA versions always win.
# CUDA 13.0 wheels include Triton with native sm_120 (Blackwell) support.
RUN pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch torchaudio torchcodec

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8808

CMD ["python", "app.py", "--port", "8808"]
