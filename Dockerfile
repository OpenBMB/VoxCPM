# VoxCPM API server image (GPU-ready, also runs on CPU)
#
# Build:
#   docker build -t voxcpm-server .
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 -v voxcpm-cache:/root/.cache voxcpm-server
#
# Run (CPU only — slow, for testing):
#   docker run -p 8000:8000 -e VOXCPM_DEVICE=cpu -v voxcpm-cache:/root/.cache voxcpm-server
#
# Then open http://localhost:8000

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY server.py app.py ./
COPY web ./web
COPY assets ./assets

# SETUPTOOLS_SCM_PRETEND_VERSION: .git is not copied into the image, so
# setuptools_scm cannot derive the version from tags.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir -e ".[server]"

EXPOSE 8000

# Model weights are downloaded on first request and cached in /root/.cache —
# mount a volume there to persist them across container restarts.
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
