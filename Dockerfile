# Dockerfile
# Production-grade GPU build for DreamForge SD v1.5 MLOps Studio

# ── Stage 1: Base with CUDA 12.1 ──────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-venv \
    git curl libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Runtime Environment ──────────────────────────────────────
FROM base AS runtime

WORKDIR /app
RUN python3.10 -m venv venv
RUN /app/venv/bin/pip install --upgrade pip

RUN /app/venv/bin/pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN /app/venv/bin/pip install -r requirements.txt

# ── Stage 3: Application Build ────────────────────────────────────────
FROM runtime AS app

WORKDIR /app

ENV PATH="/app/venv/bin:$PATH"

COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY monitoring/ ./monitoring/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY lora_weights/ ./lora_weights/

RUN chmod +x /app/scripts/entrypoint.sh

RUN mkdir -p artifacts logs mlflow-artifacts model_cache

ENV VERSION=1.1.0 \
    MODEL_CACHE_DIR=/app/model_cache \
    OUTPUT_DIR=/app/artifacts/generated \
    MLFLOW_TRACKING_URI=http://mlflow:5012 \
    ENABLE_PROMETHEUS=true

ARG HF_TOKEN=""
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

COPY scripts/download_model.py ./scripts/download_model.py
RUN python3 scripts/download_model.py

EXPOSE 8501

ENTRYPOINT ["/app/scripts/entrypoint.sh"]