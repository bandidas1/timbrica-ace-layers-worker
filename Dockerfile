# RunPod serverless worker: «Улучшить минус» — ACE-Step 1.5 `lego` layers over a person's backing track.
#
# ACE-Step 1.5 (github.com/ace-step/ACE-Step-1.5, MIT — code and weights) pinned to the commit the app repo's bench
# ran on (ca1e85f, 29.08.2026: after upstream PR #1287, which stopped the LM's codes from replacing src_audio in
# lego). Its own lock file (uv.lock) installs the exact Linux stack: torch 2.10.0+cu128 (Blackwell included).
#
# Baked weights (a cold start pulls the image, never a checkpoint): acestep-v15-base DiT + VAE + Qwen3-Embedding-0.6B,
# and Meta's audiobox-aesthetics (CC BY 4.0) that judges each take by the mix it makes.
# The 5 Hz LM is NOT needed (lego runs with thinking off). acestep-v15-xl-base (20 GB) is fetched at runtime only
# when a bench job asks for it — production uses the baked base.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false ACE_ROOT=/app/ace \
    UV_PROJECT_ENVIRONMENT=/app/ace/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common git curl ca-certificates ffmpeg libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

ARG ACE_COMMIT=ca1e85fe9430179831e6bc6be790c332190a3866
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /app/ace \
    && cd /app/ace && git checkout ${ACE_COMMIT} && rm -rf .git \
    && uv sync --frozen --no-dev --python python3.11 \
    && uv pip install --python /app/ace/.venv/bin/python runpod==1.7.9 requests librosa==0.11.0 audiobox_aesthetics==0.0.4 \
        panns_inference==0.1.1 torchlibrosa==0.1.0 matplotlib \
    && /app/ace/.venv/bin/python -c "import torch, acestep, runpod, librosa, audiobox_aesthetics; print('torch', torch.__version__)"

# PANNs Cnn14 (Kong et al., weights CC BY 4.0, zenodo.org/records/3987831) + the AudioSet labels: the instrument
# check. panns_inference reads both from ~/panns_data (and would fetch the labels with wget at import) — baked, pinned.
RUN mkdir -p /root/panns_data \
    && curl -fsSL -o '/root/panns_data/Cnn14_mAP=0.431.pth' 'https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1' \
    && curl -fsSL -o /root/panns_data/class_labels_indices.csv 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv' \
    && echo '0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31  /root/panns_data/Cnn14_mAP=0.431.pth' | sha256sum -c - \
    && echo 'cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429  /root/panns_data/class_labels_indices.csv' | sha256sum -c - \
    && /app/ace/.venv/bin/python -c "from panns_inference import AudioTagging; AudioTagging(checkpoint_path='/root/panns_data/Cnn14_mAP=0.431.pth', device='cpu'); print('panns ok')"

WORKDIR /app/worker
COPY download_models.py .
RUN /app/ace/.venv/bin/python download_models.py

COPY layers_io.py score.py handler.py io_selftest.py score_selftest.py ./
RUN /app/ace/.venv/bin/python score_selftest.py

CMD ["/app/ace/.venv/bin/python", "-u", "/app/worker/handler.py"]
