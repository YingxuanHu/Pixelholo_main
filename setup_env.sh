#!/usr/bin/env bash
# Bootstrap the shared PixelHolo Python environment.
# Usage:
#   bash setup_env.sh

set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_PATH="${ROOT_DIR}/.venv"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH. Install Miniconda/Mambaforge first." >&2
  exit 1
fi

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "[env] Creating new conda env at ${ENV_PATH}"
  conda create -y --prefix "${ENV_PATH}" python=3.10 >/dev/null
else
  echo "[env] Reusing existing env at ${ENV_PATH}"
fi

conda config --set auto_activate_base false >/dev/null

echo "[env] Installing core system libraries (ffmpeg, sox, postgresql, cudnn, openfst, kaldi)"
conda install -y -p "${ENV_PATH}" -c conda-forge ffmpeg sox postgresql cudnn openfst kaldi >/dev/null

echo "[env] Upgrading pip"
conda run -p "${ENV_PATH}" python -m pip install --upgrade pip >/dev/null

echo "[env] Installing nightly PyTorch stack (CUDA 13.0 / sm_90+)"
conda run -p "${ENV_PATH}" pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
conda run -p "${ENV_PATH}" pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 >/dev/null

echo "[env] Installing project requirements"
conda run -p "${ENV_PATH}" pip install \
  joblib==1.3.2 \
  fastapi uvicorn[standard] websockets \
  numpy==1.24.4 scipy==1.10.1 numba soundfile librosa==0.10.2.post1 \
  pydub tqdm pyyaml rich huggingface_hub \
  webrtcvad demucs pynini munch tensorboard transformers einops-exts monotonic-align auraloss pesq scikit-learn==1.2.2 faster-whisper montreal-forced-aligner==2.2.17 pgvector hdbscan \
  praatio jiwer torchmetrics onnx onnxruntime-gpu speechbrain >/dev/null

echo "[env] All dependencies installed. Activate with:"
echo "       conda activate ${ENV_PATH}"
