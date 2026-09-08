#!/usr/bin/env bash
# Add texture support to the existing hunyuan3d Python 3.10 environment.
set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_DIR="${HUNYUAN3D_DIR:-$(dirname "$PROJECT_DIR")/Hunyuan3D-2.1}"
command -v conda >/dev/null || { echo 'conda required'; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hunyuan3d
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
fi
command -v nvcc >/dev/null || { echo 'nvcc required: use a CUDA devel template'; exit 1; }
test -f "$HY_DIR/hy3dpaint/textureGenPipeline.py" || { echo "Missing Hunyuan source: $HY_DIR"; exit 1; }
python -m pip install -r "$PROJECT_DIR/requirements-texture.txt"
python -m pip install pybind11 ninja
# ABI-specific extensions must use this environment's Python/torch.
(cd "$HY_DIR/hy3dpaint/custom_rasterizer" && python -m pip install --no-build-isolation -e .)
(cd "$HY_DIR/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh)
mkdir -p "$HY_DIR/hy3dpaint/ckpt"
CHECKPOINT="$HY_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
if [ ! -s "$CHECKPOINT" ]; then
    curl --fail --location --retry 3 https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o "$CHECKPOINT.partial"
    mv "$CHECKPOINT.partial" "$CHECKPOINT"
fi
python "$PROJECT_DIR/texture_existing.py" --repo-dir "$HY_DIR" --check
echo 'Texture environment ready. Run texture_existing.py in the hunyuan3d environment.'
