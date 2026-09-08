#!/usr/bin/env bash
# vast.ai(Linux + CUDA) 에서 Hunyuan3D-2.1 + 이 스파이크를 세팅한다.
#
#   bash setup_hunyuan3d.sh
#
# TRELLIS.2 와 달리 gated 모델 의존성이 없어 승인 없이 바로 돈다.
# VRAM: shape 10GB / texture 21GB (24GB 에서는 shape 를 내리고 texture 를 올린다)
#
# set -u 는 쓰지 않는다 — conda 활성화 스크립트가 미설정 변수를 참조해서 죽는다.
set -eo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_DIR="${HUNYUAN3D_DIR:-$(dirname "$SPIKE_DIR")/Hunyuan3D-2.1}"
ENV_NAME="${ENV_NAME:-hunyuan3d}"

echo "=== 0. 환경 확인 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi 실패 — GPU 인스턴스가 맞는지 확인하세요"; exit 1
}
echo "설치 위치: $HY_DIR"
df -h "$(dirname "$HY_DIR")" | tail -1

if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi
echo "CUDA_HOME=${CUDA_HOME:-(unset)}"

echo
echo "=== 0-1. pip 인덱스 정리 ==="
# vast.ai 이미지/Hunyuan3D 설정에 중국 미러(mirrors.cloud.tencent.com,
# mirrors.aliyun.com)가 extra-index-url 로 걸려 있으면, 중국 밖 호스트에서는
# 응답이 몇 분씩 멈춘다(실측: basicsr 빌드 의존성 설치에서 무한 정지).
# PyPI 만 쓰도록 덮어쓴다. 중국 리전이면 PIP_KEEP_MIRRORS=1 로 이 절을 건너뛴다.
if [ -z "${PIP_KEEP_MIRRORS:-}" ]; then
  # PIP_EXTRA_INDEX_URL="" 로는 안 지워진다 — 미러가 pip 설정 파일에 박혀 있으면
  # 빈 환경변수가 그걸 덮지 못한다(실측: 인덱스 목록에 그대로 남음).
  # 설정 파일 자체를 무시하고 인덱스를 환경변수로만 준다.
  export PIP_CONFIG_FILE=/dev/null
  export PIP_INDEX_URL="https://pypi.org/simple"
  export PIP_DEFAULT_TIMEOUT=30
  export PIP_RETRIES=3
  echo "PIP_CONFIG_FILE=/dev/null · PIP_INDEX_URL=$PIP_INDEX_URL"
fi

echo
echo "=== 1. conda 훅 로드 ==="
# 비대화형 서브셸에서는 conda activate 가 그냥은 안 된다.
command -v conda >/dev/null 2>&1 || { echo "!! conda 없음"; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo
echo "=== 2. conda env '$ENV_NAME' (python 3.10) ==="
# Keep model environments separate. torch >=2.6 is required by current Transformers checkpoint loading.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "이미 있음: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
echo "python: $(command -v python)"

echo
echo "=== 3. Hunyuan3D-2.1 클론 ==="
if [ -d "$HY_DIR/.git" ]; then
  echo "이미 있음: $HY_DIR"
else
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY_DIR"
fi
cd "$HY_DIR"

echo
echo "=== 4. torch 2.6.0 (cu124) ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo
echo "=== 5. requirements ==="
# basicsr 은 소스 tarball 이라 빌드 격리 환경이 torch 를 **다시** 내려받는다(~2.5GB).
# env 에 이미 torch 2.6.0 이 있으므로 격리를 끄고 그걸 재사용한다.
pip install cython

# bpy==4.0 은 PyPI 에서 내려갔다. 남아 있는 최소 버전은 4.2.0 이고 python>=3.11 만
# 지원해서 이 env(3.10)에서는 어떤 버전도 못 깐다. pip 는 해결 실패 시 아무것도
# 설치하지 않으므로 이 한 줄 때문에 전체가 죽는다. shape 생성 경로에는 쓰이지
# 않으므로 빼고 설치한다(필요해지면 임포트 에러로 드러난다).
REQ_FILE=/tmp/hy_requirements_clean.txt
# requirements.txt 안에 --extra-index-url 로 중국 미러가 적혀 있으면 환경변수나
# 설정 파일로는 못 지운다. 실측: 미러 경유 1.1 MB/s vs 회선 실측 169 MB/s (150배).
# 중국 리전이면 PIP_KEEP_MIRRORS=1 로 원본을 그대로 쓴다.
if [ -n "${PIP_KEEP_MIRRORS:-}" ]; then
  grep -v '^bpy' requirements.txt > "$REQ_FILE"
else
  grep -v '^bpy' requirements.txt | grep -v 'index-url' > "$REQ_FILE"
fi
if ! diff -q requirements.txt "$REQ_FILE" >/dev/null 2>&1; then
  echo "requirements 조정: bpy 제외(PyPI 에서 4.0 삭제·4.2.0+ 는 python>=3.11)"
  [ -z "${PIP_KEEP_MIRRORS:-}" ] && echo "                  index-url 줄 제외(미러 스로틀 회피)"
fi
pip install -r "$REQ_FILE" --no-build-isolation

echo
echo "=== 6. texture 확장 빌드 (실패해도 shape 는 돈다) ==="
# custom_rasterizer / DifferentiableRenderer 는 --texture 에서만 쓰인다.
# 여기서 깨져도 shape-only 실행은 가능하므로 전체를 중단시키지 않는다.
TEXTURE_OK=1
(
  cd hy3dpaint/custom_rasterizer && pip install -e .
) || { echo "!! custom_rasterizer 빌드 실패 — --texture 는 못 씁니다"; TEXTURE_OK=0; }
(
  cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
) || { echo "!! DifferentiableRenderer 빌드 실패 — --texture 는 못 씁니다"; TEXTURE_OK=0; }

echo
echo "=== 7. RealESRGAN 체크포인트 (texture 용) ==="
mkdir -p hy3dpaint/ckpt
if [ -f hy3dpaint/ckpt/RealESRGAN_x4plus.pth ]; then
  echo "이미 있음"
else
  wget -q --show-progress \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P hy3dpaint/ckpt || echo "!! 다운로드 실패 — --texture 사용 시 필요합니다"
fi

echo
echo "=== 8. 스파이크 의존성 + Jupyter 커널 ==="
pip install -r "$SPIKE_DIR/requirements-hunyuan3d.txt"
pip install ipykernel
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo
echo "=== 9. 임포트 검증 ==="
cd "$HY_DIR"
python - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "hy3dshape"))
sys.path.insert(0, os.path.join(os.getcwd(), "hy3dpaint"))

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, f"{p.total_memory / 2**30:.1f} GiB")

missing = []
try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    print("  OK    hy3dshape (shape 생성)")
except Exception as e:
    print(f"  FAIL  hy3dshape: {e}")
    missing.append("hy3dshape")

try:
    from textureGenPipeline import Hunyuan3DPaintPipeline  # noqa: F401
    print("  OK    hy3dpaint (texture, --texture 옵션)")
except Exception as e:
    print(f"  WARN  hy3dpaint: {e}")
    print("        -> shape-only 는 정상 동작합니다(--texture 만 불가)")

for m in ("numpy", "onnxruntime", "rembg", "trimesh", "open3d", "yaml", "matplotlib"):
    try:
        __import__(m)
        print(f"  OK    {m}")
    except Exception as e:
        print(f"  FAIL  {m}: {e}")
        missing.append(m)

print("\n=> shape 실행 가능" if not missing else f"\n=> 실패: {missing}")
raise SystemExit(1 if missing else 0)
PY

cat <<EOF

=== 완료 ===
  cd $SPIKE_DIR
  conda activate $ENV_NAME

  # 1) shape 만 (빠름 · VRAM ~10GB · PCD 색은 회색)
  python img2pcd.py --backend hunyuan3d --only bumper_cover --out out

  # 2) 텍스처까지 (느림 · VRAM ~21GB · PCD 에 색이 들어감)
  python img2pcd.py --backend hunyuan3d --texture --only bumper_cover --out out

레포 자동 탐색이 안 되면:  --repo-dir $HY_DIR
산출물: out/<부품>/{mesh.glb, mesh_mm.ply, <부품>.pcd, preview.png, manifest.json}
EOF
