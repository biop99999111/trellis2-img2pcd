# 기존 범퍼 메시의 색상·질감 생성

형상 생성이 끝난 `out_bumper/bumper_cover/mesh.glb`와 `input_nobg.png`를 재사용한다.
모델을 다시 생성하지 않고 Hunyuan3D paint 모델만 로드한다. 첫 실행에는 paint 가중치와 DINOv2가 추가로 다운로드된다.

Jupyter Terminal, 기존 RTX 4090 / hunyuan3d 환경에서:

```bash
cd /workspace/trellis2-img2pcd
git pull --ff-only
bash setup_texture.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hunyuan3d
python texture_existing.py --mesh out_bumper/bumper_cover/mesh.glb --image out_bumper/bumper_cover/input_nobg.png --out out_bumper_textured
```

다운로드할 파일은 `out_bumper_textured/textured_mesh.glb`다.
색상(albedo), 금속성(metallic), 거칠기(roughness) 이미지와 UV를 GLB 내부에 포함한다.
OBJ가 필요하면 `textured_model.zip`을 다운로드하고 압축을 풀어 OBJ·MTL·JPG를 같은 폴더에 둔다.
뷰어에서는 재질 미리보기/렌더링 모드로 확인한다. 단색 Solid 모드에서는 텍스처가 보이지 않을 수 있다.

기본값은 4만 면으로 단순화한 형상, 6뷰, 생성 해상도 512, 텍스처 2048이다.
재생성 모델 없이 기존 메시를 단순화하는 것이며, 원래 면을 그대로 유지하려면 `--faces 0`을 추가한다.
단순화와 UV 전개로 정점·면 개수 및 정점 순서가 바뀔 수 있다. 기존 점군 라벨과 직접 대응하지 않는다.
입력 메시의 좌표 체계를 유지한다. 기본 입력 `mesh.glb`는 정규화 좌표이며 mm 스케일이 아니다.
재실행 시에는 출력 폴더 이름을 바꾼다. 원본 및 기존 출력은 덮어쓰지 않는다.

## 호환 처리

- NumPy 1.26.4를 유지한다.
- upstream mesh_utils의 최상위 bpy import만 메모리에서 선택 사항으로 바꾼다.
  Blender GLB 변환은 호출하지 않고 OBJ/MTL/텍스처로부터 trimesh로 PBR GLB를 만든다.
  upstream 소스 파일은 수정하지 않는다.
- BasicSR의 오래된 torchvision grayscale import에 현재 함수를 연결한다.
- paint 설정/업스케일러 체크포인트를 절대 경로로 지정한다.
- upstream은 OBJ를 쓰므로 `.obj` 경로를 전달한다. OBJ 파일 이름을 GLB로 바꾸는 방식은 사용하지 않는다.
- 출력 GLB를 다시 읽어 면·UV·색상 및 PBR 이미지가 실제로 포함되었는지 확인한다.
- `--check`는 모듈과 CUDA 확장 임포트를 확인한다. GPU 추론이나 VRAM 충분 여부를 보장하는 검사는 아니다.

## 검증 범위와 해석

CPU의 선택적 Blender import, 오류 전파, PBR 채널 배치 및 GLB 텍스처 되읽기를 회귀 테스트한다.
이번 texture-only 경로의 GPU 실행과 생성 품질은 서버에서 확인해야 한다.
사진에 없는 측면·뒷면은 모델이 추정한다. 형상의 부풀림이나 실제 치수 문제는 텍스처로 수정되지 않는다.
GPU 메모리가 부족하면 다른 GPU 작업이 없는지 확인하고,
새 폴더로 `--views 4 --resolution 384 --texture-size 1024`를 시도할 수 있다.

구현 확인에 사용한 upstream:
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/hy3dpaint/textureGenPipeline.py
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/hy3dpaint/DifferentiableRenderer/mesh_utils.py
