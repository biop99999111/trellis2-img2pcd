"""이미지 -> GLB 생성 백엔드. 여기만 GPU 를 쓴다.

img2pcd.py 의 3~6단계(mm 스케일·스무딩·샘플링·PCD·검증)는 백엔드와 무관하게
GLB 하나만 받으면 되므로, 모델을 갈아끼워도 그대로 재사용된다.

  trellis2   microsoft/TRELLIS.2-4B  — 품질 최상위. gated 의존성이 **둘**이다:
             facebook/dinov3-...(이미지 인코더, manual 승인) 와
             briaai/RMBG-2.0(배경 제거, 즉시 승인 · 비상업). 후자는 --rembg 로 교체 가능.
             텍스처가 run() 안에 포함돼 별도 단계가 없다.
  hunyuan3d  tencent/Hunyuan3D-2.1   — gated 의존성 없음. shape 10GB / texture 21GB.
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------- 공통


def _free_cuda() -> None:
    """다음 모델을 올리기 전에 VRAM 을 확실히 비운다.

    24GB 급에서는 이걸 안 하면 두 번째 부품이나 texture 단계에서 터진다.
    """
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_vram() -> float:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        return 0.0


def _find_repo(names: list[str], env_var: str, explicit: str | None, marker: str) -> Path | None:
    """모델 레포 위치를 찾는다(pip 패키지가 아니라 소스 트리를 쓰는 프로젝트들이라 필요)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(env_var):
        candidates.append(Path(os.environ[env_var]))
    here = Path(__file__).resolve().parent
    for name in names:
        candidates += [here.parent / name, Path("/workspace") / name, Path.home() / name, here / name]
    for c in candidates:
        if (c / marker).exists():
            return c.resolve()
    return None


def _accepted_kwargs(fn, wanted: dict) -> dict:
    """호출 대상이 실제로 받는 인자만 남긴다(버전차로 죽는 것 방지)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted)
    ok = {k: v for k, v in wanted.items() if k in params}
    dropped = set(wanted) - set(ok)
    if dropped:
        print(f"[gen] 호출이 안 받는 인자 제외: {sorted(dropped)}")
    return ok


def _report_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"[gen] gpu: {p.name} {p.total_memory / 2**30:.1f} GiB")
    except Exception:
        pass


# ---------------------------------------------------------------- TRELLIS.2

# transformers 5.x 는 DINOv3ViTModel 의 트랜스포머 블록을 encoder 하위
# (model.model.layer)로 옮겼는데 TRELLIS.2 원본은 model.layer 만 안다.
# setup.sh 가 transformers 를 핀 없이 깔기 때문에 오늘 설치하면 5.x 가 들어와
# `AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'` 로 죽는다
# (업스트림 이슈 #147 · PR #148/#156, 셋 다 미머지).
_DINOV3_HINT = (
    "DINOv3 레이어 경로를 찾지 못했습니다. transformers 버전 문제일 가능성이 큽니다 "
    "— requirements.txt 의 `transformers>=4.56,<5` 핀이 유지됐는지 확인하세요."
)


def _patch_dinov3_layout() -> None:
    """DinoV3FeatureExtractor 를 transformers 4.x / 5.x 양쪽에서 돌게 만든다.

    v5 에서 바뀐 것은 레이어의 '위치'뿐이고 embeddings·rope·layer 의 forward 규약은
    그대로다. 그래서 원본과 같은 연산을 그대로 재현할 수 있다.

    model.forward() 로 우회하지 않는 이유: 그 경로는 마지막에 학습된 affine 을 가진
    self.norm 을 태우는데 원본은 affine 없는 F.layer_norm 을 쓴다. 조건 특징이
    달라지면 "TRELLIS.2 가 Hunyuan3D 보다 나은가"라는 질문 자체가 오염된다.
    """
    from trellis2.modules import image_feature_extractor as ife

    if getattr(ife.DinoV3FeatureExtractor, "_layout_patched", False):
        return

    import torch.nn.functional as F

    def extract_features(self, image):
        model = self.model
        layers = getattr(model, "layer", None)          # transformers 4.56~4.57
        if layers is None:
            layers = getattr(getattr(model, "model", None), "layer", None)  # 5.x
        if layers is None or not hasattr(model, "embeddings") or not hasattr(model, "rope_embeddings"):
            raise SystemExit(_DINOV3_HINT)
        image = image.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = model.embeddings(image, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(image)
        for layer_module in layers:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    ife.DinoV3FeatureExtractor.extract_features = extract_features
    ife.DinoV3FeatureExtractor._layout_patched = True
    print("[gen] DINOv3 레이어 접근 패치 적용 (transformers 4.x/5.x 공용)")


def _patch_rembg(model_name: str) -> None:
    """배경 제거 모델을 갈아끼운다.

    pipeline.json 은 briaai/RMBG-2.0 을 박아두는데 이게 gated 라 승인 없이는
    from_pretrained 단계에서 401 로 죽는다(비상업 라이선스이기도 하다).
    구조가 같은 ZhengPeng7/BiRefNet(MIT)로 바꾸면 승인 없이 돈다.
    """
    from trellis2.pipelines import rembg

    if getattr(rembg.BiRefNet, "_forced_model", None) == model_name:
        return
    original = getattr(rembg.BiRefNet, "_original_init", rembg.BiRefNet.__init__)

    def __init__(self, model_name_arg: str = "", **kwargs):
        if model_name_arg and model_name_arg != model_name:
            print(f"[gen] rembg 교체: {model_name_arg} -> {model_name}")
        original(self, model_name, **kwargs)

    rembg.BiRefNet._original_init = original
    rembg.BiRefNet.__init__ = __init__
    rembg.BiRefNet._forced_model = model_name


def _transformers_note() -> str:
    try:
        import transformers

        version = transformers.__version__
    except Exception:
        return "transformers 미설치"
    major = int(version.split(".")[0]) if version[:1].isdigit() else 0
    warn = "  <- 5.x. 패치로 돌지만 4.57.x 를 권장" if major >= 5 else ""
    return f"transformers {version}{warn}"


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _load_failure_hint(exc: BaseException) -> str:
    """from_pretrained 가 죽었을 때 gated 401 을 알아보게 만든다."""
    text = str(exc)
    lines = [f"TRELLIS.2 파이프라인 로드 실패: {type(exc).__name__}: {text[:400]}"]
    if "401" in text or "gated" in text.lower() or "restricted" in text.lower():
        lines += [
            "",
            "gated 레포 접근 실패로 보입니다. `python check_env.py --hf` 로 어느 레포인지 확인하세요.",
            "  · facebook/dinov3-...      이미지 인코더. Meta 수동 승인 필요",
            "  · briaai/RMBG-2.0          배경 제거. 약관 동의 즉시 통과",
            "    (RMBG-2.0 을 쓰지 않으려면 --rembg ZhengPeng7/BiRefNet)",
        ]
    return chr(10).join(lines)


class Trellis2Backend:
    name = "trellis2"
    default_model = "microsoft/TRELLIS.2-4B"

    def __init__(self, st, repo_dir: str | None = None):
        self.st = st
        self.repo_dir = repo_dir
        self.pipeline = None

    def load(self):
        try:
            import trellis2  # noqa: F401
        except ImportError:
            # trellis2 는 pip 패키지가 아니라 TRELLIS.2 레포 안의 소스 디렉터리다.
            found = _find_repo(["TRELLIS.2"], "TRELLIS_DIR", self.repo_dir, "trellis2")
            if found is None:
                raise SystemExit(
                    "trellis2 를 찾을 수 없습니다.\n"
                    "  --repo-dir /workspace/TRELLIS.2  또는  export TRELLIS_DIR=..."
                ) from None
            sys.path.insert(0, str(found))
            print(f"[gen] sys.path 에 TRELLIS.2 추가: {found}")

        print(f"[gen] {_transformers_note()}")
        _patch_dinov3_layout()

        # 배경 제거 모델. 기본값(None)은 pipeline.json 의 briaai/RMBG-2.0 을 그대로 쓴다.
        rembg_model = self.st.rembg_model or os.environ.get("TRELLIS2_REMBG")
        if rembg_model:
            _patch_rembg(rembg_model)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        model_id = self.st.model_id or self.default_model
        print(f"[gen] loading {model_id} ...")
        t0 = time.time()
        try:
            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
        except Exception as exc:
            raise SystemExit(_load_failure_hint(exc)) from exc
        self.pipeline.cuda()
        print(f"[gen] loaded in {time.time() - t0:.1f}s")
        print(f"[gen] run signature: {inspect.signature(self.pipeline.run)}")
        _report_gpu()
        return self

    def _run_kwargs(self, pipeline_type: str | None) -> dict:
        wanted: dict = {}
        if self.st.seed is not None:
            wanted["seed"] = self.st.seed
        if pipeline_type:
            wanted["pipeline_type"] = pipeline_type
        if self.st.max_num_tokens:
            wanted["max_num_tokens"] = self.st.max_num_tokens
        wanted.update(self.st.run_kwargs)
        return _accepted_kwargs(self.pipeline.run, wanted)

    def generate(self, image_path: Path, out_glb: Path, st) -> dict:
        import o_voxel
        from PIL import Image

        image = Image.open(image_path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        print(f"[gen] {image_path.name} {image.size} {image.mode}")

        # 24GB 급에서 1024_cascade 가 터지면 빌린 GPU 세션을 통째로 날린다.
        # 한 번은 가벼운 해상도로 자동 재시도해서 최소한 형상 판정은 얻는다(이슈 #188).
        attempts = [st.pipeline_type]
        if st.oom_fallback and st.pipeline_type != st.oom_fallback_type:
            attempts.append(st.oom_fallback_type)

        mesh, t_gen, used_type = None, 0.0, attempts[0]
        for i, ptype in enumerate(attempts):
            kwargs = self._run_kwargs(ptype)
            _free_cuda()
            t0 = time.time()
            try:
                mesh = self.pipeline.run(image, **kwargs)[0]
            except Exception as exc:
                if not _is_oom(exc) or i == len(attempts) - 1:
                    raise
                print(f"[gen] OOM at pipeline_type={ptype} -> {attempts[i + 1]} 로 재시도")
                _free_cuda()
                continue
            t_gen = time.time() - t0
            used_type = ptype
            break
        print(f"[gen] mesh in {t_gen:.1f}s (pipeline_type={used_type})")

        if st.simplify_target:
            mesh.simplify(st.simplify_target)

        t1 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=st.decimation_target,
            texture_size=st.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        t_glb = time.time() - t1
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(out_glb), extension_webp=True)
        print(f"[gen] glb in {t_glb:.1f}s -> {out_glb}")

        peak = _peak_vram()
        # 이슈 #188: 샘플링 중간 텐서가 남아 메시 후처리에서 24GB 급이 터진다.
        del mesh, glb
        _free_cuda()
        print(f"[gen] peak vram: {peak:.2f} GiB")
        return {
            "backend": self.name,
            "pipeline_type": used_type,
            "rembg_model": self.st.rembg_model
            or os.environ.get("TRELLIS2_REMBG")
            or "briaai/RMBG-2.0",
            "t_generate_s": round(t_gen, 2),
            "t_glb_s": round(t_glb, 2),
            "peak_vram_gib": round(peak, 2),
            "textured": True,
        }

# ---------------------------------------------------------------- Hunyuan3D-2.1


class Hunyuan3DBackend:
    name = "hunyuan3d"
    default_model = "tencent/Hunyuan3D-2.1"

    def __init__(self, st, repo_dir: str | None = None):
        self.st = st
        self.repo_dir = repo_dir
        self.shape_pipeline = None
        self.root: Path | None = None

    def _ensure_paths(self) -> Path:
        root = _find_repo(
            ["Hunyuan3D-2.1", "Hunyuan3D-2"], "HUNYUAN3D_DIR", self.repo_dir, "hy3dshape"
        )
        if root is None:
            raise SystemExit(
                "Hunyuan3D-2.1 레포를 찾을 수 없습니다.\n"
                "  bash setup_hunyuan3d.sh  로 설치하거나\n"
                "  --repo-dir /workspace/Hunyuan3D-2.1  또는  export HUNYUAN3D_DIR=..."
            )
        # 공식 demo.py 와 동일하게 소스 디렉터리를 path 에 넣는다.
        for sub in ("hy3dshape", "hy3dpaint"):
            p = str(root / sub)
            if p not in sys.path:
                sys.path.insert(0, p)
        self.root = root
        print(f"[gen] Hunyuan3D 레포: {root}")
        return root

    def load(self):
        self._ensure_paths()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        model_id = self.st.model_id or self.default_model
        print(f"[gen] loading {model_id} (shape) ...")
        t0 = time.time()
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id)
        print(f"[gen] loaded in {time.time() - t0:.1f}s")
        print(f"[gen] shape call signature: {inspect.signature(self.shape_pipeline.__call__)}")
        _report_gpu()
        return self

    @staticmethod
    def _prepare_image(image_path: Path, work_dir: Path) -> Path:
        """배경 제거. 공식 demo.py 와 같은 규칙(RGB 면 rembg, RGBA 면 그대로)."""
        from PIL import Image

        image = Image.open(image_path)
        if image.mode == "RGBA":
            # 알파가 전부 불투명하면 마스크가 없는 것과 같으므로 RGB 로 낮춰 rembg 를 태운다.
            import numpy as np

            alpha = np.array(image)[:, :, 3]
            if (alpha == 255).all():
                print("[gen] RGBA 지만 알파가 전부 불투명 -> RGB 로 변환해 배경 제거")
                image = image.convert("RGB")
            else:
                return image_path

        try:
            from hy3dshape.rembg import BackgroundRemover
        except ImportError:
            try:
                from hy3dshape.utils.rembg import BackgroundRemover  # 배치에 따라 경로가 다름
            except ImportError:
                print("[gen] BackgroundRemover 를 찾지 못함 — 원본 이미지를 그대로 사용")
                return image_path

        t0 = time.time()
        out = BackgroundRemover()(image.convert("RGB"))
        work_dir.mkdir(parents=True, exist_ok=True)
        prepared = work_dir / "input_nobg.png"
        out.save(prepared)
        print(f"[gen] 배경 제거 {time.time() - t0:.1f}s -> {prepared.name}")
        return prepared

    def generate(self, image_path: Path, out_glb: Path, st) -> dict:
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_image(image_path, out_glb.parent)

        wanted: dict = {}
        if st.seed is not None:
            wanted["seed"] = st.seed
        wanted.update(st.run_kwargs)
        kwargs = _accepted_kwargs(self.shape_pipeline.__call__, wanted)

        _free_cuda()
        t0 = time.time()
        mesh = self.shape_pipeline(image=str(prepared), **kwargs)[0]
        t_gen = time.time() - t0
        peak_shape = _peak_vram()
        print(f"[gen] mesh in {t_gen:.1f}s (peak {peak_shape:.2f} GiB)")

        shape_glb = out_glb.parent / "mesh_shape.glb"
        mesh.export(str(shape_glb))
        print(f"[gen] shape glb -> {shape_glb}")

        info = {
            "backend": self.name,
            "t_generate_s": round(t_gen, 2),
            "peak_vram_gib": round(peak_shape, 2),
            "textured": False,
        }

        if not st.texture:
            # 텍스처를 건너뛰면 색이 없다 -> PCD 의 rgb 가 회색으로 채워진다.
            import shutil

            shutil.copy(shape_glb, out_glb)
            print("[gen] texture 생략 (--texture 로 활성화). PCD 색은 회색이 된다")
            del mesh
            _free_cuda()
            return info

        # shape 파이프라인을 완전히 내려야 texture(21GB)가 24GB 안에 들어간다.
        # paint 가 메시를 '경로'로 받기 때문에 이렇게 나눌 수 있다.
        del mesh
        self.shape_pipeline = None
        _free_cuda()
        print("[gen] shape 파이프라인 해제 -> texture 로드")

        from texture_existing import paint_existing
        import shutil

        texture_out = out_glb.parent / 'paint'
        painted = paint_existing(shape_glb, prepared, texture_out, self.root,
                                 views=st.paint_views, resolution=st.paint_resolution,
                                 texture_size=st.texture_size)
        shutil.copy2(texture_out / 'textured_mesh.glb', out_glb)
        _free_cuda()
        info.update(
            t_texture_s=painted['elapsed_seconds'],
            peak_vram_texture_gib=painted['peak_vram_gib'],
            textured=True,
            texture_validation=painted,
        )
        return info


# ---------------------------------------------------------------- 선택


BACKENDS = {"trellis2": Trellis2Backend, "hunyuan3d": Hunyuan3DBackend}


def load_backend(st, repo_dir: str | None = None):
    cls = BACKENDS.get(st.backend)
    if cls is None:
        raise SystemExit(f"모르는 백엔드: {st.backend} (가능: {', '.join(BACKENDS)})")
    return cls(st, repo_dir).load()
