"""Paint an existing Hunyuan mesh; export embedded PBR GLB without Blender/bpy."""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import types
import zipfile

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def optional_blender_source(source, filename):
    """Only make the upstream top-level bpy import optional, leaving mesh IO intact."""
    tree = ast.parse(source, filename)
    replacement = ast.parse('try:\n import bpy\nexcept ModuleNotFoundError as exc:\n if exc.name != "bpy":\n  raise\n bpy = None\n').body[0]
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == 'bpy':
            tree.body[i] = ast.copy_location(replacement, node)
    return compile(ast.fix_missing_locations(tree), filename, 'exec')


def require_supported_torch(version):
    from packaging.version import Version
    if Version(version.split('+')[0]) < Version('2.6.0'):
        raise RuntimeError('Texture checkpoint loading requires torch >=2.6.0. Run bash setup_texture.sh; do not bypass the torch.load safety check.')


def load_paint_modules(root):
    """Compatibility is scoped to this process; upstream files are not edited."""
    import torch
    require_supported_torch(torch.__version__)
    root = Path(root).resolve()
    paint_root = root / 'hy3dpaint'
    sys.path.insert(0, str(paint_root))
    # BasicSR 1.4.2 still imports the old torchvision module path.
    import torchvision.transforms.functional as functional
    try:
        importlib.import_module('torchvision.transforms.functional_tensor')
    except ModuleNotFoundError as exc:
        if exc.name != 'torchvision.transforms.functional_tensor':
            raise
        compat = types.ModuleType('torchvision.transforms.functional_tensor')
        compat.rgb_to_grayscale = functional.rgb_to_grayscale
        sys.modules[compat.__name__] = compat
    name = 'DifferentiableRenderer.mesh_utils'
    source = paint_root / 'DifferentiableRenderer' / 'mesh_utils.py'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(optional_blender_source(source.read_text(encoding='utf-8'), str(source)), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    pipeline = importlib.import_module('textureGenPipeline')
    params = inspect.signature(pipeline.Hunyuan3DPaintPipeline.__call__).parameters
    if not {'save_glb', 'use_remesh'}.issubset(params):
        raise RuntimeError('Unsupported paint API: save_glb/use_remesh parameters required')
    # The upstream renderer otherwise swallows missing-extension imports.
    importlib.import_module('custom_rasterizer')
    inpaint = importlib.import_module('DifferentiableRenderer.mesh_inpaint_processor')
    if not hasattr(inpaint, 'meshVerticeInpaint'):
        raise RuntimeError('Missing meshVerticeInpaint extension')
    from realesrgan import RealESRGANer
    import xatlas
    # Dino_v2 imports Lightning lazily during model loading. Check that chain
    # now so missing pkg_resources/Lightning dependencies fail before weights load.
    importlib.import_module('pkg_resources')
    importlib.import_module('pytorch_lightning')
    importlib.import_module('hunyuanpaintpbr.unet.modules')
    return pipeline


def export_pbr_glb(obj_path, glb_path):
    """Pack the upstream albedo/metallic/roughness maps in a standard GLB."""
    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    obj_path, glb_path = Path(obj_path), Path(glb_path)
    albedo_path = obj_path.with_suffix('.jpg')
    metallic_path = obj_path.with_name(obj_path.stem + '_metallic.jpg')
    roughness_path = obj_path.with_name(obj_path.stem + '_roughness.jpg')
    for path in (obj_path, obj_path.with_suffix('.mtl'), albedo_path, metallic_path, roughness_path):
        if not path.is_file():
            raise FileNotFoundError(f'Paint output is incomplete: {path}')
    mesh = trimesh.load(str(obj_path), force='mesh', process=False)
    uv = getattr(mesh.visual, 'uv', None)
    if uv is None or len(uv) != len(mesh.vertices) or not np.isfinite(uv).all():
        raise ValueError('Textured OBJ has no valid UV coordinates')
    albedo = Image.open(albedo_path).convert('RGB')
    rough = Image.open(roughness_path).convert('L')
    metal = Image.open(metallic_path).convert('L')
    if rough.size != metal.size:
        raise ValueError('Metallic and roughness map sizes differ')
    # glTF: red unused, green roughness, blue metallic.
    packed = np.zeros((rough.height, rough.width, 3), np.uint8)
    packed[..., 1], packed[..., 2] = np.asarray(rough), np.asarray(metal)
    material = PBRMaterial(name='Bumper_PBR', baseColorTexture=albedo,
                           metallicRoughnessTexture=Image.fromarray(packed),
                           baseColorFactor=[255, 255, 255, 255], metallicFactor=1.0, roughnessFactor=1.0)
    mesh.visual = TextureVisuals(uv=uv, material=material)
    mesh.export(str(glb_path))
    restored = trimesh.load(str(glb_path), force='mesh', process=False)
    result_material = getattr(restored.visual, 'material', None)
    if len(restored.faces) != len(mesh.faces) or getattr(restored.visual, 'uv', None) is None:
        raise IOError('GLB lost faces or UV coordinates')
    if result_material is None or result_material.baseColorTexture is None or result_material.metallicRoughnessTexture is None:
        raise IOError('GLB lost embedded PBR textures')
    if not np.array_equal(np.asarray(result_material.baseColorTexture.convert('RGB')), np.asarray(albedo)):
        raise IOError('GLB albedo readback differs')
    if not np.array_equal(np.asarray(result_material.metallicRoughnessTexture.convert('RGB')), packed):
        raise IOError('GLB metallic/roughness readback differs')
    return {'vertices': len(mesh.vertices), 'faces': len(mesh.faces), 'textured': True,
            'albedo_size': list(albedo.size), 'pbr_readback_verified': True}


def paint_existing(mesh_path, image_path, out, repo_dir, views=6, resolution=512, texture_size=2048, faces=40000):
    import numpy as np
    import torch
    import trimesh
    mesh_path, image_path, out, repo_dir = map(lambda p: Path(p).resolve(), (mesh_path, image_path, out, repo_dir))
    if not mesh_path.is_file() or not image_path.is_file():
        raise FileNotFoundError('Input mesh and reference image must exist')
    if out.exists():
        raise FileExistsError(f'Use a new output folder: {out}')
    if not torch.cuda.is_available():
        raise RuntimeError('Texture inference requires a CUDA GPU')
    if not 1 <= views <= 6 or resolution < 64 or texture_size < 64 or faces < 0:
        raise ValueError('Invalid views/resolution/texture-size/faces')
    pipeline_module = load_paint_modules(repo_dir)
    config = pipeline_module.Hunyuan3DPaintConfig(max_num_view=views, resolution=resolution)
    config.multiview_cfg_path = str(repo_dir / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
    config.realesrgan_ckpt_path = str(repo_dir / 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
    for path in (config.multiview_cfg_path, config.realesrgan_ckpt_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    # Upstream save_mesh(..., downsample=True) halves the baked texture.
    config.texture_size = texture_size * 2
    config.render_size = texture_size
    mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Input is not a finite triangle mesh')
    input_faces = len(mesh.faces)
    if faces and len(mesh.faces) > faces:
        mesh = mesh.simplify_quadric_decimation(face_count=faces)
    out.mkdir(parents=True)
    prepared = out / 'input_shape.obj'
    mesh.export(str(prepared))
    original_bounds = mesh.bounds.tolist()
    paint_faces = len(mesh.faces)
    del mesh
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pipeline = pipeline_module.Hunyuan3DPaintPipeline(config)
    # Match upstream OBJ writer; never pass a .glb suffix to save_mesh.
    obj = out / 'textured_mesh.obj'
    pipeline(str(prepared), image_path=str(image_path), output_mesh_path=str(obj), use_remesh=False, save_glb=False)
    del pipeline
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.empty_cache()
    info = export_pbr_glb(obj, out / 'textured_mesh.glb')
    info.update(input_mesh=str(mesh_path), reference_image=str(image_path), input_faces=input_faces,
                paint_input_faces=paint_faces, input_bounds=original_bounds,
                units='input coordinates; normalized Hunyuan mesh is not millimeters',
                views=views, resolution=resolution, texture_size=texture_size,
                elapsed_seconds=round(time.perf_counter() - start, 2), peak_vram_gib=round(peak, 2))
    (out / 'texture_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    with zipfile.ZipFile(out / 'textured_model.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            if path.is_file() and (path.name.startswith('textured_mesh') or path.name == 'texture_manifest.json'):
                bundle.write(path, path.name)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=Path)
    ap.add_argument('--image', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--repo-dir', type=Path, default=Path('/workspace/Hunyuan3D-2.1'))
    ap.add_argument('--views', type=int, default=6)
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--texture-size', type=int, default=2048)
    ap.add_argument('--faces', type=int, default=40000, help='Simplification target; 0 preserves input faces before UV unwrap')
    ap.add_argument('--check', action='store_true', help='Check paint imports/extensions without downloading GPU models')
    args = ap.parse_args(argv)
    if args.check:
        load_paint_modules(args.repo_dir)
        print('Texture imports and extensions OK (Blender not required)')
        return 0
    if any(p is None for p in (args.mesh, args.image, args.out)):
        ap.error('--mesh, --image and --out are required unless --check is used')
    print(json.dumps(paint_existing(args.mesh, args.image, args.out, args.repo_dir,
                                   args.views, args.resolution, args.texture_size, args.faces), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
