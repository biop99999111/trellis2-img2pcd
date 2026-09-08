"""CPU checks for the texture export adapter; inference requires the server GPU."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import SimpleMaterial
from texture_existing import export_pbr_glb, optional_blender_source


class TextureTests(unittest.TestCase):
    def test_optional_bpy_keeps_mesh_code(self):
        scope = {}
        with patch.dict('sys.modules', {'bpy': None}):
            exec(optional_blender_source('import bpy\ndef load_mesh():\n return 42\n', 'upstream.py'), scope)
        self.assertIsNone(scope['bpy'])
        self.assertEqual(scope['load_mesh'](), 42)

    def test_other_import_failure_is_not_hidden(self):
        with self.assertRaises(ModuleNotFoundError):
            exec(optional_blender_source('import not_a_real_texture_dependency\n', 'upstream.py'), {})

    def test_embedded_pbr_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            obj = folder / 'textured_mesh.obj'
            mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
            mesh.visual = TextureVisuals(uv=[[0, 0], [1, 0], [0, 1]], material=SimpleMaterial(image=Image.new('RGB', (8, 8), (20, 30, 40))))
            mesh.export(obj)
            obj.with_suffix('.mtl').write_text('newmtl Material\nmap_Kd textured_mesh.jpg\n', encoding='ascii')
            Image.new('RGB', (8, 8), (20, 30, 40)).save(obj.with_suffix('.jpg'))
            Image.new('L', (8, 8), 64).save(folder / 'textured_mesh_metallic.jpg')
            Image.new('L', (8, 8), 192).save(folder / 'textured_mesh_roughness.jpg')
            info = export_pbr_glb(obj, folder / 'textured_mesh.glb')
            self.assertTrue(info['pbr_readback_verified'])
            loaded = trimesh.load(folder / 'textured_mesh.glb', force='mesh', process=False)
            packed = np.asarray(loaded.visual.material.metallicRoughnessTexture)
            np.testing.assert_array_equal(packed[0, 0, :3], [0, 192, 64])

    def test_missing_texture_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                export_pbr_glb(Path(td) / 'missing.obj', Path(td) / 'out.glb')


if __name__ == '__main__':
    unittest.main()
