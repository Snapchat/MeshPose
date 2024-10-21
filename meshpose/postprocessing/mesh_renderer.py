import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # comment out for MacOS

import trimesh
import pyrender
import numpy as np
from scipy.io import loadmat


PATH_UV_MAT = 'third_party/densepose_eval/DensePoseData/UV_data/UV_Processed.mat'
MESH_COLOR = (185 / 255, 190 / 255, 255 / 255, 1.0)


class MeshRenderer:
    def __init__(self, resolution):
        super().__init__()
        self.w, self.h = resolution
        if not os.path.exists(PATH_UV_MAT):
            raise FileNotFoundError('Please download densepose_eval')
        uv_processed = loadmat(PATH_UV_MAT)
        self.uv_vertices = uv_processed['All_vertices'][0]
        self.faces = uv_processed['All_Faces']

        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.w,
                                                   viewport_height=self.h,
                                                   point_size=1.0)
        self.camera = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=10000)
        self.material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
                                                           alphaMode='OPAQUE',
                                                           baseColorFactor=MESH_COLOR)

    def transform_vertices(self, vertices):
        """
        Vertex transformation to render correctly with the pyrender Orthographic Camera
        """
        vertices[:, 0] = (vertices[:, 0] - self.w / 2) * 2 / self.h
        vertices[:, 1] = (1 - vertices[:, 1] * 2 / self.h)
        vertices[:, 2] = -10 - (vertices[:, 2] - vertices[:, 2].min()+1) * 2 / self.h
        return vertices

    def new_mesh_scene(self):
        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        scene.add(self.camera, pose=np.eye(4))
        # Add directional lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        return scene

    def render_mesh(self, scene, vertices):
        for vertices_ in vertices:
            mesh = trimesh.Trimesh(vertices=vertices_[self.uv_vertices-1],
                                   faces=self.faces-1,
                                   process=False)
            mesh = pyrender.Mesh.from_trimesh(mesh,
                                              material=self.material,
                                              smooth=True)
            scene.add(mesh, "mesh")
        color, depth = self.renderer.render(scene)

        return color, depth

    def __call__(self, image, vertices):

        transformed_vertices = [self.transform_vertices(vertices_) for vertices_ in vertices]

        scene = self.new_mesh_scene()
        rendering, depth = self.render_mesh(scene, transformed_vertices)

        mask = np.expand_dims(depth > 0, 2)
        output_image = mask * rendering + np.logical_not(mask) * image

        return output_image
