import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # comment out for MacOS

import trimesh
import pyrender
import numpy as np
from scipy.io import loadmat


class IUVRenderer:
    def __init__(self, resolution=(224, 224)):
        super().__init__()
        self.w, self.h = resolution
        uv_processed = loadmat('DensePose_COCO/densepose_uv_data/UV_Processed.mat')
        self.uv_vertices = uv_processed['All_vertices'][0]
        self.faces = uv_processed['All_Faces']
        self.face_indices = np.array(uv_processed['All_FaceIndices']).squeeze()
        self.u_norm = uv_processed['All_U_norm'].squeeze()
        self.v_norm = uv_processed['All_V_norm'].squeeze()

        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.w,
                                                   viewport_height=self.h,
                                                   point_size=1.0)
        self.camera = pyrender.camera.OrthographicCamera(xmag=1, ymag=1)

        self.vertex_uv_colors = np.stack((self.u_norm, self.v_norm, np.zeros(len(self.u_norm))), axis=1)
        self.face_i_colors = np.concatenate((np.expand_dims(self.face_indices, axis=1),
                                            np.zeros([len(self.faces), 2])), axis=1)/255

    def transform_vertices(self, vertices):
        """
        Vertex transformation to render correctly with the pyrender Orthographic Camera
        """
        vertices[:, 0] = (vertices[:, 0] - self.w / 2) * 2 / self.h
        vertices[:, 1] = (1 - vertices[:, 1] * 2 / self.h)
        vertices[:, 2] = -10-(vertices[:, 2] - vertices[:, 2].min()+1) * 2 / self.h
        return vertices

    def new_mesh_scene(self, mesh):
        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=np.zeros(3))
        scene.add(mesh, 'mesh')
        scene.add(self.camera)
        return scene

    def render_uv(self, vertices):

        mesh_uv = trimesh.Trimesh(vertices=vertices[self.uv_vertices-1],
                                  faces=self.faces-1,
                                  vertex_colors=self.vertex_uv_colors,
                                  process=False)

        mesh_uv = pyrender.Mesh.from_trimesh(mesh_uv)
        scene = self.new_mesh_scene(mesh_uv)
        uv, depth = self.renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

        return uv, depth

    def render_i(self, vertices):

        mesh_i = trimesh.Trimesh(vertices=vertices[self.uv_vertices-1],
                                 faces=self.faces-1,
                                 face_colors=self.face_i_colors,
                                 process=False)

        mesh_i = pyrender.Mesh.from_trimesh(mesh_i, smooth=False)
        scene = self.new_mesh_scene(mesh_i)
        i, depth = self.renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

        return i, depth

    def __call__(self, vertices):

        transformed_vertices = self.transform_vertices(vertices)

        rgb_uv, depth = self.render_uv(transformed_vertices)
        rgb_i, _ = self.render_i(transformed_vertices)

        segmentation = (depth > 0).astype(int)

        return rgb_i, rgb_uv, segmentation
