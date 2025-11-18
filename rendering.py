from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional
import numpy as np
import OpenGL.GL as GL

from .libs.buffer import VAO
from .libs.shader import Shader
from .libs.buffer import UManager

class RenderMode(IntEnum):
    FLAT = 0
    GOURAUD = 1
    PHONG = 2
    TEXTURE = 3
    WIREFRAME = 4


@dataclass
class RenderSettings:
    mode: RenderMode
    flat_color: np.ndarray
    light_pos: np.ndarray
    shininess: float
    light_ambient: np.ndarray
    light_diffuse: np.ndarray
    light_specular: np.ndarray
    mat_ambient: np.ndarray
    mat_diffuse: np.ndarray
    mat_specular: np.ndarray
    tex_id: Optional[int] = None
    tex_mix: float = 1.0


class Mesh:
    def __init__(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
        indices: Optional[np.ndarray] = None,
        primitive: int = GL.GL_TRIANGLES,
    ) -> None:
        self.vao = VAO()
        self.indexed = indices is not None
        self.count = 0
        self.primitive = primitive

        self.vao.add_vbo(0, positions.astype(np.float32), ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, colors.astype(np.float32), ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, normals.astype(np.float32), ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(3, texcoords.astype(np.float32), ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        if indices is not None:
            self.vao.add_ebo(indices.astype(np.uint32))
            self.count = int(indices.size)
        else:
            self.count = int(positions.shape[0])

    def draw(self) -> None:
        self.vao.activate()
        if self.indexed:
            GL.glDrawElements(self.primitive, self.count, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(self.primitive, 0, self.count)
        self.vao.deactivate()



class Renderer:
    def __init__(self) -> None:
        # Simple shaders (flat, gouraud, phong) from shaders folder
        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.shader_flat = Shader(str(shader_dir / "flat.vert"), str(shader_dir / "flat.frag"))
        self.uma_flat = UManager(self.shader_flat)
        self.shader_gouraud = Shader(
            str(shader_dir / "gouraud.vert"), str(shader_dir / "gouraud.frag")
        )
        self.uma_gouraud = UManager(self.shader_gouraud)
        self.shader_phong = Shader(
            str(shader_dir / "phong.vert"), str(shader_dir / "phong.frag")
        )
        self.uma_phong = UManager(self.shader_phong)
        self.shader_texture = Shader(
            str(shader_dir / "texture.vert"), str(shader_dir / "texture.frag")
        )
        self.uma_texture = UManager(self.shader_texture)


    def draw(
        self,
        mesh: Mesh,
        projection: np.ndarray,
        view: np.ndarray,
        model: np.ndarray,
        normal_matrix: np.ndarray,
        settings: RenderSettings,
    ) -> None:
        wireframe = settings.mode == RenderMode.WIREFRAME

        # Decide which shader to use
        active_mode = settings.mode
        if wireframe:
            # Use simple Gouraud shader for transformed wireframe drawing
            active_mode = RenderMode.GOURAUD

        

        # Simple shader paths
        if active_mode == RenderMode.FLAT:
            GL.glUseProgram(self.shader_flat.render_idx)
            modelview = (view @ model).astype(np.float32)
            self.uma_flat.upload_uniform_matrix4fv(projection.astype(np.float32), "projection", True)
            self.uma_flat.upload_uniform_matrix4fv(modelview, "modelview", True)
            self.uma_flat.upload_uniform_vector3fv(settings.flat_color.astype(np.float32), "flat_color")
        elif active_mode == RenderMode.GOURAUD:
            GL.glUseProgram(self.shader_gouraud.render_idx)
            # Provide basic transforms expected by the patch shader
            modelview = (view @ model).astype(np.float32)
            self.uma_gouraud.upload_uniform_matrix4fv(projection.astype(np.float32), "projection", True)
            self.uma_gouraud.upload_uniform_matrix4fv(modelview, "modelview", True)
            # normalMat exists in shader but unused; safe to skip or upload identity
        elif active_mode == RenderMode.PHONG:
            GL.glUseProgram(self.shader_phong.render_idx)
            modelview = (view @ model).astype(np.float32)
            # Normal matrix for modelview (3x3 -> embed in 4x4 as shader expects mat4)
            nm3 = np.linalg.inv(modelview[:3, :3]).T.astype(np.float32)
            normal_mat4 = np.eye(4, dtype=np.float32)
            normal_mat4[:3, :3] = nm3
            self.uma_phong.upload_uniform_matrix4fv(projection.astype(np.float32), "projection", True)
            self.uma_phong.upload_uniform_matrix4fv(modelview, "modelview", True)
            self.uma_phong.upload_uniform_matrix4fv(normal_mat4, "normalMat", True)

            # Build K_materials and I_light matrices (columns: diffuse, specular, ambient)
            # Build as row-major (diffuse/specular/ambient on rows), then transpose on upload
            K = np.stack([
                settings.mat_diffuse.astype(np.float32),
                settings.mat_specular.astype(np.float32),
                settings.mat_ambient.astype(np.float32),
            ], axis=0)
            I = np.stack([
                settings.light_diffuse.astype(np.float32),
                settings.light_specular.astype(np.float32),
                settings.light_ambient.astype(np.float32),
            ], axis=0)
            # Transpose=True to match GLSL column-major expectations
            self.uma_phong.upload_uniform_matrix3fv(K.astype(np.float32), "K_materials", False)
            self.uma_phong.upload_uniform_matrix3fv(I.astype(np.float32), "I_light", False)
            self.uma_phong.upload_uniform_scalar1f(float(settings.shininess), "shininess")
            # self.uma_phong.upload_uniform_vector3fv(settings.light_pos.astype(np.float32), "light_pos")
            light_world4 = np.array([
                float(settings.light_pos[0]),
                float(settings.light_pos[1]),
                float(settings.light_pos[2]),
                1.0,
            ], dtype=np.float32)
            light_eye4 = (view @ light_world4).astype(np.float32)
            light_eye3 = light_eye4[:3]
            self.uma_phong.upload_uniform_vector3fv(light_eye3, "light_pos")

        elif active_mode == RenderMode.TEXTURE:
            GL.glUseProgram(self.shader_texture.render_idx)
            modelview = (view @ model).astype(np.float32)
            self.uma_texture.upload_uniform_matrix4fv(projection.astype(np.float32), "projection", True)
            self.uma_texture.upload_uniform_matrix4fv(modelview, "modelview", True)
            self.uma_texture.upload_uniform_scalar1f(float(settings.tex_mix), "texture_mix")
            # Only bind and set sampler if we have a valid texture id
            if settings.tex_id is not None:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, settings.tex_id)
                self.uma_texture.upload_uniform_scalar1i(0, "diffuse_tex")

        # Polygon mode
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if wireframe else GL.GL_FILL)
        mesh.draw()
        if wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    def cleanup(self) -> None:
        pass
