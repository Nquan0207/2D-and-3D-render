from __future__ import annotations

import pathlib

import numpy as np
import OpenGL.GL as GL

from ..libs.buffer import UManager, VAO
from ..libs.shader import Shader


class LineAxes:
    def __init__(self) -> None:
        base = pathlib.Path(__file__).resolve().parent.parent
        vert = str(base / "shaders" / "axis.vert")
        frag = str(base / "shaders" / "axis.frag")
        self.shader = Shader(vert, frag)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        length = 1.0
        positions = np.array(
            [
                [0.0, 0.0, 0.0], [length, 0.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, length, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, length],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.4, 1.0], [0.0, 0.4, 1.0],
            ],
            dtype=np.float32,
        )
        self.vertex_count = positions.shape[0]

        self.vao.add_vbo(0, positions, ncomponents=3, dtype=GL.GL_FLOAT,
            normalized=False, stride=0, offset=None,)
        self.vao.add_vbo(1, colors, ncomponents=3,dtype=GL.GL_FLOAT,
            normalized=False,stride=0,offset=None,)

        try:
            self._line_width_range = GL.glGetFloatv(GL.GL_SMOOTH_LINE_WIDTH_RANGE)
        except GL.error.GLError:
            self._line_width_range = (1.0, 1.0)

    def draw(self, projection: np.ndarray, view: np.ndarray, model: np.ndarray, line_width: float = 2.0) -> None:
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection.astype(np.float32), "projection", True)
        self.uma.upload_uniform_matrix4fv(view.astype(np.float32), "view", True)
        self.uma.upload_uniform_matrix4fv(model.astype(np.float32), "model", True)

        try:
            min_w, max_w = float(self._line_width_range[0]), float(self._line_width_range[1])
        except (TypeError, IndexError):
            min_w, max_w = 1.0, 1.0
        safe_width = max(min_w, min(line_width, max_w))
        GL.glLineWidth(safe_width)
        self.vao.activate()
        GL.glDrawArrays(GL.GL_LINES, 0, self.vertex_count)
        self.vao.deactivate()
        GL.glLineWidth(1.0)
