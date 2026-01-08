from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import OpenGL.GL as GL


@dataclass
class RenderTarget:
    width: int
    height: int

    def __post_init__(self) -> None:
        self._fbo = GL.glGenFramebuffers(1)
        self._color_tex = GL.glGenTextures(1)
        self._mask_tex = GL.glGenTextures(1)
        self._depth_tex = GL.glGenTextures(1)
        self._create_buffers()

    def _create_buffers(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        # Color attachment (RGB)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._color_tex, 0)

        # Mask attachment (RGB for encoded IDs)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._mask_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self._mask_tex, 0)

        # Depth attachment
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._depth_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT32F, self.width, self.height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self._depth_tex, 0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status:#x}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def bind_color_pass(self, *, clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glClearColor(*clear_color)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def bind_mask_pass(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT1)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def unbind(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self._delete_textures()
        self._color_tex = GL.glGenTextures(1)
        self._mask_tex = GL.glGenTextures(1)
        self._depth_tex = GL.glGenTextures(1)
        self._create_buffers()

    def _delete_textures(self) -> None:
        GL.glDeleteTextures([self._color_tex])
        GL.glDeleteTextures([self._mask_tex])
        GL.glDeleteTextures([self._depth_tex])

    def read_color(self) -> np.ndarray:
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4).copy()

    def read_mask(self) -> np.ndarray:
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._mask_tex)
        data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4).copy()

    def read_depth(self) -> np.ndarray:
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._depth_tex)
        data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        return np.frombuffer(data, dtype=np.float32).reshape(self.height, self.width).copy()

    def __del__(self) -> None:
        try:
            GL.glDeleteFramebuffers([self._fbo])
            self._delete_textures()
        except Exception:
            pass
