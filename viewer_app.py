from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import glfw
import OpenGL.GL as GL

try:
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
except ImportError as exc:  
    raise RuntimeError(
        "imgui library is required for the SimpleViewerApp UI controls."
        " Install it with 'pip install imgui[glfw]'."
    ) from exc

from .twoD.triangle_scene import TriangleScene
from .twoD.rectangle_scene import RectangleScene
from .twoD.pentagon_scene import PentagonScene
from .twoD.hexagon_scene import HexagonScene
from .twoD.circle_scene import CircleScene
from .twoD.ellipse_scene import EllipseScene
from .twoD.trapezoid_scene import TrapezoidScene
from .twoD.star_scene import StarScene
from .twoD.arrow_scene import ArrowScene
from .threeD.cube_scene import CubeScene
from .threeD.surface_scene import SurfaceScene
from .threeD.model_scene import ModelScene
from .threeD.sphere_scene import SphereScene
from .threeD.cylinder_scene import CylinderScene
from .threeD.tetrahedron_scene import TetrahedronScene
from .threeD.prism_scene import PrismScene
from .threeD.cone_scene import ConeScene
from .threeD.frustum_scene import FrustumScene
from .threeD.atom_scene import BohrAtomScene
from .threeD.molecule_scene import MoleculeScene
from .rendering import Renderer
from .control_panel import ControlPanel


def _normalize_patterns(filetypes: Tuple[Tuple[str, str], ...]) -> List[str]:
    patterns: List[str] = []
    for _desc, spec in filetypes:
        parts = spec.replace(';', ' ').split()
        patterns.extend(parts)
    return patterns or ['*']


def _matches_filetypes(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path.lower(), pattern.lower()) for pattern in patterns)


def _macos_file_dialog(title: str) -> Optional[str]:
    script = f'''tell application "System Events"
        activate
        try
            set theFile to choose file with prompt "{title.replace('"', '\"')}"
            return POSIX path of theFile
        on error
            return ""
        end try
    end tell'''
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    path = completed.stdout.strip()
    return path or None


def open_file_dialog(title: str, filetypes: Tuple[Tuple[str, str], ...]) -> Optional[str]:
    patterns = _normalize_patterns(filetypes)

    if sys.platform == "darwin":
        path = _macos_file_dialog(title)
        if path is None or path == "":
            return None
        if not _matches_filetypes(path, patterns):
            print("Selected file does not match supported formats.")
            return None
        return path


class SimpleViewerApp:
    def __init__(self, width: int = 1000, height: int = 720) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.width = width
        self.height = height

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)

        self.window = glfw.create_window(width, height, "Simple Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(self.window)
        GL.glClearColor(0.12, 0.14, 0.18, 1.0)

        self.state = "triangle"
        self.renderer = Renderer()
        self.triangle_scene = TriangleScene(self.renderer)
        self.rectangle_scene = RectangleScene(self.renderer)
        self.pentagon_scene = PentagonScene(self.renderer)
        self.hexagon_scene = HexagonScene(self.renderer)
        self.circle_scene = CircleScene(self.renderer)
        self.ellipse_scene = EllipseScene(self.renderer)
        self.trapezoid_scene = TrapezoidScene(self.renderer)
        self.star_scene = StarScene(self.renderer)
        self.arrow_scene = ArrowScene(self.renderer)
        self.cube_scene = CubeScene(self.renderer)
        self.surface_scene = SurfaceScene(self.renderer)
        self.model_scene: Optional[ModelScene] = None
        self.sphere_scene = SphereScene(self.renderer)
        self.cylinder_scene = CylinderScene(self.renderer)
        self.tetra_scene = TetrahedronScene(self.renderer)
        self.prism_scene = PrismScene(self.renderer)
        self.cone_scene = ConeScene(self.renderer)
        self.frustum_scene = FrustumScene(self.renderer)
        self.atom_scene = BohrAtomScene(self.renderer)
        self.molecule_scene = MoleculeScene(self.renderer)
        self.control_panel = ControlPanel()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Triangle")

        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.triangle_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.rectangle_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.pentagon_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.hexagon_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.circle_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.ellipse_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.trapezoid_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.star_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.arrow_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.cube_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.surface_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.sphere_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.cylinder_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.tetra_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.prism_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.cone_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.frustum_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.atom_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.molecule_scene.on_window_resize(
            width,
            height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.triangle_scene.reset()
        self.triangle_scene.reset_view()
        glfw.set_window_size_callback(self.window, self.on_resize)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_char_callback(self.window, self.on_char)

        self.imgui_renderer: Optional[GlfwRenderer] = None
        self._imgui_context = None
        self._imgui_installed_callbacks = False
        self._init_imgui()
        if self._imgui_installed_callbacks:
            # Re-apply our callbacks when the renderer attached its own handlers
            glfw.set_window_size_callback(self.window, self.on_resize)
            glfw.set_cursor_pos_callback(self.window, self.on_cursor)
            glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
            glfw.set_scroll_callback(self.window, self.on_scroll)
            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_char_callback(self.window, self.on_char)

    def _ensure_imgui_context(self) -> bool:
        if self._imgui_context is None:
            return False
        imgui.set_current_context(self._imgui_context)
        return True

    def _init_imgui(self) -> None:
        self._imgui_context = imgui.create_context()
        imgui.set_current_context(self._imgui_context)
        try:
            self.imgui_renderer = GlfwRenderer(self.window, attach_callbacks=False)
            self._imgui_installed_callbacks = False
        except TypeError:
            # Older imgui versions use a different keyword; fall back to defaults
            self.imgui_renderer = GlfwRenderer(self.window)
            self._imgui_installed_callbacks = True
        io = imgui.get_io()
        io.display_size = (float(self.width), float(self.height))
        if hasattr(io, "ini_filename"):
            io.ini_filename = None
        elif hasattr(io, "ini_file_name"):
            io.ini_file_name = None

    # Texture loading functions removed in simplified shader setup

    # ------------------------------------------------------------------ state transitions
    def _request_quit(self) -> None:
        glfw.set_window_should_close(self.window, True)

    def enter_triangle(self) -> None:
        self.state = "triangle"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Triangle")

    def enter_cube(self) -> None:
        self.state = "cube"
        self.cube_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Cube")

    def enter_surface(self) -> None:
        self.state = "surface"
        self.surface_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Surface")

    def enter_rectangle(self) -> None:
        self.state = "rectangle"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Rectangle")

    def enter_pentagon(self) -> None:
        self.state = "pentagon"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Pentagon")

    def enter_hexagon(self) -> None:
        self.state = "hexagon"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Hexagon")

    def enter_circle(self) -> None:
        self.state = "circle"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Circle")

    def enter_ellipse(self) -> None:
        self.state = "ellipse"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Ellipse")
    
    def enter_trapezoid(self) -> None:
        self.state = "trapezoid"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Trapezoid")

    def enter_star(self) -> None:
        self.state = "star"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Star")

    def enter_arrow(self) -> None:
        self.state = "arrow"
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Arrow")

    def enter_model(self) -> None:
        if self.model_scene is None:
            return
        self.state = "model"
        self.model_scene.reset_view()
        if "Model" in self.control_panel.SHAPES:
            self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Model")

    def enter_sphere(self) -> None:
        self.state = "sphere"
        self.sphere_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Sphere")

    def enter_cylinder(self) -> None:
        self.state = "cylinder"
        self.cylinder_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Cylinder")

    def enter_tetrahedron(self) -> None:
        self.state = "tetrahedron"
        self.tetra_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Tetrahedron")

    def enter_prism(self) -> None:
        self.state = "prism"
        self.prism_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Prism")

    def enter_cone(self) -> None:
        self.state = "cone"
        self.cone_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Cone")

    def enter_frustum(self) -> None:
        self.state = "frustum"
        self.frustum_scene.reset_view()
        self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Frustum")

    def enter_atom(self) -> None:
        self.state = "atom"
        self.atom_scene.reset_view()
        if "Atom" in self.control_panel.SHAPES:
            self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Atom")

    def enter_molecule(self) -> None:
        self.state = "molecule"
        self.molecule_scene.reset_view()
        if "Molecule" in self.control_panel.SHAPES:
            self.control_panel.state.shape_idx = self.control_panel.SHAPES.index("Molecule")

    def _handle_shape_change(self, shape_name: str) -> None:
        if shape_name == "Triangle" and self.state != "triangle":
            self.enter_triangle()
        elif shape_name == "Cube" and self.state != "cube":
            self.enter_cube()
        elif shape_name == "Surface" and self.state != "surface":
            self.enter_surface()
        elif shape_name == "Rectangle" and self.state != "rectangle":
            self.enter_rectangle()
        elif shape_name == "Pentagon" and self.state != "pentagon":
            self.enter_pentagon()
        elif shape_name == "Hexagon" and self.state != "hexagon":
            self.enter_hexagon()
        elif shape_name == "Circle" and self.state != "circle":
            self.enter_circle()
        elif shape_name == "Ellipse" and self.state != "ellipse":
            self.enter_ellipse()
        elif shape_name == "Trapezoid" and self.state != "trapezoid":
            self.enter_trapezoid()
        elif shape_name == "Star" and self.state != "star":
            self.enter_star()
        elif shape_name == "Arrow" and self.state != "arrow":
            self.enter_arrow()
        elif shape_name == "Model" and self.state != "model":
            if self.model_scene is None:
                if not self._on_pick_model():
                    return
            self.enter_model()
        elif shape_name == "Sphere" and self.state != "sphere":
            self.enter_sphere()
        elif shape_name == "Cylinder" and self.state != "cylinder":
            self.enter_cylinder()
        elif shape_name == "Tetrahedron" and self.state != "tetrahedron":
            self.enter_tetrahedron()
        elif shape_name == "Prism" and self.state != "prism":
            self.enter_prism()
        elif shape_name == "Cone" and self.state != "cone":
            self.enter_cone()
        elif shape_name == "Frustum" and self.state != "frustum":
            self.enter_frustum()
        elif shape_name == "Atom" and self.state != "atom":
            self.enter_atom()
        elif shape_name == "Molecule" and self.state != "molecule":
            self.enter_molecule()

    # ------------------------------------------------------------------ callbacks
    def on_resize(self, _win, width: int, height: int) -> None:
        self.width = max(width, 1)
        self.height = max(height, 1)
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        GL.glViewport(0, 0, fb_width, fb_height)
        # Resize all instantiated scenes to keep projection/aspect in sync
        self.cube_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.triangle_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.rectangle_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.pentagon_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.hexagon_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.circle_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.ellipse_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.surface_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.atom_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.molecule_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        if self.model_scene is not None:
            self.model_scene.on_window_resize(
                self.width,
                self.height,
                framebuffer_width=fb_width,
                framebuffer_height=fb_height,
            )
        if self.imgui_renderer is not None and self._ensure_imgui_context():
            imgui.get_io().display_size = (float(self.width), float(self.height))

    def on_cursor(self, win, xpos: float, ypos: float) -> None:
        if self.imgui_renderer is not None and self._ensure_imgui_context():
            cursor_cb = getattr(self.imgui_renderer, "cursor_pos_callback", None)
            if cursor_cb is not None:
                cursor_cb(win, xpos, ypos)
            else:
                mouse_cb = getattr(self.imgui_renderer, "mouse_callback", None)
                if mouse_cb is not None:
                    mouse_cb(win, xpos, ypos)
        if self.imgui_renderer is not None and self._ensure_imgui_context() and imgui.get_io().want_capture_mouse:
            return

        if self.state == "cube":
            self.cube_scene.on_cursor(xpos, ypos)
        elif self.state == "triangle":
            self.triangle_scene.on_cursor(xpos, ypos)
        elif self.state == "surface":
            self.surface_scene.on_cursor(xpos, ypos)
        elif self.state == "rectangle":
            self.rectangle_scene.on_cursor(xpos, ypos)
        elif self.state == "pentagon":
            self.pentagon_scene.on_cursor(xpos, ypos)
        elif self.state == "hexagon":
            self.hexagon_scene.on_cursor(xpos, ypos)
        elif self.state == "circle":
            self.circle_scene.on_cursor(xpos, ypos)
        elif self.state == "ellipse":
            self.ellipse_scene.on_cursor(xpos, ypos)
        elif self.state == "trapezoid":
            self.trapezoid_scene.on_cursor(xpos, ypos)
        elif self.state == "star":
            self.star_scene.on_cursor(xpos, ypos)
        elif self.state == "arrow":
            self.arrow_scene.on_cursor(xpos, ypos)
        elif self.state == "model" and self.model_scene is not None:
            self.model_scene.on_cursor(xpos, ypos)
        elif self.state == "sphere":
            self.sphere_scene.on_cursor(xpos, ypos)
        elif self.state == "cylinder":
            self.cylinder_scene.on_cursor(xpos, ypos)
        elif self.state == "tetrahedron":
            self.tetra_scene.on_cursor(xpos, ypos)
        elif self.state == "prism":
            self.prism_scene.on_cursor(xpos, ypos)
        elif self.state == "cone":
            self.cone_scene.on_cursor(xpos, ypos)
        elif self.state == "frustum":
            self.frustum_scene.on_cursor(xpos, ypos)
        elif self.state == "atom":
            self.atom_scene.on_cursor(xpos, ypos)
        elif self.state == "molecule":
            self.molecule_scene.on_cursor(xpos, ypos)

    def on_mouse_button(self, win, button: int, action: int, mods: int) -> None:
        pressed = action == glfw.PRESS
        if self.imgui_renderer is not None and self._ensure_imgui_context():
            mouse_btn_cb = getattr(self.imgui_renderer, "mouse_button_callback", None)
            if mouse_btn_cb is not None:
                mouse_btn_cb(win, button, action, mods)
            if imgui.get_io().want_capture_mouse:
                return

        if self.state == "cube":
            self.cube_scene.on_mouse_button(button, pressed)
        elif self.state == "triangle":
            self.triangle_scene.on_mouse_button(button, pressed)
        elif self.state == "surface":
            self.surface_scene.on_mouse_button(button, pressed)
        elif self.state == "rectangle":
            self.rectangle_scene.on_mouse_button(button, pressed)
        elif self.state == "pentagon":
            self.pentagon_scene.on_mouse_button(button, pressed)
        elif self.state == "hexagon":
            self.hexagon_scene.on_mouse_button(button, pressed)
        elif self.state == "circle":
            self.circle_scene.on_mouse_button(button, pressed)
        elif self.state == "ellipse":
            self.ellipse_scene.on_mouse_button(button, pressed)
        elif self.state == "trapezoid":
            self.trapezoid_scene.on_mouse_button(button, pressed)
        elif self.state == "star":
            self.star_scene.on_mouse_button(button, pressed)
        elif self.state == "arrow":
            self.arrow_scene.on_mouse_button(button, pressed)
        elif self.state == "model" and self.model_scene is not None:
            self.model_scene.on_mouse_button(button, pressed)
        elif self.state == "sphere":
            self.sphere_scene.on_mouse_button(button, pressed)
        elif self.state == "cylinder":
            self.cylinder_scene.on_mouse_button(button, pressed)
        elif self.state == "tetrahedron":
            self.tetra_scene.on_mouse_button(button, pressed)
        elif self.state == "prism":
            self.prism_scene.on_mouse_button(button, pressed)
        elif self.state == "cone":
            self.cone_scene.on_mouse_button(button, pressed)
        elif self.state == "frustum":
            self.frustum_scene.on_mouse_button(button, pressed)
        elif self.state == "atom":
            self.atom_scene.on_mouse_button(button, pressed)
        elif self.state == "molecule":
            self.molecule_scene.on_mouse_button(button, pressed)

    def on_scroll(self, win, dx: float, dy: float) -> None:
        if self.imgui_renderer is not None and self._ensure_imgui_context():
            self.imgui_renderer.scroll_callback(win, dx, dy)
            if imgui.get_io().want_capture_mouse:
                return
        if self.state == "cube":
            self.cube_scene.on_scroll(dy)
        elif self.state == "triangle":
            self.triangle_scene.on_scroll(dy)
        elif self.state == "surface":
            self.surface_scene.on_scroll(dy)
        elif self.state == "rectangle":
            self.rectangle_scene.on_scroll(dy)
        elif self.state == "pentagon":
            self.pentagon_scene.on_scroll(dy)
        elif self.state == "hexagon":
            self.hexagon_scene.on_scroll(dy)
        elif self.state == "circle":
            self.circle_scene.on_scroll(dy)
        elif self.state == "ellipse":
            self.ellipse_scene.on_scroll(dy)
        elif self.state == "trapezoid":
            self.trapezoid_scene.on_scroll(dy)
        elif self.state == "star":
            self.star_scene.on_scroll(dy)
        elif self.state == "arrow":
            self.arrow_scene.on_scroll(dy)
        elif self.state == "model" and self.model_scene is not None:
            self.model_scene.on_scroll(dy)
        elif self.state == "sphere":
            self.sphere_scene.on_scroll(dy)
        elif self.state == "cylinder":
            self.cylinder_scene.on_scroll(dy)
        elif self.state == "tetrahedron":
            self.tetra_scene.on_scroll(dy)
        elif self.state == "prism":
            self.prism_scene.on_scroll(dy)
        elif self.state == "cone":
            self.cone_scene.on_scroll(dy)
        elif self.state == "frustum":
            self.frustum_scene.on_scroll(dy)
        elif self.state == "atom":
            self.atom_scene.on_scroll(dy)
        elif self.state == "molecule":
            self.molecule_scene.on_scroll(dy)

    def on_key(self, win, key: int, scancode: int, action: int, mods: int) -> None:
        if self.imgui_renderer is not None and self._ensure_imgui_context():
            self.imgui_renderer.keyboard_callback(win, key, scancode, action, mods)
            if imgui.get_io().want_capture_keyboard:
                return
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            self._request_quit()
            return

    def on_char(self, win, codepoint: int) -> None:
        if self.imgui_renderer is not None and self._ensure_imgui_context():
            self.imgui_renderer.char_callback(win, codepoint)

    def render_imgui_overlay(self) -> None:
        if self.imgui_renderer is None or not self._ensure_imgui_context():
            return
        self.imgui_renderer.process_inputs()
        imgui.new_frame()

        state_to_shape = {
            "triangle": "Triangle",
            "rectangle": "Rectangle",
            "pentagon": "Pentagon",
            "hexagon": "Hexagon",
            "circle": "Circle",
            "ellipse": "Ellipse",
            "trapezoid": "Trapezoid",
            "star": "Star",
            "arrow": "Arrow",
            "cube": "Cube",
            "surface": "Surface",
            "model": "Model",
            "sphere": "Sphere",
            "cylinder": "Cylinder",
            "tetrahedron": "Tetrahedron",
            "prism": "Prism",
            "cone": "Cone",
            "frustum": "Frustum",
            "atom": "Atom",
            "molecule": "Molecule",
        }
        if self.state in state_to_shape:
            active_shape = state_to_shape[self.state]
            self.control_panel.draw(
                active_shape=active_shape,
                triangle_scene=self.triangle_scene,
                cube_scene=self.cube_scene,
                surface_scene=self.surface_scene,
                rectangle_scene=self.rectangle_scene,
                pentagon_scene=self.pentagon_scene,
                hexagon_scene=self.hexagon_scene,
                circle_scene=self.circle_scene,
                ellipse_scene=self.ellipse_scene,
                trapezoid_scene=self.trapezoid_scene,
                star_scene=self.star_scene,
                arrow_scene=self.arrow_scene,
                model_scene=self.model_scene,
                sphere_scene=self.sphere_scene,
                cylinder_scene=self.cylinder_scene,
                tetra_scene=self.tetra_scene,
                prism_scene=self.prism_scene,
                cone_scene=self.cone_scene,
                frustum_scene=self.frustum_scene,
                atom_scene=self.atom_scene,
                molecule_scene=self.molecule_scene,
                on_shape_change=self._handle_shape_change,
                on_quit=self._request_quit,
                on_pick_texture=self._on_pick_texture,
                on_pick_model=self._on_pick_model,
            )

        imgui.end_frame()
        imgui.render()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.imgui_renderer.render(imgui.get_draw_data())
        GL.glDisable(GL.GL_BLEND)
        
    def _on_pick_texture(self) -> bool:
        path = open_file_dialog(
            "Select Texture Image",
            (("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tga"), ("All files", "*.*")),
        )
        if path is None:
            return False
        try:
            self.renderer.uma_texture.setup_texture("diffuse_tex", path)
        except Exception as exc:
            print(f"Failed to load texture from '{path}': {exc}")
            return False
        return True

    def _on_pick_model(self) -> bool:
        path = open_file_dialog(
            "Select Model (.obj/.ply)",
            (("Model files", "*.obj;*.ply"), ("All files", "*.*")),
        )
        if path is None:
            return False
        try:
            self.model_scene = ModelScene(self.renderer, path)
        except Exception as exc:
            print(f"Failed to load model from '{path}': {exc}")
            return False
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.model_scene.on_window_resize(
            self.width,
            self.height,
            framebuffer_width=fb_width,
            framebuffer_height=fb_height,
        )
        self.enter_model()
        return True


    def draw(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self.state == "triangle":
            self.triangle_scene.draw()
        elif self.state == "cube":
            self.cube_scene.draw()
        elif self.state == "surface":
            self.surface_scene.draw()
        elif self.state == "rectangle":
            self.rectangle_scene.draw()
        elif self.state == "pentagon":
            self.pentagon_scene.draw()
        elif self.state == "hexagon":
            self.hexagon_scene.draw()
        elif self.state == "circle":
            self.circle_scene.draw()
        elif self.state == "ellipse":
            self.ellipse_scene.draw()
        elif self.state == "trapezoid":
            self.trapezoid_scene.draw()
        elif self.state == "star":
            self.star_scene.draw()
        elif self.state == "arrow":
            self.arrow_scene.draw()
        elif self.state == "model" and self.model_scene is not None:
            self.model_scene.draw()
        elif self.state == "sphere":
            self.sphere_scene.draw()
        elif self.state == "cylinder":
            self.cylinder_scene.draw()
        elif self.state == "tetrahedron":
            self.tetra_scene.draw()
        elif self.state == "prism":
            self.prism_scene.draw()
        elif self.state == "cone":
            self.cone_scene.draw()
        elif self.state == "frustum":
            self.frustum_scene.draw()
        elif self.state == "atom":
            self.atom_scene.draw()
        elif self.state == "molecule":
            self.molecule_scene.draw()

        # Draw ImGui overlay after scene so the control panel is visible
        GL.glDisable(GL.GL_DEPTH_TEST)
        self.render_imgui_overlay()
        GL.glEnable(GL.GL_DEPTH_TEST)

    def run(self) -> None:
        print("Simple Viewer running. Use the on-screen ImGui controls or ESC to return/quit.")
        while not glfw.window_should_close(self.window):
            self.draw()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        self.shutdown()

    def shutdown(self) -> None:
        if self.imgui_renderer is not None:
            if self._ensure_imgui_context():
                self.imgui_renderer.shutdown()
            self.imgui_renderer = None
        if self._imgui_context is not None:
            imgui.set_current_context(self._imgui_context)
            imgui.destroy_context(self._imgui_context)
            self._imgui_context = None
        self.renderer.cleanup()
        glfw.terminate()


def run_app() -> None:
    app = SimpleViewerApp()
    app.run()
