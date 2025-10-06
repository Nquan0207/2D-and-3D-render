from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import imgui
import numpy as np

from .rendering import RenderMode
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


@dataclass
class ControlPanelState:
    render_mode_idx: int = 1
    shape_idx: int = 0


class ControlPanel:
    SHAPES: List[str] = [
        "Triangle",
        "Rectangle",
        "Pentagon",
        "Hexagon",
        "Circle",
        "Ellipse",
        "Trapezoid",
        "Star",
        "Arrow",
        "Cube",
        "Surface",
        "Model",
    ]
    RENDER_MODES: List[tuple[str, RenderMode]] = [
        ("Flat", RenderMode.FLAT),
        ("Gouraud", RenderMode.GOURAUD),
        ("Phong", RenderMode.PHONG),
        ("Texture", RenderMode.TEXTURE),
        ("Wireframe", RenderMode.WIREFRAME),
    ]

    def __init__(self) -> None:
        self.state = ControlPanelState()

    def draw(
        self,
        *,
        active_shape: str,
        triangle_scene: TriangleScene,
        cube_scene: CubeScene,
        surface_scene: SurfaceScene,
        rectangle_scene: RectangleScene | None = None,
        pentagon_scene: PentagonScene | None = None,
        hexagon_scene: HexagonScene | None = None,
        circle_scene: CircleScene | None = None,
        ellipse_scene: EllipseScene | None = None,
        trapezoid_scene: TrapezoidScene | None = None,
        star_scene: StarScene | None = None,
        arrow_scene: ArrowScene | None = None,
        model_scene: ModelScene | None = None,
        on_shape_change: Callable[[str], None],
        on_quit: Callable[[], None],
        on_pick_texture: Callable[[], bool] | None = None,
        on_pick_model: Callable[[], bool] | None = None,
    ) -> None:
        state = self.state
        margin = 20.0
        sidebar_width = 360.0
        imgui.set_next_window_position(margin, margin, imgui.ONCE)
        imgui.set_next_window_size(sidebar_width, 640.0, imgui.ONCE)
        window_flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
        # Use legacy-compatible signature to ensure window opens reliably across imgui versions
        if not imgui.begin("Control Panel", True, window_flags):
            imgui.end()
            return

        imgui.push_item_width(-1)

        # Pick scene based on active shape
        scene = triangle_scene
        if active_shape == "Triangle":
            scene = triangle_scene
        elif active_shape == "Rectangle" and rectangle_scene is not None:
            scene = rectangle_scene
        elif active_shape == "Pentagon" and pentagon_scene is not None:
            scene = pentagon_scene
        elif active_shape == "Hexagon" and hexagon_scene is not None:
            scene = hexagon_scene
        elif active_shape == "Circle" and circle_scene is not None:
            scene = circle_scene
        elif active_shape == "Ellipse" and ellipse_scene is not None:
            scene = ellipse_scene
        elif active_shape == "Trapezoid" and trapezoid_scene is not None:
            scene = trapezoid_scene
        elif active_shape == "Star" and star_scene is not None:
            scene = star_scene
        elif active_shape == "Arrow" and arrow_scene is not None:
            scene = arrow_scene
        elif active_shape == "Cube":
            scene = cube_scene
        elif active_shape == "Surface":
            scene = surface_scene
        elif active_shape == "Model" and model_scene is not None:
            scene = model_scene
        state.shape_idx = self.SHAPES.index(active_shape)
        state.render_mode_idx = self._render_mode_index(scene.mode)

        # Render mode combo
        imgui.text("Render Mode")
        render_labels = [label for label, _ in self.RENDER_MODES]
        changed_mode, new_mode_idx = imgui.combo("##render_mode", state.render_mode_idx, render_labels)
        if changed_mode and 0 <= new_mode_idx < len(self.RENDER_MODES):
            _, mode = self.RENDER_MODES[new_mode_idx]
            if mode == RenderMode.TEXTURE and not isinstance(scene, SurfaceScene) and on_pick_texture is not None:
                ok = on_pick_texture()
                if not ok:
                    new_mode_idx = state.render_mode_idx
                    mode = self.RENDER_MODES[new_mode_idx][1]
            scene.set_mode(mode)
            state.render_mode_idx = self._render_mode_index(scene.mode)

        imgui.separator()

        # Model loader controls
        if active_shape == "Model" and on_pick_model is not None:
            if imgui.button("Open Model (.obj/.ply)", width=-1):
                on_pick_model()
            imgui.separator()

        # Shape list
        imgui.text("Shapes")
        if imgui.begin_child("shape_list", 0, 220, True):
            for idx, shape_name in enumerate(self.SHAPES):
                is_selected = idx == state.shape_idx
                clicked, _ = imgui.selectable(shape_name, is_selected)
                if clicked and not is_selected:
                    state.shape_idx = idx
                    on_shape_change(shape_name)
            imgui.end_child()

        imgui.separator()

        if imgui.button("Quit Viewer", width=-1):
            on_quit()

        imgui.separator()

        # Transform controls
        if imgui.button("Reset Transform", width=-1):
            scene.reset_transform()
        
        if imgui.button("Reset Camera", width=-1):
            scene.reset_view()

        imgui.separator()

        imgui.text("Transform")
        translate = scene.get_translation().tolist()
        changed, translate = imgui.slider_float3(
            "Translate",
            translate[0],
            translate[1],
            translate[2],
            -10.0,
            10.0,
            "%.3f",
        )
        if changed:
            scene.set_translation(translate)

        rotation = scene.get_rotation().tolist()
        rot_changed, rotation = imgui.slider_float3(
            "Rotation (deg)",
            rotation[0],
            rotation[1],
            rotation[2],
            -360.0,
            360.0,
            "%.3f",
        )
        if rot_changed:
            scene.set_rotation(rotation)

        scale = scene.get_scale()
        scale_changed, scale = imgui.slider_float(
            "Scale",
            scale,
            0.1,
            10.0,
            "%.3f",
        )
        if scale_changed:
            scene.set_scale(scale)

        imgui.separator()

        # Appearance controls
        imgui.text("Appearance")
        base_color = scene.get_flat_color().tolist()
        color_changed, color = imgui.color_edit3("Base Color", *base_color)
        if color_changed:
            scene.set_flat_color(np.array(color, dtype=np.float32))
        if scene.mode == RenderMode.TEXTURE:
            tex_mix = getattr(scene, "tex_mix", 1.0)
            tex_mix_changed, tex_mix = imgui.slider_float(
                "Texture Mix",
                float(tex_mix),
                0.0,
                1.0,
                "%.3f",
            )
            if tex_mix_changed:
                setattr(scene, "tex_mix", float(tex_mix))
        imgui.separator()

        # Axes settings (if supported)
        if hasattr(scene, "show_axes"):
            show_axes = scene.show_axes
            toggled, show_axes = imgui.checkbox("Show Axes", show_axes)
            if toggled:
                scene.set_axes_settings(show=show_axes)
            if scene.show_axes:
                axes_scale_changed, axes_scale = imgui.slider_float(
                    "Axes Scale",
                    scene.axes_scale,
                    0.5,
                    3.5,
                    "%.3f",
                )
                if axes_scale_changed:
                    scene.set_axes_settings(scale=axes_scale)

        # Surface-specific controls
        if isinstance(scene, SurfaceScene):
            imgui.separator()
            imgui.text("Surface")
            expr_buf = getattr(self, "_surface_expr", scene.spec.expr)
            changed, expr_buf = imgui.input_text("f(x,y) =", expr_buf, 256)
            self._surface_expr = expr_buf
            if imgui.button("Apply Function", width=-1) or changed and imgui.is_key_pressed(257):
                ok, msg = scene.set_function(expr_buf)
                color = (0.3, 1.0, 0.3, 1.0) if ok else (1.0, 0.3, 0.3, 1.0)
                imgui.text_colored(msg, *color)
            res = scene.spec.resolution
            res_changed, res = imgui.slider_int("Resolution", res, 8, 256)
            if res_changed:
                scene.set_resolution(res)

        imgui.pop_item_width()
        imgui.end()

    @staticmethod
    def _render_mode_index(mode: RenderMode) -> int:
        for idx, (_, render_mode) in enumerate(ControlPanel.RENDER_MODES):
            if render_mode == mode:
                return idx
        return 0
