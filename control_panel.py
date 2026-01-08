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
from .threeD.sphere_scene import SphereScene
from .threeD.cylinder_scene import CylinderScene
from .threeD.tetrahedron_scene import TetrahedronScene
from .threeD.prism_scene import PrismScene
from .threeD.cone_scene import ConeScene
from .threeD.frustum_scene import FrustumScene
from .threeD.atom_scene import BohrAtomScene
from .threeD.molecule_scene import MoleculeScene


@dataclass
class ControlPanelState:
    render_mode_idx: int = 1
    shape_idx: int = 0
    collapsed: bool = False
    collapsed: bool = False


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
        "Sphere",
        "Cylinder",
        "Tetrahedron",
        "Prism",
        "Cone",
        "Frustum",
        "Atom",
        "Molecule",
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
        sphere_scene: SphereScene | None = None,
        cylinder_scene: CylinderScene | None = None,
        tetra_scene: TetrahedronScene | None = None,
        prism_scene: PrismScene | None = None,
        cone_scene: ConeScene | None = None,
        frustum_scene: FrustumScene | None = None,
        atom_scene: BohrAtomScene | None = None,
        molecule_scene: MoleculeScene | None = None,
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
        # Allow collapsing via the titlebar arrow next to the X button
        window_flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
        # Apply persisted collapsed state
        imgui.set_next_window_collapsed(self.state.collapsed, imgui.ALWAYS)
        # Use legacy-compatible signature to ensure window opens reliably across imgui versions
        if not imgui.begin("Control Panel", True, window_flags):
            imgui.end()
            return

        # Sync and early-out when collapsed
        self.state.collapsed = bool(imgui.is_window_collapsed())
        if self.state.collapsed:
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
        elif active_shape == "Sphere" and sphere_scene is not None:
            scene = sphere_scene
        elif active_shape == "Cylinder" and cylinder_scene is not None:
            scene = cylinder_scene
        elif active_shape == "Tetrahedron" and tetra_scene is not None:
            scene = tetra_scene
        elif active_shape == "Prism" and prism_scene is not None:
            scene = prism_scene
        elif active_shape == "Cone" and cone_scene is not None:
            scene = cone_scene
        elif active_shape == "Frustum" and frustum_scene is not None:
            scene = frustum_scene
        elif active_shape == "Atom" and atom_scene is not None:
            scene = atom_scene
        elif active_shape == "Molecule" and molecule_scene is not None:
            scene = molecule_scene
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

        if active_shape == "Atom" and atom_scene is not None:
            imgui.text("Bohr Atom")
            atom_names = atom_scene.available_atoms()
            if atom_names:
                try:
                    current_atom_idx = atom_names.index(atom_scene.atom_name)
                except ValueError:
                    current_atom_idx = 0
                changed_atom, new_atom_idx = imgui.combo("Element", current_atom_idx, atom_names)
                if changed_atom and 0 <= new_atom_idx < len(atom_names):
                    atom_scene.set_atom(atom_names[new_atom_idx])
            spacing = float(getattr(atom_scene, "shell_spacing", 1.0))
            changed_spacing, spacing = imgui.slider_float("Shell Spacing", spacing, 0.4, 2.0, "%.2f")
            if changed_spacing:
                atom_scene.set_shell_spacing(spacing)
            speed = float(getattr(atom_scene, "electron_speed", 1.0))
            changed_speed, speed = imgui.slider_float("Electron Speed", speed, 0.2, 4.0, "%.2f")
            if changed_speed:
                atom_scene.set_electron_speed(speed)
            show_orbits = bool(getattr(atom_scene, "show_orbits", True))
            toggled_orbits, show_orbits = imgui.checkbox("Show Orbit Guides", show_orbits)
            if toggled_orbits:
                atom_scene.set_show_orbits(show_orbits)
            imgui.separator()

        if active_shape == "Molecule" and molecule_scene is not None:
            imgui.text("Molecule")
            molecule_names = molecule_scene.available_molecules()
            if molecule_names:
                current_idx = int(max(0, min(len(molecule_names) - 1, getattr(molecule_scene, "molecule_index", 0))))
                changed_mol, new_idx = imgui.combo("Molecule Type", current_idx, molecule_names)
                if changed_mol and 0 <= new_idx < len(molecule_names):
                    molecule_scene.set_molecule(new_idx)
            atom_scale = float(getattr(molecule_scene, "atom_scale", 0.3))
            changed_scale, atom_scale = imgui.slider_float("Atom Radius Scale", atom_scale, 0.1, 1.0, "%.2f")
            if changed_scale:
                molecule_scene.set_atom_scale(atom_scale)
            bond_radius = float(getattr(molecule_scene, "bond_radius", 0.1))
            changed_bond, bond_radius = imgui.slider_float("Bond Radius", bond_radius, 0.02, 0.6, "%.2f")
            if changed_bond:
                molecule_scene.set_bond_radius(bond_radius)
            imgui.separator()

        # Shape list
        imgui.text("Shapes")
        # Dear ImGui expects positive width for child windows; compute available width
        avail_w = imgui.get_content_region_available().x
        safe_w = max(1.0, float(avail_w))
        # Always end child to keep ImGui stack balanced
        imgui.begin_child("shape_list", safe_w, 220.0, True)
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
            # Show contours toggle
            show_contours = getattr(scene, "show_contours", False)
            toggled, show_contours = imgui.checkbox("Show Contours", bool(show_contours))
            if toggled:
                scene.set_show_contours(show_contours)
            # Rolling ball toggle and params
            show_ball = getattr(scene, "show_ball", False)
            toggled_ball, show_ball = imgui.checkbox("Show Rolling Ball", bool(show_ball))
            if toggled_ball:
                scene.set_show_ball(show_ball)
            if getattr(scene, "show_ball", False):
                # Dynamic rolling toggle
                dyn = bool(getattr(scene, "ball_dyn", False))
                toggled_dyn, dyn = imgui.checkbox("Dynamic Rolling (physics)", dyn)
                if toggled_dyn:
                    scene.set_ball_dynamic(dyn)
                # Quick flow: Adjust -> Start (for dynamic rolling)
                if dyn:
                    if imgui.button("Adjust Position"):
                        # Enable freeze so user can tweak X/Y, then press Start
                        scene.set_ball_freeze(True)
                    if imgui.button("Start Rolling"):
                        # Start physics from current anchor (unfreezes internally)
                        if hasattr(scene, "drop_ball_from_anchor"):
                            scene.drop_ball_from_anchor()
                # Ball render mode (independent from surface)
                ball_modes = [("Flat", RenderMode.FLAT), ("Gouraud", RenderMode.GOURAUD), ("Phong", RenderMode.PHONG), ("Wireframe", RenderMode.WIREFRAME)]
                current_ball_mode = getattr(scene, "ball_mode", RenderMode.PHONG)
                labels = [label for label, _ in ball_modes]
                try:
                    current_idx = next(i for i, (_, m) in enumerate(ball_modes) if m == current_ball_mode)
                except StopIteration:
                    current_idx = 2
                changed_bm, new_idx = imgui.combo("Ball Mode", current_idx, labels)
                if changed_bm and 0 <= new_idx < len(ball_modes):
                    _, bm = ball_modes[new_idx]
                    scene.set_ball_mode(bm)
                # Parametric speed removed; dynamics or freeze control motion
                # Radius (in surface units)
                radius = float(getattr(scene, "ball_radius", 0.15))
                # Use xy-extent to give a sensible max
                try:
                    max_r = max(0.01, 0.25 * float(scene.get_xy_extent()))
                except Exception:
                    max_r = 1.0
                radius_changed, radius = imgui.slider_float("Ball Radius", radius, 0.01, max_r, "%.3f")
                if radius_changed:
                    scene.set_ball_params(radius=radius)
                if dyn:
                    # Gravity and damping for dynamics
                    g = float(getattr(scene, "ball_gravity", 9.8))
                    g_changed, g = imgui.slider_float("Gravity", g, 0.0, 30.0, "%.2f")
                    if g_changed:
                        scene.ball_gravity = g
                    damp = float(getattr(scene, "ball_damping", 0.2))
                    damp_changed, damp = imgui.slider_float("Damping", damp, 0.0, 2.0, "%.2f")
                    if damp_changed:
                        scene.ball_damping = damp
                # Freeze/inspect mode
                freeze = bool(getattr(scene, "ball_freeze", False))
                toggled_freeze, freeze = imgui.checkbox("Freeze Ball (inspect)", freeze)
                if toggled_freeze:
                    scene.set_ball_freeze(freeze)
                if freeze:
                    # Anchor sliders to position the frozen ball on domain
                    ax, ay = scene.get_ball_anchor()
                    s = scene.spec
                    ax_changed, ax = imgui.slider_float("Ball X", float(ax), float(s.x_min), float(s.x_max), "%.3f")
                    ay_changed, ay = imgui.slider_float("Ball Y", float(ay), float(s.y_min), float(s.y_max), "%.3f")
                    if ax_changed or ay_changed:
                        scene.set_ball_anchor(ax, ay)
            # Preset functions for quick selection
            presets = [
                ("Himmelblau", "(x*x + y - 11.0)**2 + (x + y*y - 7.0)**2"),
                ("Rosenbrock", "100.0*(y - x*x)**2 + (1.0 - x)**2"),
                ("Quadratic Bowl", "x*x + y*y"),
                ("Booth", "(x + 2.0*y - 7.0)**2 + (2.0*x + y - 5.0)**2"),
            ]
            preset_labels = [name for name, _ in presets]
            preset_idx = getattr(self, "_surface_preset_idx", -1)
            changed_preset, preset_idx = imgui.combo("Preset", max(preset_idx, -1), ["<none>"] + preset_labels)
            if changed_preset:
                self._surface_preset_idx = preset_idx
                if preset_idx > 0:
                    # Apply selected preset into the input buffer
                    _, expr = presets[preset_idx - 1]
                    self._surface_expr = expr
                    # Apply immediately
                    ok, msg = scene.set_function(expr)
                    color = (0.3, 1.0, 0.3, 1.0) if ok else (1.0, 0.3, 0.3, 1.0)
                    imgui.text_colored(msg, *color)

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
