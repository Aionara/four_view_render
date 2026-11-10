bl_info = {
    "name": "Four View Render",
    "author": "Aero",
    "version": (1, 4, 0),
    "blender": (3, 1, 0),
    "location": "View3D > Sidebar > 四视图",
    "description": "创建十字四视图、对应摄像机并批量渲染（兼容 Blender 3.1 ~ 5.2.3+）",
    "category": "3D View",
}

import os
import tempfile
import bpy
from mathutils import Matrix, Vector

# (名称, 摄像机相对于中心的朝外方向 direction 即局部 +Z, 摄像机画面上方向 up 即局部 +Y)
CAMERAS = (
    ("正视图", Vector((0, -1, 0)), Vector((0, 0, 1))),
    ("侧视图", Vector((1, 0, 0)), Vector((0, 0, 1))),
    ("顶视图", Vector((0, 0, 1)), Vector((0, 1, 0))),
    ("底视图", Vector((0, 0, -1)), Vector((0, -1, 0))),
)

BOUND_OBJECT_TYPES = {
    'MESH', 'CURVE', 'SURFACE', 'FONT', 'META',
    'CURVES', 'POINTCLOUD', 'VOLUME', 'GPENCIL', 'GREASEPENCIL',
}


def bounds(context):
    objs = [
        o for o in context.scene.objects
        if o.type in BOUND_OBJECT_TYPES and not o.hide_render and getattr(o, "bound_box", None)
    ]
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    if not pts:
        return Vector((0, 0, 0)), 4.0
    lo = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
    return (lo + hi) * 0.5, max(max(hi[i] - lo[i] for i in range(3)) * 1.2, 1.0)


def camera_rotation_euler(direction, up):
    """Build an exact orthonormal camera basis from local +Z (direction) and local +Y (up).

    Avoids Vector.to_track_quat('-Z', 'Y'), which hardcodes world +Z as the up
    reference and hits a collinear singularity on Top/Bottom views.
    """
    back = direction.normalized()
    right = up.normalized().cross(back).normalized()
    up_ortho = back.cross(right).normalized()
    rot_mat = Matrix((right, up_ortho, back)).transposed()
    return rot_mat.to_euler('XYZ')


def get_camera(context, label, direction, up, center, scale):
    obj = bpy.data.objects.get(label)
    if obj is not None and obj.type != 'CAMERA':
        raise RuntimeError("对象名已被非摄像机占用：" + label)
    if obj is None:
        data = bpy.data.cameras.new(label + "_数据")
        obj = bpy.data.objects.new(label, data)
    if obj.name not in context.scene.objects:
        context.scene.collection.objects.link(obj)
    obj.name = label
    obj.data.name = label + "_数据"
    obj.data.type = 'ORTHO'
    obj.data.ortho_scale = scale
    obj.parent = None
    obj.rotation_mode = 'XYZ'
    obj.location = center + direction.normalized() * (scale * 2 + 1)
    obj.rotation_euler = camera_rotation_euler(direction, up)
    obj.hide_render = False
    if hasattr(obj, "hide_viewport"):
        obj.hide_viewport = False
    try:
        obj.hide_set(False)
    except Exception:
        pass
    obj["four_view_label"] = label
    return obj


def view_quat(view):
    axes = {
        'FRONT': (Vector((0, -1, 0)), Vector((0, 0, 1))),
        'RIGHT': (Vector((1, 0, 0)), Vector((0, 0, 1))),
        'TOP': (Vector((0, 0, 1)), Vector((0, 1, 0))),
        'BOTTOM': (Vector((0, 0, -1)), Vector((0, -1, 0))),
    }
    direction, up = axes[view]
    return camera_rotation_euler(direction, up).to_quaternion()


def _find_window_for_screen(screen):
    ctx_win = getattr(bpy.context, "window", None)
    if ctx_win is not None and getattr(ctx_win, "screen", None) == screen:
        return ctx_win
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            if win.screen == screen:
                return win
    for wm in bpy.data.window_managers:
        if wm.windows:
            return wm.windows[0]
    return ctx_win


def _call_op_with_override(op_callable, context=None, window=None, screen=None,
                           area=None, region=None, **op_kwargs):
    """Execute a Blender operator with context override across Blender 3.1 ~ 5.2.3+.

    - Blender 3.2+ supports `context.temp_override(...)` (and Blender 4.0+ removed dict overrides).
    - Blender 3.1 does not have `temp_override` and requires passing an override dict as the
      first positional argument to `bpy.ops.<module>.<op>(override_dict, **op_kwargs)`.
    """
    ctx = context if context is not None else bpy.context
    if hasattr(ctx, "temp_override"):
        override_kwargs = {}
        if window is not None:
            override_kwargs["window"] = window
        if screen is not None:
            override_kwargs["screen"] = screen
        if area is not None:
            override_kwargs["area"] = area
        if region is not None:
            override_kwargs["region"] = region
        with ctx.temp_override(**override_kwargs):
            return op_callable(**op_kwargs)

    # Fallback for Blender 3.1 (legacy dictionary context override)
    try:
        override = ctx.copy() if ctx is not None else bpy.context.copy()
        if not isinstance(override, dict):
            override = {}
    except Exception:
        override = {}

    if window is not None:
        override["window"] = window
        if getattr(window, "workspace", None) is not None:
            override["workspace"] = window.workspace
        if getattr(window, "scene", None) is not None:
            override["scene"] = window.scene
        if getattr(window, "view_layer", None) is not None:
            override["view_layer"] = window.view_layer
    if screen is not None:
        override["screen"] = screen
        if getattr(screen, "scene", None) is not None and override.get("scene") is None:
            override["scene"] = screen.scene
    if area is not None:
        override["area"] = area
        spaces = getattr(area, "spaces", None)
        if spaces is not None and getattr(spaces, "active", None) is not None:
            override["space_data"] = spaces.active
    if region is not None:
        override["region"] = region
        if getattr(region, "data", None) is not None:
            override["region_data"] = region.data
    override["blend_data"] = bpy.data
    return op_callable(override, **op_kwargs)


def _op_rna_prop_names(op_callable):
    """Return the set of property identifiers accepted by a bpy.ops operator."""
    try:
        return {p.identifier for p in op_callable.get_rna_type().properties}
    except Exception:
        return set()


def _tag_redraw_screen(screen):
    if screen is None:
        return
    for area in getattr(screen, "areas", ()):
        try:
            area.tag_redraw()
        except Exception:
            pass


def apply_camera_views(screen_name):
    """Apply per-viewport camera overrides after Blender finishes area splitting."""
    screen = bpy.data.screens.get(screen_name)
    if screen is None:
        return None
    areas = [a for a in screen.areas if a.type == 'VIEW_3D']
    if len(areas) != 4:
        return 0.1
    rows = sorted(areas, key=lambda a: -a.y)
    ordered = sorted(rows[:2], key=lambda a: a.x) + sorted(rows[2:], key=lambda a: a.x)
    for area, (label, direction, up) in zip(ordered, CAMERAS):
        camera = bpy.data.objects.get(label)
        if camera is None or camera.type != 'CAMERA':
            return None
        space = area.spaces.active
        if space is None or space.type != 'VIEW_3D':
            space = next((s for s in area.spaces if s.type == 'VIEW_3D'), None)
        if space is None:
            continue
        space.use_local_camera = True
        space.camera = camera
        if space.region_3d is not None:
            space.region_3d.view_rotation = camera_rotation_euler(direction, up).to_quaternion()
            space.region_3d.view_perspective = 'CAMERA'
            space.region_3d.view_camera_zoom = 0
            space.region_3d.view_camera_offset = (0.0, 0.0)
        area.tag_redraw()
    return None


class FOURVIEW_Settings(bpy.types.PropertyGroup):
    resolution_percent: bpy.props.IntProperty(
        name="分辨率比例", default=100, min=1, max=100
    )
    output_folder: bpy.props.StringProperty(
        name="输出目录", subtype='DIR_PATH', default='//four_view_renders/'
    )
    auto_update_cameras: bpy.props.BoolProperty(
        name="渲染前自动更新摄像机",
        default=True,
        description="渲染前根据当前场景物体自动更新四台摄像机的位置、正交缩放与朝向",
    )


class FOURVIEW_OT_cameras(bpy.types.Operator):
    bl_idname = "fourview.create_cameras"
    bl_label = "创建 / 更新四视角摄像机"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        center, scale = bounds(context)
        try:
            for label, direction, up in CAMERAS:
                get_camera(context, label, direction, up, center, scale)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        context.view_layer.update()
        if context.screen is not None:
            apply_camera_views(context.screen.name)
        self.report({'INFO'}, "四台摄像机已创建或更新")
        return {'FINISHED'}


class FOURVIEW_OT_layout(bpy.types.Operator):
    bl_idname = "fourview.setup_layout"
    bl_label = "切换到四视图布局"
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def split_area(context, screen, area, direction):
        window = _find_window_for_screen(screen)
        window_region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if window_region is None or window is None:
            raise RuntimeError("目标视口没有可用的 WINDOW 区域")
        result = _call_op_with_override(
            bpy.ops.screen.area_split,
            context=context,
            window=window,
            screen=screen,
            area=area,
            region=window_region,
            direction=direction,
            factor=0.5,
        )
        if 'FINISHED' not in result:
            raise RuntimeError("area_split 未完成：" + str(result))
        _tag_redraw_screen(screen)

    @staticmethod
    def _join_pair(context, window, screen, source, target, cursor):
        """Join or close `source` into `target` across Blender 3.1 ~ 5.2.3+."""
        region = next((r for r in source.regions if r.type == 'WINDOW'), None)
        before_count = len([a for a in screen.areas if a.type == 'VIEW_3D'])
        join_props = _op_rna_prop_names(bpy.ops.screen.area_join)

        src_center = (int(source.x + source.width // 2), int(source.y + source.height // 2))
        tgt_center = (int(target.x + target.width // 2), int(target.y + target.height // 2))

        # 1. Blender 4.3 ~ 5.2.3+: area_join(source_xy=..., target_xy=...)
        if "source_xy" in join_props and "target_xy" in join_props:
            for s_xy, t_xy in ((tgt_center, src_center), (src_center, tgt_center)):
                try:
                    res = _call_op_with_override(
                        bpy.ops.screen.area_join,
                        context=context,
                        window=window,
                        screen=screen,
                        area=source,
                        region=region,
                        source_xy=s_xy,
                        target_xy=t_xy,
                    )
                    if 'FINISHED' in res and len([a for a in screen.areas if a.type == 'VIEW_3D']) < before_count:
                        _tag_redraw_screen(screen)
                        return True
                except Exception:
                    pass

        # 2. Blender 3.1 ~ 4.2: area_join(cursor=...)
        elif "cursor" in join_props or not join_props:
            try:
                res = _call_op_with_override(
                    bpy.ops.screen.area_join,
                    context=context,
                    window=window,
                    screen=screen,
                    area=source,
                    region=region,
                    cursor=cursor,
                )
                if 'FINISHED' in res and len([a for a in screen.areas if a.type == 'VIEW_3D']) < before_count:
                    _tag_redraw_screen(screen)
                    return True
            except Exception:
                pass

        # 3. Older signature fallback: area_join(min_x=..., min_y=..., max_x=..., max_y=...)
        elif "min_x" in join_props:
            try:
                res = _call_op_with_override(
                    bpy.ops.screen.area_join,
                    context=context,
                    window=window,
                    screen=screen,
                    area=source,
                    region=region,
                    min_x=tgt_center[0],
                    min_y=tgt_center[1],
                    max_x=src_center[0],
                    max_y=src_center[1],
                )
                if 'FINISHED' in res and len([a for a in screen.areas if a.type == 'VIEW_3D']) < before_count:
                    _tag_redraw_screen(screen)
                    return True
            except Exception:
                pass

        # 4. Universal fallback for Blender 3.0 ~ 5.2.3+: bpy.ops.screen.area_close()
        if hasattr(bpy.ops.screen, "area_close"):
            try:
                res = _call_op_with_override(
                    bpy.ops.screen.area_close,
                    context=context,
                    window=window,
                    screen=screen,
                    area=source,
                    region=region,
                )
                if 'FINISHED' in res:
                    _tag_redraw_screen(screen)
                    return True
            except Exception:
                pass

        return False

    @staticmethod
    def join_to_one(context, screen):
        window = _find_window_for_screen(screen)
        if window is None:
            raise RuntimeError("找不到所属窗口")
        for _ in range(12):
            areas = [a for a in screen.areas if a.type == 'VIEW_3D']
            if len(areas) <= 1:
                return
            pairs = []
            for i, a in enumerate(areas):
                for b in areas[i + 1:]:
                    overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                    overlap_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                    if abs(a.x + a.width - b.x) <= 4 and overlap_y > 8:
                        cursor = (int(a.x + a.width), int(max(a.y, b.y) + overlap_y / 2))
                        aligned = abs(a.y - b.y) <= 4 and abs(a.height - b.height) <= 4
                    elif abs(b.x + b.width - a.x) <= 4 and overlap_y > 8:
                        cursor = (int(b.x + b.width), int(max(a.y, b.y) + overlap_y / 2))
                        aligned = abs(a.y - b.y) <= 4 and abs(a.height - b.height) <= 4
                    elif abs(a.y + a.height - b.y) <= 4 and overlap_x > 8:
                        cursor = (int(max(a.x, b.x) + overlap_x / 2), int(a.y + a.height))
                        aligned = abs(a.x - b.x) <= 4 and abs(a.width - b.width) <= 4
                    elif abs(b.y + b.height - a.y) <= 4 and overlap_x > 8:
                        cursor = (int(max(a.x, b.x) + overlap_x / 2), int(b.y + b.height))
                        aligned = abs(a.x - b.x) <= 4 and abs(a.width - b.width) <= 4
                    else:
                        continue
                    source, target = (a, b) if a.width * a.height <= b.width * b.height else (b, a)
                    pairs.append((0 if aligned else 1, source.width * source.height, source, target, cursor))

            joined = False
            if pairs:
                pairs.sort(key=lambda item: (item[0], item[1]))
                for _align_rank, _area_size, source, target, cursor in pairs:
                    if FOURVIEW_OT_layout._join_pair(context, window, screen, source, target, cursor):
                        joined = True
                        break

            if not joined and hasattr(bpy.ops.screen, "area_close"):
                smallest = min(areas, key=lambda a: a.width * a.height)
                region = next((r for r in smallest.regions if r.type == 'WINDOW'), None)
                try:
                    res = _call_op_with_override(
                        bpy.ops.screen.area_close,
                        context=context,
                        window=window,
                        screen=screen,
                        area=smallest,
                        region=region,
                    )
                    if 'FINISHED' in res:
                        _tag_redraw_screen(screen)
                        joined = True
                except Exception:
                    pass

            if not joined:
                raise RuntimeError("找不到可合并的相邻 3D 视口")
        raise RuntimeError("合并旧视口超过安全步数")

    @staticmethod
    def is_quad(areas):
        if len(areas) != 4:
            return False
        for coords in ([a.x + a.width / 2 for a in areas],
                       [a.y + a.height / 2 for a in areas]):
            values = sorted(coords)
            gap0 = values[1] - values[0]
            gap1 = values[2] - values[1]
            gap2 = values[3] - values[2]
            if gap1 < 20 or gap1 < max(gap0, gap2):
                return False
        return True

    def execute(self, context):
        screen = context.screen
        if screen is None:
            self.report({'ERROR'}, "当前上下文没有可用的屏幕布局")
            return {'CANCELLED'}
        missing = [
            label for label, _, _ in CAMERAS
            if bpy.data.objects.get(label) is None or bpy.data.objects[label].type != 'CAMERA'
        ]
        if missing:
            self.report({'ERROR'}, "缺少摄像机：" + "、".join(missing) + "；请先创建摄像机")
            return {'CANCELLED'}
        bpy.app.timers.register(
            lambda name=screen.name: _layout_progress(name), first_interval=0.05
        )
        self.report({'INFO'}, "正在创建均分十字四视图并绑定对应摄像机")
        return {'FINISHED'}


def _layout_progress(screen_name, attempts=0):
    def schedule_next():
        bpy.app.timers.register(
            lambda name=screen_name, count=attempts + 1: _layout_progress(name, count),
            first_interval=0.08,
        )
        return None

    screen = bpy.data.screens.get(screen_name)
    if screen is None or attempts >= 24:
        return None
    window = _find_window_for_screen(screen)
    if window is None:
        return None
    areas = [a for a in screen.areas if a.type == 'VIEW_3D']
    if not areas:
        return None

    # Wait one tick if Blender has not yet calculated geometry for a newly split area
    if len(areas) >= 2 and any(a.width <= 1 or a.height <= 1 for a in areas):
        _tag_redraw_screen(screen)
        return schedule_next()

    if FOURVIEW_OT_layout.is_quad(areas):
        apply_camera_views(screen_name)
        return None

    if len(areas) > 4:
        try:
            FOURVIEW_OT_layout.join_to_one(bpy.context, screen)
        except Exception:
            return None
        return schedule_next()

    if len(areas) == 1:
        target, direction = areas[0], 'VERTICAL'
    elif len(areas) == 2:
        # Check if the 2 viewports are side-by-side (left/right) or stacked (top/bottom)
        dx = abs(areas[0].x - areas[1].x)
        dy = abs(areas[0].y - areas[1].y)
        if dx >= dy:
            target = max(areas, key=lambda area: (area.width, -area.x))
            direction = 'HORIZONTAL'
        else:
            target = max(areas, key=lambda area: (area.height, -area.y))
            direction = 'VERTICAL'
    elif len(areas) == 3:
        min_h = min(a.height for a in areas)
        max_h_area = max(areas, key=lambda a: a.height)
        min_w = min(a.width for a in areas)
        max_w_area = max(areas, key=lambda a: a.width)
        if max_h_area.height > min_h * 1.4:
            target, direction = max_h_area, 'HORIZONTAL'
        elif max_w_area.width > min_w * 1.4:
            target, direction = max_w_area, 'VERTICAL'
        else:
            try:
                FOURVIEW_OT_layout.join_to_one(bpy.context, screen)
            except Exception:
                return None
            return schedule_next()
    else:
        try:
            FOURVIEW_OT_layout.join_to_one(bpy.context, screen)
        except Exception:
            return None
        return schedule_next()

    try:
        FOURVIEW_OT_layout.split_area(bpy.context, screen, target, direction)
    except Exception:
        return None
    return schedule_next()


def _resolve_output_folder(raw_path):
    """Resolve Blender path (including '//' on unsaved .blend files) to a writable directory."""
    folder = bpy.path.abspath(raw_path)
    if raw_path.startswith("//") and not bpy.data.filepath:
        rel_subpath = raw_path[2:].lstrip("/\\") or "four_view_renders"
        base_temp = getattr(bpy.app, "tempdir", None) or tempfile.gettempdir()
        folder = os.path.join(base_temp, rel_subpath)
    folder = os.path.abspath(folder)
    try:
        os.makedirs(folder, exist_ok=True)
        return folder
    except OSError:
        if raw_path.startswith("//"):
            rel_subpath = raw_path[2:].lstrip("/\\") or "four_view_renders"
            base_temp = getattr(bpy.app, "tempdir", None) or tempfile.gettempdir()
            fallback = os.path.abspath(os.path.join(base_temp, rel_subpath))
            os.makedirs(fallback, exist_ok=True)
            return fallback
        raise


class FOURVIEW_OT_render(bpy.types.Operator):
    bl_idname = "fourview.render_all"
    bl_label = "渲染四个视角"

    def execute(self, context):
        scene = context.scene
        settings = scene.fourview_settings

        if settings.auto_update_cameras:
            center, scale = bounds(context)
            try:
                for label, direction, up in CAMERAS:
                    get_camera(context, label, direction, up, center, scale)
            except Exception as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

        cameras = []
        for label, _, _ in CAMERAS:
            obj = scene.objects.get(label)
            if obj is None or obj.type != 'CAMERA':
                self.report({'ERROR'}, "缺少摄像机：" + label + "。请先创建 / 更新四视角摄像机。")
                return {'CANCELLED'}
            cameras.append((label, obj))

        try:
            folder = _resolve_output_folder(settings.output_folder)
        except Exception as exc:
            self.report({'ERROR'}, "无法创建输出目录：" + str(exc))
            return {'CANCELLED'}

        old_camera = scene.camera
        old_path = scene.render.filepath
        old_percent = scene.render.resolution_percentage
        old_use_sequencer = scene.render.use_sequencer

        # Save and temporarily clear timeline marker cameras so they cannot
        # override scene.camera during RE_RenderFrame initialization.
        saved_markers = [(m, m.camera) for m in scene.timeline_markers if m.camera is not None]

        # Save and temporarily disable viewport local camera overrides across
        # 3D viewports so the active sidebar viewport never hijacks rendering.
        saved_spaces = []
        for scr in bpy.data.screens:
            for area in scr.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            saved_spaces.append((space, space.use_local_camera, space.camera))

        render_props = _op_rna_prop_names(bpy.ops.render.render)

        try:
            scene.render.resolution_percentage = settings.resolution_percent
            scene.render.use_sequencer = False
            for m, _cam in saved_markers:
                m.camera = None
            for space, _use_local, _cam in saved_spaces:
                space.use_local_camera = False

            for label, camera in cameras:
                camera.hide_render = False
                scene.camera = camera
                context.view_layer.update()
                depsgraph = context.evaluated_depsgraph_get()
                depsgraph.update()
                scene.render.filepath = os.path.join(folder, label)

                render_kwargs = {"write_still": True}
                if not render_props or "use_viewport" in render_props:
                    render_kwargs["use_viewport"] = False
                if not render_props or "scene" in render_props:
                    render_kwargs["scene"] = scene.name
                if not render_props or "layer" in render_props:
                    render_kwargs["layer"] = context.view_layer.name

                bpy.ops.render.render(**render_kwargs)
        except Exception as exc:
            self.report({'ERROR'}, "渲染失败：" + str(exc))
            return {'CANCELLED'}
        finally:
            scene.camera = old_camera
            scene.render.filepath = old_path
            scene.render.resolution_percentage = old_percent
            scene.render.use_sequencer = old_use_sequencer
            for m, cam in saved_markers:
                m.camera = cam
            for space, use_local, cam in saved_spaces:
                space.use_local_camera = use_local
                space.camera = cam
            if context.screen is not None:
                apply_camera_views(context.screen.name)

        self.report({'INFO'}, "四个视角已渲染到：" + folder)
        return {'FINISHED'}


class FOURVIEW_PT_panel(bpy.types.Panel):
    bl_label = "四视图渲染"
    bl_idname = "FOURVIEW_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "四视图"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.fourview_settings
        layout.operator("fourview.create_cameras", icon='CAMERA_DATA')
        layout.operator("fourview.setup_layout", icon='WINDOW')
        layout.operator("fourview.render_all", icon='RENDER_STILL')
        layout.prop(settings, "auto_update_cameras")
        layout.prop(settings, "resolution_percent")
        layout.prop(settings, "output_folder")
        layout.label(text="左上正视，右上侧视，左下顶视，右下底视")


CLASSES = (
    FOURVIEW_Settings,
    FOURVIEW_OT_cameras,
    FOURVIEW_OT_layout,
    FOURVIEW_OT_render,
    FOURVIEW_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fourview_settings = bpy.props.PointerProperty(type=FOURVIEW_Settings)


def unregister():
    if hasattr(bpy.types.Scene, 'fourview_settings'):
        del bpy.types.Scene.fourview_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()