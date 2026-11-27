"""Builds a Blender scene from the scene.json beside this script (RustXMALab, File > Export >
Blender scene; parity ledger F85). The same script is written into every export; only
scene.json differs.

Three ways to use it (Blender 4.2 or later):

  1. In Blender: Scripting tab, Open this file, Run Script. The scene is added to the file
     that is open, in a new collection named after the export. Running it again in the same
     file replaces what an earlier run of this export made (every collection, object, camera,
     mesh, material and action it made carries the custom property "xma_scene" with the
     export's name; only those are removed, and an object of your own that you put into one
     of its collections is moved to the scene's top collection, not deleted).
  2. From a shell: blender --python build_scene.py   (opens Blender with the scene built)
  3. Without a window, to save a .blend beside this script:
       blender --background --factory-startup --python build_scene.py -- --save NAME.blend [--pack]
     The file is saved with paths relative to it. --pack packs still images into the .blend
     (Blender cannot pack an image sequence: a sequence stays a set of files beside it).

Other arguments after "--": --json PATH reads another scene.json.

scene.json, formats 1 and 2
---------------------------
Format 1 has no bodies and points (empty lists); format 2 has them and "point_size". A
script older than format 2 refuses a format 2 file.
{
  "format": 1 | 2,                   this script refuses another number
  "generator": "RustXMALab x.y.z",
  "name": "...",                     the collection's name
  "kind": "cameras" | "trial",
  "project": "file.xma",
  "calibration_image": n | null,     1-based, the calibration image whose poses are used
  "trial": null | {"number": n, "name": "...", "first_frame": a, "last_frame": b},
  "scene": {"frame_start": a, "frame_end": b, "fps": f},
                                     fps 0: the scene's frame rate is left as it is
  "cameras": [{
      "name": "Camera1",             object names: Camera1, Camera1_plane, Camera1_material
      "label": "...",                the camera's name in the project
      "number": 1,
      "width": w, "height": h,       the image, pixels
      "fx": .., "fy": .., "cx": .., "cy": ..,
                                     the camera matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
                                     of the undistorted image; pixel centres at integers,
                                     origin top-left, y down
      "rotation": [[3],[3],[3]],     R, world to camera (OpenCV axes: x right, y down,
      "translation": [3],            z forward): x_camera = R X + t
      "flipped": bool,               the camera's images are mirrored on load
      "undistorted": bool,           the images are undistorted (grid or lens model)
      "plane_distance": d,           the image plane's distance from the camera
      "image": null | {
          "path": "images/camera1/...",   relative to scene.json
          "sequence": null | {"first": a, "count": n},
                                     a numbered sequence: files for frames a..a+n-1
                                     (1-based, the number in each file's name)
          "mirrored": bool           the files are mirrored left to right relative to the
                                     image the calibration refers to
      }
  }],
  "point_size": s,                   format 2: the points' display size, scene units
  "bodies": [{                       rigid bodies (format 2)
      "name": "...",                 the Empty's name; its mesh is "<name>_mesh"
      "number": n,                   1-based, the body's number in the trial
      "filtered": bool,
      "first_frame": a,              the scene frame of matrices[0]
      "matrices": [null | [[4],[4],[4],[4]], ...],
                                     per frame, body (reference, CT) coordinates to world,
                                     rows: [R^T | -R^T t] of the stored pose (the matrix of
                                     File > Export > Rigid body transformations); null:
                                     no pose (the body is hidden at that frame)
      "mesh": null | {"path": "meshes/....obj", "scale": s, "source": "..."}
                                     an OBJ file imported without axis conversion, a
                                     child of the Empty with scale s: vertex v is at
                                     M (s v) in the world
  }],
  "points": [{                       3D points of markers (format 2)
      "name": "...", "number": n, "filtered": bool, "first_frame": a,
      "positions": [null | [x, y, z], ...]    per frame; null: hidden at that frame
  }]
}
Bodies go into the sub-collection "Rigid bodies", points into "Points". A body is keyed
(location, rotation_euler in XYZ order with each frame's angles chosen close to the frame
before, scale) at each frame with a pose; hide_viewport and hide_render are keyed where
it has none, so it is not seen gliding through frames without a pose.

Units are the calibration object's, unchanged.

The camera (derived from K, w and h; checked against the program's own projection in
crates/xma-engine/tests/blender_scene.rs):
  matrix_world: rotation R^T diag(1,-1,-1), location -R^T t   (Blender looks down -Z, +Y up)
  sensor_fit HORIZONTAL, lens / sensor_width = fx / w
  pixel aspect y / x = fx / fy
  shift_x = ((w - 1) / 2 - cx) / w
  shift_y = (cy - (h - 1) / 2) / fy * fx / w
  render resolution w x h. A scene has one resolution: it is set from camera 1. Each camera
  object holds its own as custom properties (xma_width, xma_height, xma_pixel_aspect_x,
  xma_pixel_aspect_y); set the scene's to them before looking through another camera
  (use_camera("Camera2") in this script does it).
The image plane: a quad whose corners are the image's corners (pixel edges -0.5 and w-0.5,
-0.5 and h-0.5) seen at depth 1, parented to the camera and scaled by its custom property
"distance" (drivers on its scale): change "distance" to slide it along the view; it keeps
filling the frame.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

FORMAT = 1           # scene.json without bodies and points
FORMAT_ANIMATED = 2  # with bodies or points; scripts written before it refuse it
FORMATS = (FORMAT, FORMAT_ANIMATED)


# The custom property that marks what this script made, set to the export's name.
MARK = "xma_scene"
_export = ["RustXMALab"]


def mark(*ids):
    """Marks data-blocks as made by this run of the export."""
    for i in ids:
        if i is not None:
            i[MARK] = _export[0]


def remove_previous(name, scene):
    """Removes what an earlier run of the export `name` made in this file (the data-blocks
    marked with its name), so that a second run does not make "Camera1.001" or
    "Rigid bodies.001". Objects of the user's own in its collections are moved to the scene's
    top collection. Returns how many objects were removed."""
    def ours(i):
        try:
            return i.get(MARK) == name
        except TypeError:
            return False
    colls = [c for c in bpy.data.collections if ours(c)]
    for c in colls:
        for o in list(c.objects):
            if not ours(o) and all(u in colls for u in o.users_collection):
                scene.collection.objects.link(o)
        for ch in list(c.children):
            if not ours(ch) and ch.name not in scene.collection.children:
                scene.collection.children.link(ch)
    objects = [o for o in bpy.data.objects if ours(o)]
    actions = []
    for o in objects:
        ad = o.animation_data
        if ad is not None and ad.action is not None and ours(ad.action):
            actions.append(ad.action)
    n = len(objects)
    bpy.data.batch_remove(objects + colls)
    # Data the removed objects used, now without users (an action, camera, mesh or
    # material of the export that something else still uses is kept).
    data = [a for a in bpy.data.actions if ours(a)] + \
        [d for d in bpy.data.cameras if ours(d)] + \
        [d for d in bpy.data.meshes if ours(d)]
    bpy.data.batch_remove([d for d in data if d.users == 0])
    bpy.data.batch_remove([m for m in bpy.data.materials if ours(m) and m.users == 0])
    return n


def log(msg):
    print("build_scene: " + msg)
    sys.stdout.flush()


def script_args():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def option(args, name):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return None


def find_json(args):
    given = option(args, "--json")
    if given:
        return os.path.abspath(given)
    candidates = []
    f = globals().get("__file__")
    if f:
        candidates.append(f)
    for t in bpy.data.texts:
        if t.filepath:
            candidates.append(bpy.path.abspath(t.filepath))
    candidates.append(os.getcwd() + os.sep + "x")
    for c in candidates:
        p = os.path.join(os.path.dirname(os.path.abspath(c)), "scene.json")
        if os.path.isfile(p):
            return p
    raise RuntimeError(
        "scene.json not found beside build_scene.py; open the script from the exported "
        "folder or give -- --json PATH")


def lens_settings(w, h, fx, fy, cx, cy):
    """Blender camera settings that make it see what the X-ray camera saw."""
    sensor = 36.0
    lens = fx * sensor / w
    if lens < 1.0:
        sensor = 36.0 / lens
        lens = fx * sensor / w
    if fx >= fy:
        ax, ay = 1.0, fx / fy
    else:
        ax, ay = fy / fx, 1.0
    return {
        "sensor_width": sensor,
        "lens": lens,
        "shift_x": ((w - 1) / 2.0 - cx) / w,
        "shift_y": (cy - (h - 1) / 2.0) / fy * fx / w,
        "pixel_aspect_x": ax,
        "pixel_aspect_y": ay,
    }


def use_camera(name, scene=None):
    """Makes `name` the scene's camera and sets the render resolution and pixel aspect to
    that camera's own."""
    scene = scene or bpy.context.scene
    ob = bpy.data.objects[name]
    scene.camera = ob
    r = scene.render
    r.resolution_x = int(ob["xma_width"])
    r.resolution_y = int(ob["xma_height"])
    r.resolution_percentage = 100
    r.pixel_aspect_x = float(ob["xma_pixel_aspect_x"])
    r.pixel_aspect_y = float(ob["xma_pixel_aspect_y"])


def add_scale_drivers(plane):
    for axis in range(3):
        fc = plane.driver_add("scale", axis)
        drv = fc.driver
        drv.type = "AVERAGE"
        var = drv.variables.new()
        var.name = "distance"
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "OBJECT"
        var.targets[0].id = plane
        var.targets[0].data_path = '["distance"]'


def load_image(base, img):
    path = os.path.join(base, img["path"])
    if not os.path.isfile(path):
        log("image not found, plane left empty: " + path)
        return None
    image = bpy.data.images.load(path, check_existing=True)
    if img.get("sequence"):
        image.source = "SEQUENCE"
    return image


def make_material(name, image, seq):
    mat = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True  # always on from Blender 5.0, where setting it is deprecated
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (150, 0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = "xma_image"
        tex.location = (-200, 0)
        tex.image = image
        tex.interpolation = "Linear"
        tex.extension = "CLIP"
        if seq:
            iu = tex.image_user
            iu.frame_duration = int(seq["count"])
            iu.frame_start = int(seq["first"])
            iu.frame_offset = int(seq["first"]) - 1
            iu.use_cyclic = False
            iu.use_auto_refresh = True
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    return mat


def make_camera(coll, base, c):
    w, h = int(c["width"]), int(c["height"])
    fx, fy, cx, cy = (float(c[k]) for k in ("fx", "fy", "cx", "cy"))
    name = c["name"]
    s = lens_settings(w, h, fx, fy, cx, cy)
    cd = bpy.data.cameras.new(name)
    cd.type = "PERSP"
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = s["sensor_width"]
    cd.lens = s["lens"]
    cd.shift_x = s["shift_x"]
    cd.shift_y = s["shift_y"]
    d = float(c["plane_distance"])
    cd.clip_start = max(d * 1e-3, 1e-6)
    cd.clip_end = max(d * 4.0, 1.0)
    ob = bpy.data.objects.new(name, cd)
    mark(cd, ob)
    coll.objects.link(ob)
    R = Matrix([list(map(float, row)) for row in c["rotation"]])
    t = Vector([float(v) for v in c["translation"]])
    Rt = R.transposed()
    M = (Rt @ Matrix.Diagonal((1.0, -1.0, -1.0))).to_4x4()
    M.translation = -(Rt @ t)
    ob.matrix_world = M
    ob["xma_label"] = c.get("label", "")
    ob["xma_number"] = int(c.get("number", 0))
    ob["xma_width"] = w
    ob["xma_height"] = h
    ob["xma_fx"] = fx
    ob["xma_fy"] = fy
    ob["xma_cx"] = cx
    ob["xma_cy"] = cy
    ob["xma_pixel_aspect_x"] = s["pixel_aspect_x"]
    ob["xma_pixel_aspect_y"] = s["pixel_aspect_y"]
    ob["xma_flipped"] = bool(c.get("flipped", False))
    ob["xma_undistorted"] = bool(c.get("undistorted", False))

    # The plane: the image's corners at depth 1 in the camera's frame (Blender axes).
    corners = [(-0.5, -0.5), (-0.5, h - 0.5), (w - 0.5, h - 0.5), (w - 0.5, -0.5)]
    verts = [((u - cx) / fx, -(v - cy) / fy, -1.0) for (u, v) in corners]
    me = bpy.data.meshes.new(name + "_plane")
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    me.update()
    img = c.get("image")
    mirrored = bool(img and img.get("mirrored"))
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        u, v = corners[loop.vertex_index]
        uu = (u + 0.5) / w
        if mirrored:
            uu = 1.0 - uu
        uv.data[loop.index].uv = (uu, 1.0 - (v + 0.5) / h)
    plane = bpy.data.objects.new(name + "_plane", me)
    mark(me, plane)
    coll.objects.link(plane)
    plane.parent = ob
    plane.matrix_parent_inverse = Matrix.Identity(4)
    plane.location = (0.0, 0.0, 0.0)
    plane["distance"] = d
    plane["SID"] = int(c.get("number", 0))
    add_scale_drivers(plane)
    image = load_image(base, img) if img else None
    mat = make_material(name + "_material", image, img.get("sequence") if img else None)
    mark(mat)
    me.materials.append(mat)
    return ob, image


def key_hidden(ob, frames_hidden):
    """Keys hide_viewport and hide_render where the hidden state changes."""
    last = None
    for f, hidden in frames_hidden:
        if hidden == last:
            continue
        ob.hide_viewport = hidden
        ob.hide_render = hidden
        ob.keyframe_insert("hide_viewport", frame=f)
        ob.keyframe_insert("hide_render", frame=f)
        last = hidden
    ob.hide_viewport = False
    ob.hide_render = False


def fcurves_of(ob):
    """The object's F-curves (Blender 4.4 and later keep them in a slot's channel bag)."""
    ad = ob.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    try:
        from bpy_extras import anim_utils
        bag = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
        if bag is not None:
            return list(bag.fcurves)
    except (ImportError, AttributeError):
        pass
    return list(getattr(act, "fcurves", []))


def mark_action(ob):
    ad = ob.animation_data
    if ad is not None and ad.action is not None:
        mark(ad.action)


def set_interpolation(ob):
    """Linear between pose keys (no overshoot between measured frames), constant for the
    hidden state."""
    for fc in fcurves_of(ob):
        constant = fc.data_path.startswith("hide")
        for k in fc.keyframe_points:
            k.interpolation = "CONSTANT" if constant else "LINEAR"


def import_obj(path):
    """The objects Blender's OBJ importer makes from `path`, its vertices as in the file."""
    before = set(bpy.data.objects)
    mats = set(bpy.data.materials)
    kw = dict(filepath=path, forward_axis="Y", up_axis="Z", global_scale=1.0,
              use_split_objects=False, use_split_groups=False)
    try:
        bpy.ops.wm.obj_import(validate_meshes=False, **kw)
    except TypeError:
        bpy.ops.wm.obj_import(**kw)
    mark(*[m for m in bpy.data.materials if m not in mats])
    return [o for o in bpy.data.objects if o not in before]


def make_body(coll, base, b, size):
    ob = bpy.data.objects.new(b["name"], None)
    mark(ob)
    coll.objects.link(ob)
    ob.empty_display_type = "ARROWS"
    ob.empty_display_size = size * 5.0
    ob.rotation_mode = "XYZ"
    ob["xma_body"] = int(b.get("number", 0))
    ob["xma_filtered"] = bool(b.get("filtered", False))
    first = int(b["first_frame"])
    prev = None
    hidden = []
    for k, m in enumerate(b.get("matrices") or []):
        f = first + k
        hidden.append((f, m is None))
        if m is None:
            continue
        M = Matrix([list(map(float, row)) for row in m])
        loc, rot, scale = M.decompose()
        eul = rot.to_matrix().to_euler("XYZ", prev) if prev is not None else \
            rot.to_matrix().to_euler("XYZ")
        prev = eul
        ob.location = loc
        ob.rotation_euler = eul
        ob.scale = scale
        for path in ("location", "rotation_euler", "scale"):
            ob.keyframe_insert(path, frame=f)
    key_hidden(ob, hidden)
    set_interpolation(ob)
    mark_action(ob)
    mesh = b.get("mesh")
    if mesh:
        path = os.path.join(base, mesh["path"])
        if not os.path.isfile(path):
            log("mesh not found, left out: " + path)
        else:
            made = import_obj(path)
            meshes = [o for o in made if o.type == "MESH"]
            for o in made:
                if o not in meshes:
                    bpy.data.objects.remove(o)
            if not meshes:
                log("the OBJ importer made no mesh from " + path)
            else:
                me = meshes[0]
                for extra in meshes[1:]:
                    log("more than one mesh in %s: only the first is kept" % path)
                    bpy.data.objects.remove(extra)
                for c in list(me.users_collection):
                    c.objects.unlink(me)
                coll.objects.link(me)
                mark(me, me.data)
                me.name = b["name"] + "_mesh"
                me.data.name = b["name"] + "_mesh"
                me.parent = ob
                me.matrix_parent_inverse = Matrix.Identity(4)
                me.location = (0.0, 0.0, 0.0)
                me.rotation_mode = "XYZ"
                me.rotation_euler = (0.0, 0.0, 0.0)
                s = float(mesh.get("scale", 1.0))
                me.scale = (s, s, s)
                me["xma_body"] = int(b.get("number", 0))
                me["xma_mesh_source"] = mesh.get("source", "")
                me.data["xma_mesh_path"] = mesh["path"]
    return ob


def make_point(coll, p, size):
    ob = bpy.data.objects.new(p["name"], None)
    mark(ob)
    coll.objects.link(ob)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = size
    ob["xma_marker"] = int(p.get("number", 0))
    ob["xma_filtered"] = bool(p.get("filtered", False))
    first = int(p["first_frame"])
    hidden = []
    for k, xyz in enumerate(p.get("positions") or []):
        f = first + k
        hidden.append((f, xyz is None))
        if xyz is None:
            continue
        ob.location = [float(v) for v in xyz]
        ob.keyframe_insert("location", frame=f)
    key_hidden(ob, hidden)
    set_interpolation(ob)
    mark_action(ob)
    return ob


def build(data, base):
    if data.get("format") not in FORMATS:
        raise RuntimeError("scene.json has format %r; this script reads formats %s"
                           % (data.get("format"), ", ".join(str(f) for f in FORMATS)))
    scene = bpy.context.scene
    name = data.get("name") or "RustXMALab"
    _export[0] = name
    removed = remove_previous(name, scene)
    if removed:
        log("replaced what an earlier run made in this file (%d objects removed)" % removed)
    coll = bpy.data.collections.new(name)
    mark(coll)
    scene.collection.children.link(coll)
    cams = []
    images = []
    for c in data.get("cameras", []):
        ob, image = make_camera(coll, base, c)
        cams.append(ob)
        if image is not None:
            images.append((image, bool(c["image"].get("sequence"))))
    sc = data.get("scene") or {}
    if "frame_start" in sc and "frame_end" in sc:
        scene.frame_start = int(sc["frame_start"])
        scene.frame_end = int(sc["frame_end"])
        scene.frame_set(int(sc["frame_start"]))
    fps = sc.get("fps")
    if fps and fps > 0:
        n = max(1, int(round(fps)))
        scene.render.fps = n
        scene.render.fps_base = n / float(fps)
    if cams:
        use_camera(cams[0].name, scene)
    size = float(data.get("point_size") or 1.0)
    bodies = data.get("bodies") or []
    points = data.get("points") or []
    if bodies:
        sub = bpy.data.collections.new("Rigid bodies")
        mark(sub)
        coll.children.link(sub)
        for b in bodies:
            make_body(sub, base, b, size)
    if points:
        sub = bpy.data.collections.new("Points")
        mark(sub)
        coll.children.link(sub)
        for p in points:
            make_point(sub, p, size)
    if bodies or points:
        scene.frame_set(scene.frame_current)  # the keyed values of the current frame
    log("built %d cameras, %d rigid bodies and %d points in collection %s"
        % (len(cams), len(bodies), len(points), coll.name))
    return images


def main():
    args = script_args()
    path = find_json(args)
    base = os.path.dirname(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    save = option(args, "--save")
    if save:
        # An empty file (--factory-startup gives the default cube, camera and light).
        bpy.ops.wm.read_homefile(use_empty=True)
    images = build(data, base)
    if not save:
        return
    target = save if os.path.isabs(save) else os.path.join(base, save)
    target = os.path.abspath(target)
    blend_dir = os.path.dirname(target)
    for image, _ in images:
        image.filepath = "//" + os.path.relpath(bpy.path.abspath(image.filepath),
                                                blend_dir).replace(os.sep, "/")
    if "--pack" in args:
        for image, seq in images:
            if seq:
                log("not packed (Blender cannot pack an image sequence): " + image.name)
                continue
            try:
                image.pack()
                log("packed " + image.name)
            except RuntimeError as e:
                log("not packed: %s: %s" % (image.name, e))
    bpy.ops.wm.save_as_mainfile(filepath=target, relative_remap=True)
    log("saved " + target)


if __name__ == "__main__":
    main()
