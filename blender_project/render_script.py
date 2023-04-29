import os
import sys
from argparse import Namespace
from glob import glob

import bmesh
import bpy
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

"""
blender -b render_flame.blend --python script.py --render-anim -- \
    --output_prefix render_ \
    --image_size 512 \
    --source_dir ../test_out/test-000
"""


def _load_source(args: Namespace):
    mesh_files = None
    data_npy = None

    if os.path.isdir(args.source_path):
        mesh_files = utils.find_files(os.path.join(args.source_path), r"\d+\.npy", recursive=False, abspath=True)
        mesh_files = sorted(mesh_files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))
        num_max_frames = len(mesh_files)
        if num_max_frames == 0:
            raise FileNotFoundError("(!) Failed to find any frames")
        output_prefix = args.output_prefix if args.output_prefix is not None else os.path.dirname(mesh_files[0]) + "_"

    elif os.path.isfile(args.source_path):
        data_npy = np.load(args.source_path)
        num_max_frames = len(data_npy)
        output_prefix = (
            args.output_prefix
            if args.output_prefix is not None
            else os.path.splitext(args.source_path)[0] + "-blender-"
        )

    else:
        files = sorted(glob(args.source_path))
        assert len(files) > 0
        data_list = []
        for fpath in files:
            v = utils.read_ply(fpath)
            data_list.append(v)
        data_npy = np.asarray(data_list, np.float32)
        num_max_frames = len(data_npy)
        output_prefix = args.output_prefix

    return num_max_frames, mesh_files, data_npy, output_prefix


def _get_materials(args: Namespace):
    m = bpy.data.materials.get("material_avg")
    # assert not args.with_texture
    if not args.with_texture:
        node_tree = m.node_tree
        nodes = node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        node_tree.links.remove(bsdf.inputs['Base Color'].links[0])
        if args.base_color is not None:
            base_color = tuple(args.base_color)
            if len(base_color) == 3:
                base_color += (1,)
            bsdf.inputs["Base Color"].default_value = base_color
    return m


LMKS_VIDX = [
    1526, 916, 917, 908, 880, 826, 786, 773,  # 0...7
    1705, 121, 116, 94, 214, 237, 247, 246,  # 8...15
    659, 1271, 1273, 1275, 1078, 1113, 1484, 407, 414, 631, 629, 627, 35,  # 16...28
    495, 491, 492, 512, 510, 440, 441, 1460, 1103, 1102, 1172, 1174, 1154, 1153, 1157,  # 29...
    1049, 1048, 1198, 1193, 1194, 1236, 1237, 1450, 579, 578, 536, 535, 540, 380, 381,  # 58
]


if __name__ == "__main__":
    parser = utils.ArgumentParserForBlender()
    parser.add_argument("-S", "--source_path", type=str, required=True)
    parser.add_argument("-O", "--output_prefix", type=str, required=True)
    parser.add_argument("-s", "--stt_frame", type=int)
    parser.add_argument("-e", "--end_frame", type=int)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--format", type=str, default="mp4")
    parser.add_argument("--scale", type=float, default=10)
    parser.add_argument("--is_source_offsets", action="store_true")
    parser.add_argument("--idle_fpath", type=str)
    parser.add_argument("--image_w", type=int, default=688)
    parser.add_argument("--image_h", type=int, default=880)
    parser.add_argument("--render_samples", type=int, default=64)
    parser.add_argument("--camera", type=str, default="Camera")
    parser.add_argument("--template_path", type=str)
    parser.add_argument("--template_vidx", type=str)
    parser.add_argument("--smooth_shading", action="store_true")
    parser.add_argument("--with_texture", action="store_true")
    parser.add_argument("--base_color", nargs=4, type=float)
    parser.add_argument("--small_size", action="store_true")
    parser.add_argument("-Q", "--quiet", action="store_true")
    args = parser.parse_args()

    # * Parse source
    num_max_frames, mesh_files, data_npy, output_prefix = _load_source(args)
    assert mesh_files is None or data_npy is None  # only one source is valid
    # frames stt, end and count
    stt_frame = args.stt_frame if args.stt_frame is not None else 0
    end_frame = args.end_frame if args.end_frame is not None else num_max_frames
    num_frames = end_frame - stt_frame

    if args.is_source_offsets:
        assert args.idle_fpath is not None
        tmpl_verts = utils.read_obj(args.idle_fpath, 5023)
    else:
        tmpl_verts = None

    # * Settings
    scene = bpy.context.scene
    # > get camera
    camera_object = bpy.data.objects[args.camera]
    # > material
    material = _get_materials(args)

    # > get flame object
    vert_idx = None
    if args.template_path is None:
        flame_object = bpy.data.objects["TMPL"]
    else:
        # if args.template_vidx is None or not os.path.exists(args.template_vidx):
        #     raise ValueError("(!) --template_path is set, but --template_vidx is not given!")
        if args.template_vidx is not None:
            with open(args.template_vidx) as fp:
                line = " ".join(x.strip() for x in fp.readlines())
                vert_idx = [int(x) for x in line.split()]
        bpy.data.objects["TMPL"].hide_set(True)
        bpy.data.objects["TMPL"].hide_render = True
        bpy.ops.import_scene.obj(filepath=args.template_path, split_mode="OFF")
        new_name = os.path.splitext(os.path.basename(args.template_path))[0]
        if new_name == "TMPL":
            new_name += ".001"
        flame_object = bpy.data.objects[new_name]
        flame_object.scale = (10, 10, 10)
        flame_object.data.materials.clear()
        flame_object.data.materials.append(material)
        
    # > Scaling.
    flame_object.scale = (args.scale, args.scale, args.scale)

    # > Shading mode: smooth / flat
    for poly in flame_object.data.polygons:
        poly.use_smooth = args.smooth_shading

    # # subsurface division modifier
    # m = flame_object.modifiers.new('My SubDiv', 'SUBSURF')
    # m.levels = 1
    # m.render_levels = 2
    # m.quality = 2

    # > Set output and render
    if args.format == "mp4":
        print("(+) Render video: {}{:04d}-{:04d}.mp4".format(output_prefix, stt_frame + 1, end_frame))
    else:
        if output_prefix[-1] != "/":
            output_prefix += "/"
        print("(+) Render images: {}".format(os.path.dirname(output_prefix)))
    print(f"Will render {num_frames} frames.")
    sys.stdout.flush()

    A = min(args.image_h, args.image_w)
    utils.set_output_properties(scene, (args.image_w, args.image_h), output_prefix)
    utils.set_eevee_renderer(
        scene, camera_object,
        num_samples=args.render_samples,
        small_size=args.small_size,
        file_format=args.format
    )

    # * Create animation
    action = bpy.data.actions.new("MeshAnimation")
    flame_object.data.animation_data_create()
    flame_object.data.animation_data.action = action
    # > Create fcurves for animation
    fcurves_list = []
    for iv, v in enumerate(flame_object.data.vertices):
        fcurves = [action.fcurves.new(f"vertices[{v.index:d}].co", index=k) for k in range(3)]
        for fcurve in fcurves:
            fcurve.keyframe_points.add(count=num_frames)
        fcurves_list.extend(fcurves)
    utils.set_animation(scene, fps=args.fps, frame_start=stt_frame + 1, frame_end=end_frame)
    # > Set keyframes
    os.makedirs(output_prefix, exist_ok=True)
    num_verts = len(flame_object.data.vertices)
    co_list = np.zeros((num_verts * 3, 2 * num_frames), np.float32)
    for i in range(num_frames):
        # get verts.
        if mesh_files is not None:
            vdata = np.load(mesh_files[i + stt_frame])
        else:
            vdata = data_npy[i + stt_frame]
        if args.is_source_offsets:
            vdata = vdata + tmpl_verts
        verts = vdata[vert_idx]
        verts = np.squeeze(verts)
        # projection
        xy_list = []
        for vidx in LMKS_VIDX:
            v = utils.Vector(verts[vidx] * args.scale)
            p = utils.project_3d_point(camera_object, v)
            xx = (p.x * 0.5 + 0.5) * args.image_w
            yy = (-p.y * 0.5 + 0.5) * args.image_h
            xy_list.append((int(xx), int(yy)))
        with open(output_prefix + f"{i+1:04d}.txt", "w") as fp:
            for x, y in xy_list:
                fp.write(f"{x} {y}\n")

        # set animation keyframe
        vdata = verts.flatten()
        co_list[:, i * 2 + 0] = i + stt_frame + 1
        co_list[:, i * 2 + 1] = vdata
        print("(+) Read keyframe {}".format(i + 1), end="\r")
    # foreach set
    for fcu, vals in zip(fcurves_list, co_list):
        fcu.keyframe_points.foreach_set("co", vals)
