import argparse
import os
import re
import sys
import time
from typing import Any, Callable, List, Tuple

import bmesh
import bpy
import numpy as np


class timeit(object):
    def __init__(self, tag: str = "timeit"):
        self.tag = tag

    def __enter__(self):
        self.ts = time.time()

    def __exit__(self, *args: Any):
        self.te = time.time()
        print("<{}> cost {:.2f} ms".format(self.tag, (self.te - self.ts) * 1000))
        return False

    def __call__(self, method: Callable[..., Any]):
        def timed(*args: Any, **kw: Any):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print("<{}> cost {:.2f} ms".format(method.__name__, (te - ts) * 1000))
            return result

        return timed


def read_obj(filepath: str, num_verts: int):
    # load vertices
    with open(filepath) as f:
        lines = f.readlines()

    vi = 0
    vertices = np.zeros((num_verts, 3), np.float32)
    for line in lines:
        line = line.strip().split()
        if len(line) == 0:
            continue
        if line[0] == "v":
            vertices[vi, 0] = float(line[1])
            vertices[vi, 1] = float(line[2])
            vertices[vi, 2] = float(line[3])
            vi += 1
    return vertices


def read_ply(ply_path: str):
    import plyfile
    plydata = plyfile.PlyData.read(ply_path)
    verts = np.stack((plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]), axis=1)
    verts = verts.astype(np.float32)
    return verts
    # faces = np.stack(plydata["face"]["vertex_indices"], axis=0)
    # faces = faces.astype(np.int32)
    # return verts, faces


def find_files(directory: str, pattern: str, recursive: bool = True, abspath: bool = False) -> List[str]:
    regex = re.compile(pattern)
    file_list: List[str] = []
    for root, _, files in os.walk(directory):
        for f in files:
            if regex.match(f) is not None:
                file_list.append(os.path.join(root, f))
        if not recursive:
            break
    map_func = os.path.abspath if abspath else os.path.relpath
    return list(map(map_func, sorted(file_list)))  # type: ignore


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                               Custom ArgumentParser                                              * #
# * ---------------------------------------------------------------------------------------------------------------- * #

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self) -> List[Any]:
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as _:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                   Blender utils                                                  * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def set_output_properties(
    scene: bpy.types.Scene,
    image_size: Tuple[int, int],
    output_file_path: str = ""
) -> None:
    scene.render.resolution_x = image_size[0]
    scene.render.resolution_y = image_size[1]
    scene.render.resolution_percentage = 100

    if output_file_path:
        scene.render.filepath = output_file_path


def set_animation(
    scene: bpy.types.Scene,
    fps: int = 60,
    frame_start: int = 1,
    frame_end: int = 5,
    frame_current: int = 1
) -> None:
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_current = frame_current


def set_eevee_renderer(
    scene: bpy.types.Scene,
    camera_object: bpy.types.Object,
    num_samples: int,
    use_motion_blur: bool = False,
    use_transparent_bg: bool = True,
    small_size: bool = False,
    file_format: str = "mp4",
) -> None:
    scene.camera = camera_object

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.use_motion_blur = use_motion_blur
    scene.render.film_transparent = use_transparent_bg

    # video
    if file_format == "mp4":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS" if small_size else "LOSSLESS"
    else:
        assert file_format == "png", "Unknown format: {}".format(file_format)
        scene.render.image_settings.file_format = "PNG"

    scene.eevee.taa_render_samples = num_samples
