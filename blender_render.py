import os
import re
import subprocess
import shutil
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

_DIR = os.path.dirname(os.path.abspath(__file__))
ImageU8 = npt.NDArray[np.uint8]


def _fill_hole(trans: ImageU8):
    im_th = trans[..., -1] if trans.ndim == 3 else trans
    assert im_th.ndim == 2
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)  # type: ignore
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # type: ignore
    im_out = im_th | im_floodfill_inv
    return im_out[:, :, None]


def _postprocess_fill_inner_hole(filepath: str):
    if os.path.splitext(filepath)[1].lower() != '.png':
        return

    im: ImageU8 = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # type: ignore
    if im.shape[-1] == 3:
        return

    im_alpha = _fill_hole(im[..., 3])
    im = np.concatenate([im[..., :3], im_alpha], axis=-1)  # type: ignore
    cv2.imwrite(filepath, im)


def _postprocess_get_contour(filepath: str):
    if os.path.splitext(filepath)[1].lower() != '.png':
        return

    im: ImageU8 = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # type: ignore
    if im.shape[-1] == 3:
        return
    
    def dilatation(img: ImageU8, ksz: int = 5):
        kernel = np.ones((ksz, ksz), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)
        if img_dilation.ndim == 2:
            img_dilation = img_dilation[..., None]
        return img_dilation

    im_alpha = im[..., 3:]
    im_contour = dilatation(im_alpha) - im_alpha
    
    txt_path = os.path.splitext(filepath)[0] + ".txt"
    if os.path.exists(txt_path):
        xy_list = []
        with open(txt_path) as fp:
            for line in fp:
                x, y = line.strip().split()
                xy_list.append((int(x), int(y)))
        assert len(xy_list) == 59
        for idx_list in [
            list(range(0, 8)) + [0],
            list(range(8, 16)) + [8],
            list(range(16, 29)),
            list(range(29, 57)) + [29],
        ]:
            for k in range(len(idx_list) - 1):
                i = idx_list[k]
                j = idx_list[k + 1]
                cv2.line(im_contour, xy_list[i], xy_list[j], (255, 255, 255), thickness=2)

    cv2.imwrite(os.path.splitext(filepath)[0] + "_cntr.png", im_contour)


class Pattern(object):
    N_FRAMES = r"Will render (\d+) frames"
    OUTPUT = r"\(\+\) Render (video|images): (.+)"
    PROGRESS = r"Append frame (\d+)"
    PROGRESS_IMG = r"Saved: '.*/(\d+)\.png'"
    FRAME_INFO = r"^Fra:.*"
    TIME_INFO = r" Time:.*"
    ANY = r"(.+)"

    ALL = [N_FRAMES, OUTPUT, PROGRESS, PROGRESS_IMG, FRAME_INFO, TIME_INFO, ANY]
    ALL_COMPILED = [re.compile(x) for x in ALL]
    INDEX_OF = {j: i for i, j in enumerate(ALL)}

    @classmethod
    def match(cls, txt: str):
        for regex, pattern in zip(cls.ALL, cls.ALL_COMPILED):
            res = pattern.match(txt)
            if res is not None:
                return regex, res
        return None, None


class BlenderRenderProgress(object):
    def __init__(
        self,
        *args: Any,
        progress: Optional[Progress] = None,
        task: Optional[int] = None,
        keep_showing: bool = False,
        fill_inner_hole: bool = True,
        draw_contour: bool = True,
        **kwargs: Any
    ):
        # modify args
        if "format" in kwargs and kwargs["format"] != "mp4":
            if kwargs["output_prefix"][-1] != "/":
                kwargs["output_prefix"] += "/"

        self.line = ""
        self.line_bytes = bytearray()
        self.args = list(args)
        for k, v in kwargs.items():
            if isinstance(v, bool):
                # NOTE: Only 'store_true' is considered.
                if v:
                    self.args.append("--" + k)
            elif v is not None:
                if isinstance(v, (tuple, list)):
                    self.args.append("--" + k)
                    for x in v:
                        self.args.append(str(x))
                else:
                    self.args.append("--" + k)
                    self.args.append(str(v))
        if progress is None:
            self.progress = Progress(
                "{task.description}",
                SpinnerColumn(),
                TextColumn("[progress.percentage][{task.completed:.0f}/{task.total:.0f}] {task.percentage:>3.0f}%"),
                BarColumn(bar_width=None),
                # "•",
                TimeElapsedColumn(),
                "<",
                TimeRemainingColumn(),
            )
            self.stop_on_end = True
        else:
            self.progress: Progress = progress
            self.stop_on_end = False
        self.task: Optional[int] = task
        self.keep_showing = keep_showing
        self.fill_inner_hole = fill_inner_hole
        self.draw_contour = draw_contour

        # Find the blender
        self.blender_cmd = ""
        for path in [
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ]:
            ret = shutil.which(path)
            if ret is not None and len(ret) > 0:
                self.blender_cmd = ret
                break
        assert len(self.blender_cmd) > 0, f"Failed to find executable blender cmd!"

    def new_task(self):
        # information
        self._is_started = False
        self.n_frames = 0
        self.output = None

    def run(self) -> bool:
        self.new_task()

        # * run in subprocess and iter each line of output
        args = [
            self.blender_cmd,
            "--background",
            os.path.join(_DIR, "blender_project", "render_flame.blend"),
            "--python",
            os.path.join(_DIR, "blender_project", "render_script.py"),
            "--render-anim",
            "--",
        ] + list(self.args)
        # print(" ".join(args))
        # quit()

        p = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        try:
            while True:
                line = None
                out = p.stdout.read(1)
                if out == b"" and p.poll() is not None:
                    break
                elif out != b"":
                    if out not in b"\r\n":
                        self.line_bytes.extend(out)
                    if out in b"\r\n" or self.line_bytes[-6:] == bytearray(b"[y/N] "):
                        if self.line_bytes[-6:] == bytearray(b"[y/N] "):
                            p.stdin.write(b"\n")
                        line = self.new_line()
                # nothing to print
                if line is not None:
                    self.process_line(line)
            if self.task is not None:
                self.progress.update(self.task, completed=self.n_frames, visible=self.keep_showing, refresh=True)
            if self.stop_on_end:
                self.progress.stop()
            return True
        except KeyboardInterrupt:
            p.kill()
            self.safe_exit()
            raise KeyboardInterrupt()
        except Exception as e:
            p.kill()
            self.safe_exit()
            raise e

        return False

    def safe_exit(self):
        if self.task is not None:
            self.progress.remove_task(self.task)
            if self.stop_on_end:
                self.progress.stop()

    def new_line(self):
        self.line = bytes(self.line_bytes).decode("utf-8")
        self.line_bytes = bytearray()
        return self.line

    def process_line(self, line: str):
        # print(line)
        pattern, match = Pattern.match(line)

        if pattern is Pattern.N_FRAMES:
            self.n_frames = int(match.group(1))
        elif pattern is Pattern.OUTPUT:
            self.output = match.group(2)
        elif pattern in [Pattern.PROGRESS, Pattern.PROGRESS_IMG]:
            cur_frame = int(match.group(1))

            if self.progress is not None and self.task is None:
                self.task = self.progress.add_task("")
            if self.task is not None and not self._is_started:
                self.progress.start()
                self.progress.start_task(self.task)
                self.progress.update(
                    self.task,
                    description="Blender: " + os.path.basename(self.output),
                    total=self.n_frames,
                    completed=0,
                    visible=True,
                    refresh=True,
                )
                self._is_started = True
            if self.task is not None:
                self.progress.update(self.task, completed=cur_frame)
            
            if self.output is not None:
                filepath = os.path.join(self.output, f"{cur_frame:04d}.png")
                if os.path.exists(filepath) and self.fill_inner_hole:
                    _postprocess_fill_inner_hole(filepath)
                if os.path.exists(filepath) and self.draw_contour:
                    _postprocess_get_contour(filepath)
                # if os.path.exists(filepath):
        elif pattern in [Pattern.FRAME_INFO, Pattern.TIME_INFO]:
            pass
        else:
            # print(line)
            pass

    # N_FRAMES = r"Will render (\d+) frames"
    # OUTPUT = r"\(\+\) Render video: (.+)"
    # PROGRESS = r"Append frame (\d+)"
    # FRAME_INFO = r"^Fra:.*"
    # TIME_INFO = r" Time:.*"
    # ANY = r"(.+)"


def blender_render(
    source_path: str,
    output_prefix: str,
    stt_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: int = 25,
    format: str = "png",
    is_source_offsets: bool = False,
    idle_fpath: Optional[str] = None,
    image_w: int = 688,
    image_h: int = 880,
    render_samples: int = 64,
    camera: str = "Camera",
    smooth_shading: bool = True,
    with_texture: bool = False,
    base_color: Optional[Tuple[float, float, float, float]] = None,
    small_size: bool = False,
    keep_showing: bool = True,
    fill_inner_hole: bool = False,
    draw_contour: bool = False,
    **kwargs: Any,
):
    if base_color is not None:
        assert len(base_color) == 4, "'base_color' must be a 4-float tuple."

    pbar = BlenderRenderProgress(
        source_path=source_path,
        output_prefix=output_prefix,
        stt_frame=stt_frame,
        end_frame=end_frame,
        fps=fps,
        format=format,
        is_source_offsets=is_source_offsets,
        idle_fpath=idle_fpath,
        image_w=image_w,
        image_h=image_h,
        render_samples=render_samples,
        camera=camera,
        smooth_shading=smooth_shading,
        with_texture=with_texture,
        base_color=base_color,
        small_size=small_size,
        keep_showing=keep_showing,
        fill_inner_hole=fill_inner_hole,
        draw_contour=draw_contour,
        **kwargs
    )
    pbar.run()
    return pbar.output


if __name__ == "__main__":
    import sys
    blender_render(
        source_path=sys.argv[1],
        output_prefix="./test_out/",
        smooth_shading=True,
        end_frame=50,
        base_color=(0, 0.7, 0.7, 0.5),
        format="png",
        fill_inner_hole=True,
        draw_contour=True,
    )