"""Visualization helpers: offscreen MP4/PNG (headless default) and the interactive viewer.

Headless rendering needs MUJOCO_GL=egl (GPU) or osmesa (CPU); the viewer needs a display.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Callable

import mujoco


def check_gl() -> None:
    if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa", "glfw"):
        warnings.warn("MUJOCO_GL is not set; on a headless server use `export MUJOCO_GL=egl` (GPU) or osmesa.")


def free_camera(cfg: dict) -> mujoco.MjvCamera:
    rc = cfg["render"]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = rc["lookat"]
    cam.distance, cam.azimuth, cam.elevation = rc["distance"], rc["azimuth"], rc["elevation"]
    return cam


def snapshot(m: mujoco.MjModel, d: mujoco.MjData, cfg: dict, path: str | Path) -> Path:
    import imageio.v3 as iio

    check_gl()
    rc = cfg["render"]
    with mujoco.Renderer(m, height=rc["height"], width=rc["width"]) as r:
        r.update_scene(d, camera=free_camera(cfg))
        img = r.render()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, img)
    return path


class VideoRecorder:
    """Captures a frame whenever sim time crosses the next frame time.

    slowmo > 1 plays back slower: frames are taken every 1/(fps*slowmo) s of sim time.
    """

    def __init__(self, m: mujoco.MjModel, cfg: dict, path: str | Path, slowmo: float = 1.0):
        import imageio.v2 as iio

        check_gl()
        rc = cfg["render"]
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(m, height=rc["height"], width=rc["width"])
        self.writer = iio.get_writer(self.path, fps=rc["fps"], macro_block_size=8)
        self.cam = free_camera(cfg)
        self.dt_frame = 1.0 / (rc["fps"] * slowmo)
        self.t_next = 0.0

    def maybe_capture(self, d: mujoco.MjData) -> None:
        if d.time + 1e-9 >= self.t_next:
            self.renderer.update_scene(d, camera=self.cam)
            self.writer.append_data(self.renderer.render())
            self.t_next += self.dt_frame

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def simulate(m: mujoco.MjModel, d: mujoco.MjData, duration: float,
             control: Callable[[mujoco.MjModel, mujoco.MjData], bool | None],
             pre_step: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
             post_step: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
             cfg: dict | None = None, video: str | Path | None = None, viewer: bool = False,
             slowmo: float = 1.0, video_duration: float | None = None) -> None:
    """Run the physics loop: pre_step (state edits) -> mj_step1 -> control (sets ctrl) -> mj_step2 -> post_step.

    mj_step1 computes all position/velocity-dependent quantities (incl. qfrc_bias) for the
    current state, so the controller sees a consistent state; mj_step2 integrates.
    If control returns True the run stops after that step (e.g. juggling finished or failed).
    The video, if any, stops after `video_duration` seconds of sim time (None = whole run).
    """
    rec = VideoRecorder(m, cfg, video, slowmo) if video else None
    view = None
    if viewer:
        from mujoco import viewer as mj_viewer   # imported lazily: needs a display

        view = mj_viewer.launch_passive(m, d)
        view.cam.lookat[:] = cfg["render"]["lookat"]
        view.cam.distance = cfg["render"]["distance"]
        view.cam.azimuth = cfg["render"]["azimuth"]
        view.cam.elevation = cfg["render"]["elevation"]
    n_steps = int(round(duration / m.opt.timestep))
    wall0, sim0 = time.perf_counter(), d.time
    try:
        if rec:
            rec.maybe_capture(d)
        for _ in range(n_steps):
            if view is not None and not view.is_running():
                break
            if pre_step:
                pre_step(m, d)
            mujoco.mj_step1(m, d)
            stop = control(m, d)
            mujoco.mj_step2(m, d)
            if post_step:
                post_step(m, d)
            if rec and (video_duration is None or d.time <= video_duration):
                rec.maybe_capture(d)
            if stop:
                break
            if view is not None:
                view.sync()
                lag = (d.time - sim0) * slowmo - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
    finally:
        if rec:
            rec.close()
        if view is not None:
            view.close()
