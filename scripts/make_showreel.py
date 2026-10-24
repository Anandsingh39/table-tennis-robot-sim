"""Presentation videos (~1 min each), assembled from simulation clips with captions.

  results/showreel_project.mp4   whole project: juggling -> noisy juggling -> MPC return -> lob rally -> rails ->
                                 spin -> flat drives -> self-play RL (existing clips, trimmed and captioned)
  results/showreel_flat.mp4      flat-drive rally from several camera angles + slow motion (new renders)

Usage:
    python scripts/make_showreel.py [--which project|flat|both] [--skip-render]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import REPO, RESULTS, load_config  # noqa: E402

W, H, FPS = 1280, 720, 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CLIPS = RESULTS / "showreel_clips"

# camera presets (MuJoCo free camera), chosen from test snapshots
CAMS = {
    "side": dict(lookat=[2.02, 0.0, 1.0], distance=3.6, azimuth=90.0, elevation=-8.0),
    "behind_a": dict(lookat=[1.2, 0.0, 1.0], distance=2.6, azimuth=0.0, elevation=-14.0),
    "top": dict(lookat=[2.02, 0.0, 0.8], distance=4.2, azimuth=90.0, elevation=-70.0),
    "close_b": dict(lookat=[3.55, -0.05, 1.05], distance=1.5, azimuth=90.0, elevation=-6.0),
}

# (name, camera, seed, hits, slow-motion factor)
FLAT_RENDERS = [
    ("flat_side", "side", 3, 14, 1.0),
    ("flat_behind_a", "behind_a", 5, 14, 1.0),
    ("flat_top", "top", 7, 14, 1.0),
    ("flat_slowmo", "close_b", 9, 5, 3.0),
]


def _font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def fit(frame: np.ndarray) -> np.ndarray:
    """Letterbox any frame into W x H."""
    img = Image.fromarray(frame)
    s = min(W / img.width, H / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (12, 20, 32))
    canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return np.asarray(canvas)


def caption(frame: np.ndarray, text: str, tag: str | None = None) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    d.rectangle([0, H - 64, W, H], fill=(0, 0, 0, 170))
    d.text((24, H - 50), text, font=_font(28), fill=(255, 255, 255, 255))
    if tag:
        d.rectangle([0, 0, W, 38], fill=(0, 0, 0, 110))
        d.text((16, 7), tag, font=_font(20), fill=(220, 220, 220, 255))
    return np.asarray(Image.alpha_composite(img, over).convert("RGB"))


def card(title: str, lines: list[str], seconds: float) -> list[np.ndarray]:
    img = Image.new("RGB", (W, H), (14, 28, 48))
    d = ImageDraw.Draw(img)
    tw = d.textlength(title, font=_font(46, True))
    d.text(((W - tw) / 2, H / 2 - 40 - 30 * len(lines)), title, font=_font(46, True), fill=(255, 255, 255))
    for i, ln in enumerate(lines):
        lw = d.textlength(ln, font=_font(28))
        d.text(((W - lw) / 2, H / 2 + 30 + 42 * i - 30 * len(lines) + 30), ln, font=_font(28), fill=(200, 215, 230))
    f = np.asarray(img)
    return [f] * int(round(seconds * FPS))


def clip(path: Path, start: float, dur: float | None, text: str, tag: str | None) -> list[np.ndarray]:
    rd = iio.get_reader(path)
    src_fps = rd.get_meta_data()["fps"]
    frames = []
    for i, fr in enumerate(rd):
        t = i / src_fps
        if t < start:
            continue
        if dur is not None and t >= start + dur:
            break
        # resample to FPS: keep a source frame when it is the first one at/after the next output time
        if len(frames) <= (t - start) * FPS:
            frames.append(caption(fit(fr), text, tag))
    rd.close()
    return frames


def write(frames, path: Path):
    wr = iio.get_writer(path, fps=FPS, macro_block_size=8, quality=8)
    for f in frames:
        wr.append_data(f)
    wr.close()
    print(f"wrote {path} ({len(frames) / FPS:.1f} s)")


def _render(args):
    name, cam, seed, hits, slowmo = args
    from tt.rally2 import run_rally

    cfg = load_config(REPO / "config" / "rally_flat.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["render"].update(CAMS[cam])
    out = CLIPS / f"{name}.mp4"
    r = run_rally(cfg, seed, hits, video=out, slowmo=slowmo)
    return name, len(r.hits), r.fault


def project_reel():
    tag = "OpenArm table tennis robots · MuJoCo simulation"
    segs = [
        ("juggle_nominal.mp4", 2.0, 7.0, "1 · Juggling with a hitting-velocity law: 200 consecutive bounces, apex error < 1 mm"),
        ("juggle_noisy.mp4", 2.0, 6.0, "2 · Juggling from noisy measurements (200 Hz, σ = 2 mm) + Kalman filter"),
        ("e1_return.mp4", 0.0, 6.0, "3 · MPC swing: one arm returns launched balls (100/100 returned)"),
        ("e2_rally.mp4", 0.0, 7.0, "4 · Two robots with MPC: cooperative rally (20/20 rallies × 30 hits)"),
        ("e2_rail_rally.mp4", 0.0, 7.0, "5 · Robots on sideways rails: balls anywhere across the table"),
        ("e2_spin_v1_rally.mp4", 0.0, 7.0, "6 · Spin: topspin & backspin with a grippy-rubber contact model"),
        ("e2_flat_rally.mp4", 0.0, 8.0, "7 · Flat drives along the table (paddle mounted at 80°)"),
        ("rl_point.mp4", 0.0, None, "8 · Self-play RL shot strategy: aims low & wide — opponent cannot reach"),
    ]
    frames = card("Table tennis robots in simulation",
                  ["OpenArm 7-DOF arm + paddle · MuJoCo · torque control",
                   "juggling → MPC returns → two-robot rally → spin, flat drives, self-play RL"], 3.0)
    for f, s, d, txt in segs:
        frames += clip(RESULTS / f, s, d, txt, tag)
    frames += card("Results", ["Juggling: 200/200 bounces in 750 Monte Carlo runs (σ up to 10 mm)",
                               "Two-robot MPC rally: 20/20 rallies × 30 hits, no torque saturation"], 2.5)
    write(frames, RESULTS / "showreel_project.mp4")


def flat_reel(render: bool):
    CLIPS.mkdir(parents=True, exist_ok=True)
    if render:
        with ProcessPoolExecutor(len(FLAT_RENDERS)) as ex:
            for name, hits, fault in ex.map(_render, FLAT_RENDERS):
                print(f"rendered {name}: {hits} hits ({fault})")
    tag = "Flat-drive rally · two OpenArms on rails · MPC swings"
    frames = card("Flat drives along the table",
                  ["two OpenArms, paddles mounted at 80°, sideways rails",
                   "ball 5.8–6.5 m/s, apex ~1.3 m · each robot plans its swing with MPC"], 2.5)
    frames += clip(RESULTS / "e2_flat_rally.mp4", 0.0, None, "Wide view — 12-hit rally", tag)
    frames += clip(CLIPS / "flat_side.mp4", 0.0, None, "Side view — low, fast trajectories over the net", tag)
    frames += clip(CLIPS / "flat_behind_a.mp4", 0.0, None, "Behind robot A — the rail slides to reach wide balls", tag)
    frames += clip(CLIPS / "flat_top.mp4", 0.0, None, "Top view — placement across the table width", tag)
    frames += clip(CLIPS / "flat_slowmo.mp4", 0.0, None, "Slow motion ×3 — paddle meets the ball (robot B)", tag)
    frames += card("Flat-drive rally", ["20/20 rallies reached the 30-hit cap (600 hits)",
                                        "no torque saturation · balls across ±0.5 m of the table"], 2.5)
    write(frames, RESULTS / "showreel_flat.mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="both", choices=["project", "flat", "both"])
    ap.add_argument("--skip-render", action="store_true", help="reuse rendered flat clips")
    args = ap.parse_args()
    if args.which in ("flat", "both"):
        flat_reel(not args.skip_render)
    if args.which in ("project", "both"):
        project_reel()


if __name__ == "__main__":
    main()
