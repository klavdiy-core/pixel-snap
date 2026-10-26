#!/usr/bin/env python3
"""pixel-snap: put fake pixel art back on its real grid.

Finds the true pixel size and grid offset of an upscaled / AI-generated
"pixel art" image, takes one colour per cell, snaps colours to N colours
or to a given palette, and reports the mixed cells (mixels).
"""
import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

__version__ = "0.5.2"


@dataclass
class Grid:
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    smooth: bool = False   # smooth upscale: the true colour is only at the cell centre
    warning: str = ""      # what was decided about the axes, said out loud


@dataclass
class Result:
    image: np.ndarray      # one pixel per cell, uint8 RGB or RGBA
    mixed: np.ndarray      # bool, one flag per cell
    grid: Grid
    xs: list               # cell boundaries along x (len = cells + 1)
    ys: list
    colors: np.ndarray     # final colours used, uint8 Nx3


# ---------- grid detection ----------

def _edge_profile(rgb, axis, order=1):
    """Mean colour jump between neighbouring columns (axis=1) or rows (axis=0).

    order=2 is the mean bend instead: after a smooth (bilinear/bicubic) upscale every
    cell is a ramp, there are no jumps, but the ramps bend at the cell centres.
    """
    d = np.abs(np.diff(rgb.astype(np.int32), n=order, axis=axis)).sum(axis=2)
    return d.mean(axis=0 if axis == 1 else 1)


def _coherence(edge, pos, periods):
    """|sum e * exp(2*pi*i*x/p)| / sum e for each period p, plus the phase."""
    out = np.empty(len(periods), dtype=np.complex128)
    for k in range(0, len(periods), 512):
        p = periods[k:k + 512, None]
        out[k:k + 512] = (edge[None, :] * np.exp(2j * np.pi * pos[None, :] / p)).sum(1)
    return np.abs(out) / edge.sum(), np.angle(out)


def _best_period(edge, max_period=None, around=None):
    """Return (period, offset, coherence). A cell starts at x = i+1 for every jump edge[i].

    Cell borders sit at offset + k*period. Their jumps add up in phase at the true
    period and at its divisors (p/2, p/3 ...), and cancel at its multiples, so the
    answer is the LARGEST strong peak of the edge spectrum.
    """
    n = len(edge) + 1
    if edge.max() <= 0 or n < 8:
        return 1.0, 0.0, 0.0
    edge = edge - np.median(edge)          # drop the noise floor
    edge[edge < 0] = 0
    if edge.sum() <= 0:
        return 1.0, 0.0, 0.0
    pos = np.arange(1, n, dtype=np.float64)
    hi = min(max_period or n / 4, n / 4)  # at least four cells, or every period looks coherent
    lo = 2.0
    if around:                             # a hint, or the other axis: look only near it
        lo, hi = max(2.0, around[0]), min(hi, around[1])
    if hi <= lo:
        return 1.0, 0.0, 0.0
    periods = np.exp(np.arange(np.log(lo), np.log(hi), 0.001))
    coh, _ = _coherence(edge, pos, periods)
    best = coh.max()
    # random noise reaches a coherence of about 4.9 / sqrt(n) at its best period (measured 25.10: 30 seeds per size,
    # 24..256 px), so a fixed 0.3 let a 64 px noise image «find» a 13 x 3 grid; real grids sit at 0.86..1.0
    if best < max(0.3, 5.5 / np.sqrt(n)):
        return 1.0, 0.0, 0.0
    peaks = [i for i in range(1, len(coh) - 1)
             if coh[i] >= coh[i - 1] and coh[i] >= coh[i + 1] and coh[i] >= 0.8 * best]
    i = max(peaks) if peaks else int(coh.argmax())
    # refine around the chosen peak
    fine = np.linspace(periods[max(i - 2, 0)], periods[min(i + 2, len(periods) - 1)], 81)
    fcoh, fang = _coherence(edge, pos, fine)
    j = int(fcoh.argmax())
    p = float(fine[j])
    return p, float((fang[j] / (2 * np.pi) * p) % p), float(fcoh[j])


BEND_BLUR = 1.0   # measured: 2.0 doubled the period of small, already-blurred cells; 1.0 beat it everywhere


def _blur(rgb, sigma=None):
    sigma = sigma or BEND_BLUR
    """Separable Gaussian blur. The bends of a smooth upscale are low-frequency and survive it;
    JPEG's 8x8 block seams and ringing are high-frequency and do not."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = rgb.astype(np.float64)
    for axis in (0, 1):
        pad = [(r, r) if a == axis else (0, 0) for a in range(out.ndim)]
        p = np.pad(out, pad, mode="reflect")
        out = sum(k[i] * np.take(p, np.arange(i, i + out.shape[axis]), axis=axis) for i in range(len(k)))
    return out


def _axis_grid(rgb, axis, max_scale, soft=None, around=None):
    """(period, offset, smooth, coherence) of one axis."""
    p1, o1, c1 = _best_period(_edge_profile(rgb, axis), max_scale, around)
    p2, o2, c2 = _best_period(_edge_profile(rgb if soft is None else soft, axis, order=2), max_scale, around)
    if p2 > 1 and c2 > c1:
        # the bends are cleaner than the jumps: a smooth upscale.
        # bend[i] sits on pixel i+1, the centre of a cell; its border is half a cell back.
        return p2, (o2 + 0.5 - p2 / 2) % p2, True, c2
    return p1, o1, False, c1


def cell_fit(rgb, px, py, ox, oy):
    """Share of the image's colour variance that the cells' own means explain (1 = every cell one flat colour).
    It does not depend on how the period was chosen: real pixel art keeps most of its variance between cells even
    after a smooth upscale or JPEG; a photo given a 60 px «grid» does not (external review, 25.10)."""
    a = np.asarray(rgb, dtype=np.float64)[..., :3]
    h, w = a.shape[:2]
    xs, ys = _bounds(w, px, ox), _bounds(h, py, oy)
    lx = np.searchsorted(xs, np.arange(w), side="right") - 1
    ly = np.searchsorted(ys, np.arange(h), side="right") - 1
    lab = ly[:, None] * (len(xs) + 1) + lx[None, :]
    flat = lab.ravel()
    n = np.bincount(flat).astype(np.float64)
    total = ((a - a.reshape(-1, 3).mean(0)) ** 2).sum()
    if total <= 0:
        return 1.0
    within = 0.0
    for c in range(3):
        v = a[..., c].ravel()
        s = np.bincount(flat, weights=v)
        s2 = np.bincount(flat, weights=v * v)
        within += (s2 - np.where(n > 0, s * s / np.maximum(n, 1), 0)).sum()
    return float(1 - within / total)


GRID_FIT = 0.75     # measured 25.10: 960 upscaled sprites at their right grid all >= 0.836 (nearest, bicubic, bilinear,
                    # Lanczos, JPEG 60); 30 photos with their invented grids <= 0.73
SAME_PIXEL = 1.15   # axes within 15 % of each other are one square pixel found twice with an error
MIXED_FORCED = 0.2  # with --scale, more than this share of mixed cells means the forced grid is off


def detect_grid(img, max_scale=None, hint=None, square=True):
    """hint: the pixel size you expect; only periods within 15 % of it are searched. square: when the two axes
    disagree a little, or one finds nothing, use the stronger axis's period for both (pixel art is square);
    a large disagreement (5 x 9) is a real non-square pixel and is kept. Every such decision is in .warning."""
    rgb = np.asarray(img)[..., :3]
    soft = _blur(rgb)
    near = (hint / SAME_PIXEL, hint * SAME_PIXEL) if hint else None
    px, ox, sx, cx = _axis_grid(rgb, 1, max_scale, soft, near)
    py, oy, sy, cy = _axis_grid(rgb, 0, max_scale, soft, near)
    found = (px, py, ox, oy)            # as the two axes saw it, before any square decision
    warning = ""
    if square and (px > 1) != (py > 1):
        p = max(px, py)
        axis = 0 if px > 1 else 1                        # the axis that found nothing
        q, o, s, _ = _axis_grid(rgb, axis, max_scale, soft, (p * 0.98, p * 1.02))
        o = o if q > 1 else 0.0
        if axis == 0:
            py, oy, sy = p, o, s
        else:
            px, ox, sx = p, o, s
        warning = (f"the {'y' if axis == 0 else 'x'} axis showed no grid; used the other axis's pixel size "
                   f"{p:.3g} for both (square pixels)")
    elif px > 1 and py > 1 and max(px, py) / min(px, py) > 1.03:
        r = max(px, py) / min(px, py)
        if square and r <= SAME_PIXEL:
            p, keep = (px, "x") if cx >= cy else (py, "y")
            weak = 0 if keep == "x" else 1
            q, o, s, _ = _axis_grid(rgb, weak, max_scale, soft, (p * 0.98, p * 1.02))
            was = (px, py)
            if keep == "x":
                py, oy = p, (o if q > 1 else oy)
            else:
                px, ox = p, (o if q > 1 else ox)
            warning = (f"axes disagreed: x {was[0]:.3g}, y {was[1]:.3g}; used square {p:.3g} from the stronger "
                       f"{keep} axis. Check it, or pass --scale (or --nonsquare to keep both)")
        else:
            warning = (f"pixels are not square: x {px:.3g}, y {py:.3g} (aspect 1:{max(px, py) / min(px, py):.2g}); "
                       f"kept as found. Pass --scale to force one size")
    if px > 1 or py > 1:
        # the period search always finds its best period; whether the image IS a pixel grid is a separate question.
        # Every photo tried (astronaut, coffee, a cat ... at 384x256) got a 30-130 px «grid» (external review)
        # no retry at a divisor: smaller cells always fit a smooth photo better (up to 0.89 measured), so a divisor
        # «fitting» proves nothing. A sprite whose period was found at twice its cell gets 1:1 and the --scale hint
        # instead of a wrong grid.
        # measured on the grid both axes found (a slightly stretched sprite forced square misfits by design); when
        # one axis found nothing, on the grid that is used
        fx, fy, fox, foy = found if found[0] > 1 and found[1] > 1 else (px, py, ox, oy)
        fit = cell_fit(rgb, fx, fy, fox, foy)
        if fit < GRID_FIT:
            # a near miss on a whole-number pixel is usually 1-px dithering through a smooth filter: name the flag
            # (a second review: 53 of 443 such sprites went 1:1 at fit 0.67-0.75; with --scale they come back whole)
            p = max(px, py)
            hint = (f"if this IS pixel art (1-px dithering through a smooth upscale fits poorly), pass --scale "
                    f"{round(p)}" if fit >= 0.6 and abs(p - round(p)) < 0.1 else "Pass --scale if you know the pixel size")
            return Grid(1.0, 1.0, 0.0, 0.0, False,
                        f"no pixel grid: the best grid found ({px:.3g} x {py:.3g} px) explains only {fit:.0%} "
                        "of the colour, so this looks like a photo or a painting, not upscaled pixel art; "
                        f"the output is the input, 1:1. {hint}")
    if px <= 1 and py <= 1:
        warning = "no grid found: this does not look like an upscaled pixel image; the output is the input, 1:1"
    return Grid(px, py, ox, oy, sx or sy, warning)


def _bounds(n, scale, offset):
    if scale <= 1:
        return list(range(n + 1))
    k0 = int(np.floor(-offset / scale))
    starts = (offset + scale * np.arange(k0, k0 + int(n / scale) + 3))
    starts = [int(round(x)) for x in starts if 1 <= round(x) <= n - 1]
    return sorted(set([0] + starts + [n]))


# ---------- colours ----------

def _parse_hex(s):
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bad hex colour: {s!r}")
    return [int(s[i:i + 2], 16) for i in (0, 2, 4)]


def load_palette(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    cols = []
    if lines and lines[0].strip() == "GIMP Palette":
        for ln in lines[1:]:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ":" in ln:
                continue
            parts = ln.split()
            cols.append([int(v) for v in parts[:3]])
    else:
        for ln in lines:
            ln = ln.split(";")[0].strip()
            if ln:
                cols.append(_parse_hex(ln))
    if not cols:
        raise ValueError(f"no colours in {path}")
    return np.array(cols, dtype=np.uint8)


def _nearest(colors, palette):
    c = colors.astype(np.int32)[:, None, :]
    p = palette.astype(np.int32)[None, :, :]
    return ((c - p) ** 2).sum(axis=2).argmin(axis=1)


def _kmeans(colors, k, iters=30):
    """Deterministic k-means with farthest-point init."""
    pts = colors.astype(np.float64)
    uniq = np.unique(colors, axis=0)
    if len(uniq) <= k:
        return uniq.astype(np.uint8)
    centers = [np.median(pts, axis=0)]
    for _ in range(k - 1):
        d = np.min([((pts - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(pts[d.argmax()])
    centers = np.array(centers)
    for _ in range(iters):
        lab = ((pts[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
        new = np.array([pts[lab == i].mean(0) if (lab == i).any() else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return np.clip(np.rint(centers), 0, 255).astype(np.uint8)


# ---------- snapping ----------

def snap(img, colors=None, palette=None, grid=None, mix_threshold=48.0, mix_fraction=0.2):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    has_alpha = arr.shape[2] == 4
    grid = grid or detect_grid(arr)
    h, w = arr.shape[:2]
    xs = _bounds(w, grid.scale_x, grid.offset_x)
    ys = _bounds(h, grid.scale_y, grid.offset_y)
    gh, gw = len(ys) - 1, len(xs) - 1
    out = np.zeros((gh, gw, 4), dtype=np.uint8)
    mixed = np.zeros((gh, gw), dtype=bool)
    # the centre of the cell: blur and anti-aliasing live on the borders;
    # after a smooth upscale only the middle pixel (or the two around the middle) holds the true colour
    if grid.smooth:
        def window(a, b):
            return a + (b - a - 1) // 2, a + (b - a) // 2 + 1
    else:
        def window(a, b):
            return a + (b - a) // 4, b - (b - a) // 4
    for j in range(gh):
        wy0, wy1 = window(ys[j], ys[j + 1])
        for i in range(gw):
            wx0, wx1 = window(xs[i], xs[i + 1])
            cell = arr[wy0:wy1, wx0:wx1].reshape(-1, arr.shape[2]).astype(np.int32)
            med = np.median(cell, axis=0)
            out[j, i, :3] = np.rint(med[:3])
            out[j, i, 3] = (255 if med[3] >= 128 else 0) if has_alpha else 255
            dist = np.sqrt(((cell[:, :3] - med[:3]) ** 2).sum(1))
            mixed[j, i] = (dist > mix_threshold).mean() > mix_fraction

    opaque = out[..., 3] > 0
    cols = out[..., :3][opaque]
    if not opaque.any():
        pass             # a fully transparent sprite: nothing to recolour (argmin of nothing raised - external review)
    elif palette is not None:
        palette = np.asarray(palette, dtype=np.uint8)
        out[..., :3][opaque] = palette[_nearest(cols, palette)]
    elif colors:
        centers = _kmeans(cols, colors)
        out[..., :3][opaque] = centers[_nearest(cols, centers)]
    out[..., :3][~opaque] = 0
    used = np.unique(out[..., :3][opaque], axis=0) if opaque.any() else np.zeros((0, 3), np.uint8)
    image = out if has_alpha else out[..., :3]
    return Result(image, mixed, grid, xs, ys, used)


def mixel_map(img, res):
    """The source image with mixed cells painted red and the grid drawn faintly."""
    base = np.asarray(Image.fromarray(np.asarray(img)).convert("RGB")).astype(np.float64)
    base = base * 0.6 + 255 * 0.4 * (base.mean(2, keepdims=True) / 255)  # wash out
    for j in range(res.mixed.shape[0]):
        for i in range(res.mixed.shape[1]):
            if res.mixed[j, i]:
                y0, y1, x0, x1 = res.ys[j], res.ys[j + 1], res.xs[i], res.xs[i + 1]
                base[y0:y1, x0:x1] = base[y0:y1, x0:x1] * 0.35 + np.array([255, 0, 0]) * 0.65
    for x in res.xs[1:-1]:
        base[:, x] = base[:, x] * 0.7
    for y in res.ys[1:-1]:
        base[y, :] = base[y, :] * 0.7
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


# ---------- CLI ----------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="pixel-snap", description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="snapped PNG (default: <input>.snapped.png)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("-c", "--colors", type=int, help="reduce to N colours")
    g.add_argument("-p", "--palette", help="snap to a palette file (.hex or GIMP .gpl)")
    ap.add_argument("--scale", type=float, help="force the pixel size instead of detecting it")
    ap.add_argument("--offset", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="force the grid offset (with --scale)")
    ap.add_argument("--scale-hint", type=float, help="the pixel size you expect: search only near it")
    ap.add_argument("--nonsquare", action="store_true", help="keep two slightly different axes as found")
    ap.add_argument("--upscale", type=int, default=1, help="scale the result up by N (nearest neighbour)")
    ap.add_argument("--mixel-map", help="write a PNG with mixed cells marked red")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    try:
        src = Image.open(a.input)
        src.load()
    except (OSError, ValueError) as e:            # missing, unreadable or not an image: one line, not a traceback
        print(f"pixel-snap: cannot open {a.input}: {e.strerror or e}" if isinstance(e, OSError) and e.strerror
              else f"pixel-snap: cannot open {a.input}: {e}", file=sys.stderr)
        return 2
    src = src.convert("RGBA" if "A" in src.getbands() or "transparency" in src.info else "RGB")
    arr = np.asarray(src)
    grid = None
    if a.scale:
        ox, oy = a.offset or (0, 0)
        grid = Grid(a.scale, a.scale, ox % a.scale, oy % a.scale)
    palette = load_palette(a.palette) if a.palette else None
    if grid is None:
        grid = detect_grid(arr, hint=a.scale_hint, square=not a.nonsquare)
    res = snap(arr, colors=a.colors, palette=palette, grid=grid)
    if a.scale and res.mixed.mean() > MIXED_FORCED:
        # a forced size with a wrong offset straddles every border: say it, the detector would not have done this
        res.grid.warning = (f"with --scale {a.scale:g}, {100 * res.mixed.mean():.0f}% of cells are mixed: the size or "
                            f"the offset is probably wrong. Drop --scale, or pass --offset X Y")

    out_path = a.output or a.input.rsplit(".", 1)[0] + ".snapped.png"
    im = Image.fromarray(res.image)
    if a.upscale > 1:
        im = im.resize((im.width * a.upscale, im.height * a.upscale), Image.NEAREST)
    im.save(out_path)
    if a.mixel_map:
        mixel_map(arr, res).save(a.mixel_map)

    report = {
        "input": a.input,
        "input_size": [arr.shape[1], arr.shape[0]],
        "pixel_size": [round(res.grid.scale_x, 3), round(res.grid.scale_y, 3)],
        "offset": [round(res.grid.offset_x, 2), round(res.grid.offset_y, 2)],
        "cells": [res.mixed.shape[1], res.mixed.shape[0]],
        "mixed_cells": int(res.mixed.sum()),
        "mixed_percent": round(100.0 * res.mixed.mean(), 2),
        "colors_used": len(res.colors),
        "warning": res.grid.warning,
        "output": out_path,
    }
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"pixel size  {report['pixel_size'][0]}x{report['pixel_size'][1]}  offset {report['offset']}")
        print(f"cells       {report['cells'][0]}x{report['cells'][1]}  mixed {report['mixed_cells']} "
              f"({report['mixed_percent']}%)")
        print(f"colours     {report['colors_used']}")
        if res.grid.warning:
            print(f"WARNING     {res.grid.warning}")
        print(f"wrote       {out_path}" + (f"  map {a.mixel_map}" if a.mixel_map else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
