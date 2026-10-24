"""Render examples/demo.gif from a REAL run of pixel_snap.

A 16x16 sprite is wrecked the way image models wreck pixel art (fractional pixel
size, blur, JPEG, one patch drawn at another pixel size). The grid lines
are the grid pixel_snap detected; the last frame is the sprite it rebuilt.

    pip install pillow numpy
    python examples/make_demo_gif.py
"""
import io
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pixel_snap  # noqa: E402

SIDE, BAR = 480, 40
BG, FG = (24, 26, 33), (220, 223, 228)


def font():
    for name in ("consolab.ttf", "DejaVuSansMono-Bold.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = font()

COLOURS = {".": (32, 34, 52), "k": (20, 12, 28), "r": (208, 70, 72), "d": (140, 40, 50),
           "w": (240, 236, 220), "s": (222, 190, 150), "b": (150, 110, 80)}
MUSHROOM = """
................
.....kkkkkk.....
...kkrrwwrrkk...
..krrrwwwwrrrk..
.krwwrrwwrrrrrk.
.krwwrrrrrwwrrk.
krrrrrrrrrwwwrrk
krrwwrrrrrrwwrdk
kdrwwwrrrwwrrddk
.kddrrrrrrrrddk.
..kkkkkkkkkkkk..
....kssssssk....
....kskssksk....
....kssssssk....
....kbbbbbbk....
.....kkkkkk.....
"""


def wrecked():
    rows = [r for r in MUSHROOM.split("\n") if r]
    sprite = np.array([[COLOURS[c] for c in r] for r in rows], dtype=np.uint8)
    scale = 13.7
    side = round(16 * scale)
    img = Image.fromarray(sprite).resize((side, side), Image.NEAREST)
    canvas = Image.new("RGB", (side + 11, side + 11), COLOURS["."])
    canvas.paste(img, (5, 6))
    # a mixel: a highlight drawn with pixels half the size, a checker the real grid cannot hold
    half = scale / 2
    d = ImageDraw.Draw(canvas)
    x0, y0 = 5 + 7 * scale, 6 + 7 * scale
    for j in range(4):
        for i in range(4):
            if (i + j) % 2 == 0:
                d.rectangle((x0 + i * half, y0 + j * half, x0 + (i + 1) * half - 1, y0 + (j + 1) * half - 1),
                            fill=COLOURS["w"])
    canvas = canvas.filter(ImageFilter.GaussianBlur(1.0))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=55)
    return Image.open(buf).convert("RGB")


def card(img, caption):
    out = Image.new("RGB", (SIDE, SIDE + BAR), BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle((0, SIDE, SIDE, SIDE + BAR), fill=(40, 43, 54))
    d.text((14, SIDE + 10), caption, font=FONT, fill=FG)
    return out


def main():
    src = wrecked()
    res = pixel_snap.snap(np.asarray(src), palette=np.array(list(COLOURS.values()), dtype=np.uint8))
    g, (gh, gw) = res.grid, res.mixed.shape
    k = SIDE / src.width
    xs, ys = [x * k for x in res.xs], [y * k for y in res.ys]

    messy = src.resize((SIDE, SIDE), Image.NEAREST)
    gridded = messy.copy()
    d = ImageDraw.Draw(gridded)
    for x in xs:
        d.line((x, 0, x, SIDE), fill=(0, 230, 255), width=1)
    for y in ys:
        d.line((0, y, SIDE, y), fill=(0, 230, 255), width=1)
    for j, i in zip(*np.nonzero(res.mixed)):
        d.rectangle((xs[i], ys[j], xs[i + 1], ys[j + 1]), outline=(255, 60, 60), width=3)
    clean = Image.fromarray(res.image).convert("RGB").resize((SIDE, SIDE), Image.NEAREST)

    frames = [
        card(messy, "BEFORE: 13.7 px pixels, blur, JPEG, a mixel"),
        card(gridded, f"real grid: {g.scale_x:.2f} px cells, {int(res.mixed.sum())} broken (red)"),
        card(clean, f"AFTER: {gw}x{gh} sprite, {len(np.unique(res.image.reshape(-1, res.image.shape[-1]), axis=0))} colours"),
    ]
    out = os.path.join(HERE, "demo.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[2200, 2600, 3200], loop=0, optimize=True)
    print(f"{out}: {os.path.getsize(out) // 1024} KB, grid {g.scale_x:.3f}x{g.scale_y:.3f}, mixed {int(res.mixed.sum())}")


if __name__ == "__main__":
    main()
