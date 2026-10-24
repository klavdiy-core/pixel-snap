import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pixel_snap  # noqa: E402

PALETTE = np.array(
    [[20, 12, 28], [68, 36, 52], [48, 52, 109], [78, 74, 78],
     [133, 76, 48], [52, 101, 36], [208, 70, 72], [222, 238, 214]],
    dtype=np.uint8,
)


def make_sprite(seed=1, size=16):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(PALETTE), size=(size, size))
    return PALETTE[idx]


def degrade(sprite, scale=7, offset=(3, 5), noise=6, blur=0.8, seed=2):
    """Upscale with nearest neighbour, pad with an off-grid border, blur, add noise."""
    big = np.repeat(np.repeat(sprite, scale, axis=0), scale, axis=1)
    oy, ox = offset
    h, w = big.shape[:2]
    canvas = np.zeros((h + oy + 4, w + ox + 4, 3), dtype=np.uint8)
    canvas[:] = PALETTE[0]
    canvas[oy:oy + h, ox:ox + w] = big
    img = Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img).astype(np.int16)
    arr += np.random.default_rng(seed).integers(-noise, noise + 1, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _circ(a, b, p):
    """Distance between two offsets on a circle of length p."""
    d = abs(a - b) % p
    return min(d, p - d)


def crop_to_sprite(out, sprite, cell_offset):
    cy, cx = cell_offset
    return out[cy:cy + sprite.shape[0], cx:cx + sprite.shape[1], :3]


class DetectGrid(unittest.TestCase):
    def test_finds_scale_and_offset(self):
        for scale in (4, 7, 8, 12):
            with self.subTest(scale=scale):
                img = degrade(make_sprite(), scale=scale, offset=(3, 2))
                grid = pixel_snap.detect_grid(img)
                self.assertAlmostEqual(grid.scale_x, scale, delta=0.05)
                self.assertAlmostEqual(grid.scale_y, scale, delta=0.05)
                self.assertLess(_circ(grid.offset_x, 2, scale), 0.5)
                self.assertLess(_circ(grid.offset_y, 3, scale), 0.5)


class SnapRoundTrip(unittest.TestCase):
    def test_palette_returns_exact_sprite(self):
        sprite = make_sprite()
        img = degrade(sprite, scale=7, offset=(3, 5))
        res = pixel_snap.snap(img, palette=PALETTE)
        # offset 3/5 < scale 7, so the sprite starts at cell (1, 1)
        got = crop_to_sprite(res.image, sprite, (1, 1))
        np.testing.assert_array_equal(got, sprite)

    def test_n_colors_keeps_the_partition(self):
        sprite = make_sprite(seed=5)
        img = degrade(sprite, scale=7, offset=(0, 0))
        res = pixel_snap.snap(img, colors=len(PALETTE))
        got = crop_to_sprite(res.image, sprite, (0, 0))
        # same cells share a colour exactly when they shared one in the source
        src_ids = np.unique(sprite.reshape(-1, 3), axis=0, return_inverse=True)[1].ravel()
        got_ids = np.unique(got.reshape(-1, 3), axis=0, return_inverse=True)[1].ravel()
        pairs = set(zip(src_ids.tolist(), got_ids.tolist()))
        self.assertEqual(len(pairs), len(set(src_ids.tolist())))
        self.assertEqual(len(pairs), len(set(got_ids.tolist())))

    def test_mixed_cells_are_flagged(self):
        sprite = make_sprite()
        img = degrade(sprite, scale=7, offset=(0, 0), blur=0, noise=0)
        # paint a half-cell smear: one cell gets two colours
        img[7:14, 10:14] = (255, 255, 0)
        res = pixel_snap.snap(img, palette=PALETTE)
        self.assertTrue(res.mixed[1, 1])
        self.assertEqual(int(res.mixed.sum()), 1)


class Attacks(unittest.TestCase):
    def test_different_scale_per_axis(self):
        sprite = make_sprite(seed=3)
        big = np.repeat(np.repeat(sprite, 9, axis=0), 5, axis=1)
        grid = pixel_snap.detect_grid(big)
        self.assertEqual((round(grid.scale_x), round(grid.scale_y)), (5, 9))

    def test_noise_and_blur_are_not_mixels(self):
        img = degrade(make_sprite(), scale=7, offset=(0, 0))
        res = pixel_snap.snap(img, palette=PALETTE)
        self.assertEqual(int(res.mixed.sum()), 0)

    def test_transparent_background_survives(self):
        sprite = make_sprite()
        big = np.repeat(np.repeat(sprite, 6, axis=0), 6, axis=1)
        rgba = np.dstack([big, np.full(big.shape[:2], 255, np.uint8)])
        rgba[:12, :, 3] = 0  # top two rows of cells transparent
        res = pixel_snap.snap(rgba, palette=PALETTE)
        self.assertEqual(res.image.shape, (16, 16, 4))
        self.assertTrue((res.image[:2, :, 3] == 0).all())
        np.testing.assert_array_equal(res.image[2:, :, :3], sprite[2:])

    def test_fractional_scale(self):
        # AI images rarely land on an integer pixel size: 16 cells in 117 px = 7.31 px
        sprite = make_sprite(seed=4)
        for width in (117, 120, 100, 250):
            with self.subTest(width=width):
                im = np.asarray(Image.fromarray(sprite).resize((width, width), Image.NEAREST))
                res = pixel_snap.snap(im, palette=PALETTE)
                self.assertAlmostEqual(res.grid.scale_x, width / 16, delta=0.05)
                np.testing.assert_array_equal(res.image, sprite)

    def test_smooth_upscale(self):
        # bilinear / bicubic upscale: no flat cell centres, every cell is a ramp
        sprite = make_sprite(seed=6)
        for method in (Image.BILINEAR, Image.BICUBIC):
            for width in (112, 219):
                with self.subTest(method=method, width=width):
                    im = np.asarray(Image.fromarray(sprite).resize((width, width), method))
                    res = pixel_snap.snap(im, palette=PALETTE)
                    self.assertTrue(res.grid.smooth)
                    self.assertAlmostEqual(res.grid.scale_x, width / 16, delta=0.05)
                    self.assertAlmostEqual(res.grid.scale_y, width / 16, delta=0.05)
                    np.testing.assert_array_equal(res.image, sprite)

    def test_smooth_upscale_saved_as_jpeg(self):
        # the 8x8 JPEG blocks used to out-shout the faint bends (README limit until 0.3.0)
        import io
        sprite = make_sprite(seed=7)
        for method in (Image.BILINEAR, Image.BICUBIC):
            for width in (219, 300):
                with self.subTest(method=method, width=width):
                    buf = io.BytesIO()
                    Image.fromarray(sprite).resize((width, width), method).save(buf, "JPEG", quality=55)
                    im = np.asarray(Image.open(buf).convert("RGB"))
                    res = pixel_snap.snap(im, palette=PALETTE)
                    self.assertAlmostEqual(res.grid.scale_x, width / 16, delta=0.1)
                    self.assertAlmostEqual(res.grid.scale_y, width / 16, delta=0.1)
                    self.assertEqual(res.image.shape, sprite.shape)
                    self.assertGreaterEqual((res.image == sprite).all(-1).mean(), 0.95)

    def test_crisp_images_are_not_smooth(self):
        for img in (degrade(make_sprite(), scale=7), degrade(make_sprite(), scale=12, blur=0, noise=0)):
            self.assertFalse(pixel_snap.detect_grid(img).smooth)

    def test_flat_image_does_not_crash(self):
        flat = np.full((20, 20, 3), 128, np.uint8)
        res = pixel_snap.snap(flat, colors=4)
        self.assertEqual(res.grid.scale_x, 1.0)
        self.assertEqual(len(res.colors), 1)


class Palettes(unittest.TestCase):
    def test_reads_hex_and_gpl(self):
        with tempfile.TemporaryDirectory() as d:
            hx = os.path.join(d, "p.hex")
            with open(hx, "w") as f:
                f.write("140c1c\n#442434\n")
            gpl = os.path.join(d, "p.gpl")
            with open(gpl, "w") as f:
                f.write("GIMP Palette\nName: t\nColumns: 2\n#\n 20  12  28\tblack\n68 36 52 plum\n")
            want = [[20, 12, 28], [68, 36, 52]]
            self.assertEqual(pixel_snap.load_palette(hx).tolist(), want)
            self.assertEqual(pixel_snap.load_palette(gpl).tolist(), want)


class Cli(unittest.TestCase):
    def test_cli_writes_outputs(self):
        sprite = make_sprite()
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.png")
            Image.fromarray(degrade(sprite)).save(src)
            out = os.path.join(d, "out.png")
            mp = os.path.join(d, "map.png")
            code = pixel_snap.main([src, "-o", out, "--colors", "8", "--mixel-map", mp, "--upscale", "4"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(mp))
            w, h = Image.open(out).size
            self.assertEqual(w % 4, 0)


if __name__ == "__main__":
    unittest.main()
