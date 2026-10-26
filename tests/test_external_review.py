"""An external review (25.10): a fully transparent sprite raised ValueError (argmin of an empty sequence)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pixel_snap  # noqa: E402


class Transparent(unittest.TestCase):
    def test_a_fully_transparent_sprite_with_colors(self):
        r = pixel_snap.snap(np.zeros((16, 16, 4), dtype=np.uint8), colors=8, grid=pixel_snap.Grid(1, 1, 0, 0))
        self.assertEqual(r.image.shape, (16, 16, 4))
        self.assertFalse(r.image[..., 3].any())
        self.assertEqual(len(r.colors), 0)

    def test_a_fully_transparent_sprite_with_a_fixed_palette(self):
        pal = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        r = pixel_snap.snap(np.zeros((16, 16, 4), dtype=np.uint8), palette=pal, grid=pixel_snap.Grid(1, 1, 0, 0))
        self.assertFalse(r.image[..., 3].any())


class Photos(unittest.TestCase):
    """An external review (25.10): a photorealistic eye at 384x256 got a ~64 px grid. Real photos as the reviewer's
    class of input (tests/fixtures/README.md): 0.5.0 invented a grid on each."""

    def test_a_photo_has_no_grid(self):
        from PIL import Image
        here = os.path.join(os.path.dirname(__file__), "fixtures")
        for name in ("photo_astronaut.jpg", "photo_coffee.jpg", "photo_chelsea.jpg"):
            with self.subTest(name=name):
                g = pixel_snap.detect_grid(np.asarray(Image.open(os.path.join(here, name)).convert("RGB")))
                self.assertEqual((g.scale_x, g.scale_y), (1.0, 1.0), g.warning)
                self.assertIn("no pixel grid", g.warning)

    def test_a_near_miss_names_the_scale_flag(self):
        # a case a second review found: 16x16, 4 colours, 1-px noise, Lanczos to 128x128
        from PIL import Image
        rng = np.random.default_rng(1)
        spr = rng.integers(0, 256, (4, 3))[rng.integers(0, 4, (16, 16))].astype(np.uint8)
        big = np.asarray(Image.fromarray(spr).resize((128, 128), Image.LANCZOS))
        g = pixel_snap.detect_grid(big)
        self.assertEqual(g.scale_x, 1.0)                                       # it goes 1:1 (the known price) ...
        self.assertIn("--scale 8", g.warning)                                  # ... and names the way back

    def test_an_upscaled_sprite_still_has_its_grid(self):
        rng = np.random.default_rng(3)
        spr = rng.integers(0, 256, (6, 3))[rng.integers(0, 6, (24, 24))].astype(np.uint8)
        big = np.repeat(np.repeat(spr, 8, 0), 8, 1)
        g = pixel_snap.detect_grid(big)
        self.assertEqual((round(g.scale_x), round(g.scale_y)), (8, 8))


if __name__ == "__main__":
    unittest.main()
