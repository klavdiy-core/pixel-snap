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


if __name__ == "__main__":
    unittest.main()
