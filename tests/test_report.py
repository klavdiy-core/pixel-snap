"""The re-grade (7.5/10): what the report says when the user is about to be misled. Each test is red on 0.4.0."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pixel_snap  # noqa: E402
from test_pixel_snap import make_sprite  # noqa: E402


def cli(img, *args):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.png")
        Image.fromarray(img).save(p)
        out = io.StringIO()
        with redirect_stdout(out):
            pixel_snap.main([p, "--json", *args])
        return json.loads(out.getvalue())


def shifted(scale=6, off=(2, 4)):
    big = np.repeat(np.repeat(make_sprite(seed=8, size=20), scale, axis=0), scale, axis=1)
    return np.pad(big, ((off[1], 0), (off[0], 0), (0, 0)), mode="edge")


class Report(unittest.TestCase):
    def test_a_forced_scale_that_mixes_many_cells_is_warned(self):
        rep = cli(shifted(), "--scale", "6")                     # right size, wrong offset (0, 0)
        self.assertGreater(rep["mixed_percent"], 20)
        self.assertIn("--scale", rep["warning"])
        self.assertIn("offset", rep["warning"])
        rep = cli(shifted(), "--scale", "6", "--offset", "2", "4")
        self.assertEqual(rep["warning"], "")

    def test_no_grid_at_all_is_said(self):
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)  # every pixel its own cell: nothing upscaled
        rep = cli(noise)
        self.assertEqual(rep["pixel_size"], [1, 1])
        self.assertIn("no grid", rep["warning"])

    def test_a_missing_file_is_one_clear_line_and_exit_2(self):
        err = io.StringIO()
        with redirect_stderr(err):
            code = pixel_snap.main([os.path.join(tempfile.gettempdir(), "no_such_image_qq.png")])
        self.assertEqual(code, 2)
        self.assertIn("cannot open", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_the_offset_is_printed_inside_one_pixel(self):
        rep = cli(shifted(scale=6, off=(2, 4)), "--scale", "6", "--offset", "8", "16")   # 8 = 6 + 2, 16 = 12 + 4
        self.assertEqual(rep["offset"], [2, 4])
        self.assertEqual(rep["warning"], "")                       # the same grid as offset 2 4: nothing mixed

    def test_a_non_square_pixel_says_its_aspect(self):
        rep = cli(np.repeat(np.repeat(make_sprite(seed=3), 9, axis=0), 5, axis=1))
        self.assertIn("aspect 1:1.8", rep["warning"])


if __name__ == "__main__":
    unittest.main()
