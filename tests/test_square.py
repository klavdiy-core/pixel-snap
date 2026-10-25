"""The grader (5/10): when the axes disagree the result must be square and say so loudly; a --scale-hint narrows the
search. Each test is red on 0.3.0."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pixel_snap  # noqa: E402
from test_pixel_snap import make_sprite  # noqa: E402


class Square(unittest.TestCase):
    def test_a_slight_disagreement_becomes_square_with_a_warning(self):
        big = np.repeat(np.repeat(make_sprite(seed=4, size=24), 6, axis=0), 6, axis=1)
        tall = np.asarray(Image.fromarray(big).resize((big.shape[1], int(big.shape[0] * 1.07)), Image.NEAREST))
        g = pixel_snap.detect_grid(tall)
        self.assertAlmostEqual(g.scale_x, g.scale_y, places=6)
        self.assertIn("square", g.warning)
        g2 = pixel_snap.detect_grid(tall, square=False)                  # --nonsquare keeps what was found
        self.assertGreater(abs(g2.scale_x - g2.scale_y), 0.2)

    def test_an_axis_with_no_edges_takes_the_other_axis_period(self):
        stripes = np.repeat(np.repeat(make_sprite(seed=5, size=24)[:1], 96, axis=0), 4, axis=1)   # columns only
        g = pixel_snap.detect_grid(stripes)
        self.assertEqual((round(g.scale_x), round(g.scale_y)), (4, 4))
        self.assertIn("no grid", g.warning)

    def test_a_large_disagreement_is_kept_and_said(self):
        big = np.repeat(np.repeat(make_sprite(seed=3), 9, axis=0), 5, axis=1)
        g = pixel_snap.detect_grid(big)
        self.assertEqual((round(g.scale_x), round(g.scale_y)), (5, 9))
        self.assertIn("not square", g.warning)

    def test_matching_axes_say_nothing(self):
        big = np.repeat(np.repeat(make_sprite(seed=6), 7, axis=0), 7, axis=1)
        self.assertEqual(pixel_snap.detect_grid(big).warning, "")


class Hint(unittest.TestCase):
    def test_a_hint_narrows_the_search_to_its_neighbourhood(self):
        big = np.repeat(np.repeat(make_sprite(seed=7, size=24), 6, axis=0), 6, axis=1)
        self.assertEqual(round(pixel_snap.detect_grid(big).scale_x), 6)
        self.assertEqual(round(pixel_snap.detect_grid(big, hint=3).scale_x), 3)      # a divisor, when asked for

    def cli(self, img, *args):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "in.png")
            Image.fromarray(img).save(p)
            out = io.StringIO()
            with redirect_stdout(out):
                pixel_snap.main([p, "--json", *args])
            return json.loads(out.getvalue())

    def test_the_cli_reports_the_warning_and_takes_the_hint(self):
        rep = self.cli(np.repeat(np.repeat(make_sprite(seed=3), 9, axis=0), 5, axis=1))
        self.assertIn("not square", rep["warning"])
        rep = self.cli(np.repeat(np.repeat(make_sprite(seed=7, size=24), 6, axis=0), 6, axis=1), "--scale-hint", "3")
        self.assertEqual([round(v) for v in rep["pixel_size"]], [3, 3])
        self.assertEqual(rep["warning"], "")


if __name__ == "__main__":
    unittest.main()
