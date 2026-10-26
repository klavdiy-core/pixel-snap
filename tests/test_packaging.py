"""The package and the script must say the same version, and pip must find the command the README names."""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pixel_snap  # noqa: E402


class Packaging(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
            self.toml = f.read()

    def test_versions_agree(self):
        m = re.search(r'^version\s*=\s*"([^"]+)"', self.toml, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), pixel_snap.__version__)

    def test_command_points_at_main(self):
        self.assertRegex(self.toml, r'pixel-snap\s*=\s*"pixel_snap:main"')
        self.assertTrue(callable(pixel_snap.main))


if __name__ == "__main__":
    unittest.main()
