"""Public import and distribution-name checks for PyGCP."""
from importlib import metadata, resources
import unittest

import pygcp
from pygcp import core


class PackageTests(unittest.TestCase):
    def test_public_api_comes_from_pygcp_core(self):
        self.assertEqual(pygcp.__name__, "pygcp")
        for name in pygcp.__all__:
            if name != "__version__":
                self.assertIs(getattr(pygcp, name), getattr(core, name))
        self.assertEqual(pygcp.run_mle.__module__, "pygcp.core")

    def test_typing_marker_is_included(self):
        self.assertTrue(resources.files(pygcp).joinpath("py.typed").is_file())

    def test_distribution_is_pygcp_rm_but_import_is_pygcp(self):
        try:
            distribution = metadata.distribution("pygcp-rm")
        except metadata.PackageNotFoundError:
            self.skipTest("Install pygcp-rm to check distribution metadata")
        self.assertEqual(distribution.metadata["Name"], "pygcp-rm")
        self.assertEqual(distribution.version, pygcp.__version__)
        self.assertEqual(distribution.read_text("top_level.txt").split(), ["pygcp"])


if __name__ == "__main__":
    unittest.main()
