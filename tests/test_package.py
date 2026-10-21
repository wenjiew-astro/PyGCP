"""Public import and distribution-name checks for PyGCP."""
from importlib import metadata, resources
import copy
import unittest

import pygcp
from pygcp import core


class PackageTests(unittest.TestCase):
    def test_report_rho_uses_six_decimal_places_without_changing_result(self):
        for rho, expected in ((0.9999938558, "0.999994"), (0.8, "0.800000"),
                              (0.00033535, "0.000335")):
            with self.subTest(rho=rho):
                result = {
                    "p_hat": {"lag": 2.5, "tau_drw": 20.0, "rho": rho,
                              "transfer": "tophat", "transfer_width": 0.6},
                    "summary": {},
                    "config": {"scan_mode": "profile-local", "fit_transfer_width": True},
                    "joint_loglike": 10.0,
                    "joint_objective": 10.0,
                    "optimizer_diagnostics": {"converged": True},
                    "fit_score_kind": "data_loglike",
                }
                original = copy.deepcopy(result)
                line = next(line for line in pygcp.format_report(result).splitlines()
                            if line.startswith("joint fit:"))
                self.assertEqual(line, "joint fit: lag_centroid=2.5000, tau_drw=20.0000, "
                                       f"rho={expected}")
                self.assertEqual(result, original)

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
