"""Public import and distribution-name checks for PyGCP."""
from importlib import metadata, resources
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pygcp
from pygcp import core
from pygcp import test_pygcp


class PackageTests(unittest.TestCase):
    def test_bundled_example_data_match_source_tables(self):
        continuum, line = test_pygcp._load_example_data()
        source_dir = Path(__file__).resolve().parent
        for table, filename in ((continuum, "continuum"), (line, "line")):
            self.assertEqual(table.ndim, 2)
            self.assertEqual(table.shape[1], 3)
            self.assertGreaterEqual(table.shape[0], 5)
            self.assertTrue(np.isfinite(table).all())
            np.testing.assert_array_equal(table, np.loadtxt(source_dir / filename))

    def test_installation_test_defaults_do_not_write_or_change_fit_settings(self):
        result = {"example": "result"}
        with patch.object(test_pygcp, "run_mle", return_value=result) as fit, \
             patch.object(test_pygcp, "format_report", return_value="Fit report"), \
             redirect_stdout(io.StringIO()) as output:
            self.assertIs(test_pygcp.main([]), result)
        continuum, line = fit.call_args.args
        self.assertEqual(continuum.shape[1], 3)
        self.assertEqual(line.shape[1], 3)
        self.assertEqual(fit.call_args.kwargs, {
            "transfer": "tophat", "lag_min": -10.0, "lag_max": 10.0,
            "lag_step": 0.1, "run_scan": False, "compute_hesse_errors": False,
            "save": False, "make_figures": False, "show_figures": False,
            "output_prefix": "results/example",
        })
        self.assertIn("installation test passed", output.getvalue())

    def test_installation_test_accepts_explicit_save_and_plot_prefix(self):
        with patch.object(test_pygcp, "run_mle", return_value={}) as fit, \
             patch.object(test_pygcp, "format_report", return_value="Fit report"), \
             redirect_stdout(io.StringIO()):
            test_pygcp.main(["--save", "--plot", "--output-prefix", "custom/run1"])
        self.assertTrue(fit.call_args.kwargs["save"])
        self.assertTrue(fit.call_args.kwargs["make_figures"])
        self.assertEqual(fit.call_args.kwargs["output_prefix"], "custom/run1")

    def test_installation_test_module_help_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-m", "pygcp.test_pygcp", "--help"],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output-prefix", completed.stdout)
        self.assertIn("--save", completed.stdout)

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
