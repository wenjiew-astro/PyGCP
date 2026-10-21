"""Numerical regressions for the causal three-Gaussian response.

Run with ``python -m unittest discover -s tests -p 'test_flexible_numerics.py'``.
The independent references use scipy.integrate.quad, not a dense midpoint grid.
"""

import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, logsumexp

from pygcp import core
from pygcp import _flexible


TC = np.array([0.0, 0.7, 3.2, 8.0, 13.0])
TL = np.array([0.2, 2.0, 6.5, 9.0])
INITIAL_WEIGHTS = np.array([0.25, 0.50, 0.25])
N169_LAG = 4.020054671533811
N169_WIDTH = 4.539353763093351
N169_WEIGHTS = np.array([0.7756053625817966, 0.08457254462760921, 0.13982209279059432])


def integrate_with_breaks(function, breaks):
    boundaries = sorted(set(float(value) for value in breaks if math.isfinite(value)))
    return sum(quad(function, left, right, epsabs=1e-11, epsrel=1e-10, limit=150)[0]
               for left, right in zip(boundaries, boundaries[1:]))


def independent_covariance(t_c, t_l, tau, rho, metadata):
    """Integrate the response and its difference density with independent QUADPACK.

    The product-of-two-Gaussians identity analytically integrates the inner
    variable of the difference density. No package integration helpers are used.
    """
    centers = np.asarray(metadata["component_centers"])
    sigma = float(metadata["component_sigma"])
    mixture = np.asarray(metadata["mixture_weights"])
    causal = bool(metadata["enforce_nonnegative"])
    log_mass = float(logsumexp(np.log(mixture) + log_ndtr(centers / sigma))) if causal else 0.0
    log_norm = math.log(sigma * math.sqrt(2 * math.pi))

    def density(q):
        if causal and q < 0:
            return 0.0
        return float(np.exp(logsumexp(np.log(mixture) - log_norm
                         - 0.5 * ((q - centers) / sigma) ** 2) - log_mass))

    def difference_density(v):
        u = (v - (centers[:, None] - centers[None, :])) / (math.sqrt(2) * sigma)
        terms = (np.log(mixture)[:, None] + np.log(mixture)[None, :]
                 - math.log(2 * math.sqrt(math.pi) * sigma) - 0.5 * u * u)
        if causal:
            terms = terms + log_ndtr((centers[:, None] + centers[None, :] - abs(v))
                                    / (math.sqrt(2) * sigma)) - 2 * log_mass
        return float(np.exp(logsumexp(terms)))

    # Finite 14-sigma tails are below the requested accuracy in these tests.
    low = 0.0 if causal else float(min(centers) - 14 * sigma)
    high = max(float(max(centers) + 14 * sigma), low + 20 * sigma)
    response_breaks = [low, high] + [float(c + step * sigma) for c in centers
                                    for step in (-8, -4, 0, 4, 8)
                                    if low < c + step * sigma < high]
    diff_extent = max(abs(high), abs(low)) * 2 + 14 * sigma
    diff_breaks = [-diff_extent, 0.0, diff_extent]
    diff_breaks += [float(c1 - c2 + step * sigma) for c1 in centers for c2 in centers
                    for step in (-8, -4, 0, 4, 8)]

    def raw_auto(delta):
        return integrate_with_breaks(
            lambda v: difference_density(v) * math.exp(-abs(delta - v) / tau),
            diff_breaks + [delta],
        )

    variance = raw_auto(0.0)
    cross = np.empty((len(t_c), len(t_l)))
    for i, tc in enumerate(t_c):
        for j, tl in enumerate(t_l):
            delta = float(tl - tc)
            cross[i, j] = rho * integrate_with_breaks(
                lambda q: density(q) * math.exp(-abs(q - delta) / tau),
                response_breaks + ([delta] if low < delta < high else []),
            ) / math.sqrt(variance)
    auto = np.array([[raw_auto(float(ti - tj)) / variance for tj in t_l] for ti in t_l])
    return cross, auto


class FlexibleNumericalRegression(unittest.TestCase):
    def test_previously_failing_centroid_parameters(self):
        cases = [(2.0, width) for width in (0.05, 0.10, 0.20, 1.0, 5.0, 20.0)]
        cases += [(3.0, 0.05), (4.0, 0.05)]
        for lag, width in cases:
            with self.subTest(lag=lag, width=width):
                roots = _flexible.center_roots_for_lag(lag, width, INITIAL_WEIGHTS)
                self.assertGreaterEqual(len(roots), 1)
                for center in roots:
                    metadata = _flexible.metadata_from_center(center, width, INITIAL_WEIGHTS)
                    self.assertAlmostEqual(metadata["centroid"], lag, delta=max(1e-9, 1e-9 * lag))
                    q, weights = _flexible.quadrature_from_metadata(metadata, grid_size=61)
                    self.assertTrue(np.all(q >= 0.0))
                    self.assertTrue(np.all(weights >= 0.0))
                    self.assertTrue(np.all(np.isfinite(weights)))
                    self.assertAlmostEqual(float(sum(weights)), 1.0, places=12)
                    self.assertAlmostEqual(float(q @ weights), lag, delta=max(2e-6, 2e-6 * lag))
                    matrix = core.build_reverberation_covariance(
                        TC, TL, lag, 20.0, 0.8, transfer="flexible_gaussian3",
                        transfer_width=width, flexible_weights=INITIAL_WEIGHTS,
                        enforce_nonnegative_transfer=True, flexible_center=center,
                    )
                    self.assertIsNotNone(matrix)
                    self.assertTrue(np.all(np.isfinite(matrix)))

    def test_multiple_roots_are_exposed_and_do_not_depend_on_grid(self):
        roots = _flexible.center_roots_for_lag(N169_LAG, N169_WIDTH, N169_WEIGHTS)
        self.assertEqual(len(roots), 3)
        np.testing.assert_array_less(roots[:-1], roots[1:])
        with self.assertRaises(_flexible.AmbiguousCentroidError):
            core.make_transfer_quadrature(
                "flexible_gaussian3", N169_LAG, N169_WIDTH,
                flexible_weights=N169_WEIGHTS, enforce_nonnegative=True,
            )
        with self.assertRaises(_flexible.AmbiguousCentroidError):
            core.build_reverberation_covariance(
                TC, TL, N169_LAG, 20.0, 0.8, transfer="flexible_gaussian3",
                transfer_width=N169_WIDTH, flexible_weights=N169_WEIGHTS,
                enforce_nonnegative_transfer=True,
            )
        for center in roots:
            metadata = _flexible.metadata_from_center(center, N169_WIDTH, N169_WEIGHTS)
            self.assertAlmostEqual(metadata["centroid"], N169_LAG, delta=1e-8)
            for grid in (61, 121, 241):
                _, _, actual = core._flexible_gaussian3_quadrature(
                    N169_LAG, N169_WIDTH, N169_WEIGHTS,
                    grid_size=grid, enforce_nonnegative=True, flexible_center=center,
                )
                assert_allclose(actual["component_centers"], metadata["component_centers"], atol=1e-12)
        matrices = [core.build_reverberation_covariance(
            TC, TL, N169_LAG, 23.373441749608023, 0.9944086787354051,
            transfer="flexible_gaussian3", transfer_width=N169_WIDTH,
            enforce_nonnegative_transfer=True, flexible_weights=N169_WEIGHTS,
            flexible_center=center,
        ) for center in roots]
        self.assertGreater(np.max(np.abs(matrices[0] - matrices[-1])), 1e-4)

    def test_explicit_common_center_is_continuous_on_a_branch(self):
        roots = _flexible.center_roots_for_lag(N169_LAG, N169_WIDTH, N169_WEIGHTS)
        for center in roots:
            nearby = [_flexible.metadata_from_center(center + shift, N169_WIDTH, N169_WEIGHTS)
                      for shift in (-1e-5, 0.0, 1e-5)]
            means = [item["centroid"] for item in nearby]
            self.assertLess(max(means) - min(means), 1e-3)
            matrices = []
            for metadata in nearby:
                cross, auto = _flexible.ou_covariance_from_metadata(TC, TL, 20.0, 0.8, metadata)
                matrices.append(np.block([[core.ou_corr(TC, TC, 20.0), cross], [cross.T, auto]]))
            self.assertLess(np.max(np.abs(matrices[-1] - matrices[0])), 1e-4)

    def test_near_tangent_centroid_roots_are_not_lost(self):
        for bounds, sign, expected_counts in (
            ((1.2, 1.9), -1.0, (3, 2, 1)),
            ((4.0, 4.5), 1.0, (1, 2, 3)),
        ):
            with self.subTest(bounds=bounds):
                optimum = minimize_scalar(
                    lambda center: sign * _flexible.metadata_from_center(
                        center, N169_WIDTH, N169_WEIGHTS
                    )["centroid"],
                    bounds=bounds, method="bounded", options={"xatol": 1e-13},
                )
                target = _flexible.metadata_from_center(
                    optimum.x, N169_WIDTH, N169_WEIGHTS
                )["centroid"]
                for delta, count in zip((-1e-8, 0.0, 1e-8), expected_counts):
                    roots = _flexible.center_roots_for_lag(
                        target + delta, N169_WIDTH, N169_WEIGHTS
                    )
                    self.assertEqual(len(roots), count)
                    for center in roots:
                        centroid = _flexible.metadata_from_center(
                            center, N169_WIDTH, N169_WEIGHTS
                        )["centroid"]
                        self.assertAlmostEqual(centroid, target + delta, delta=1e-9)

    def test_near_cusp_two_stationary_points_in_one_cell_are_resolved(self):
        first = 0.6571050945940982
        ratio = N169_WEIGHTS[1] / (N169_WEIGHTS[1] + N169_WEIGHTS[2])
        mixture = np.array([first, (1.0 - first) * ratio, (1.0 - first) * (1.0 - ratio)])
        lag = 0.9976909807366909
        roots = _flexible.center_roots_for_lag(lag, 1.0, mixture)
        self.assertEqual(len(roots), 3)
        assert_allclose(roots, [0.73493440, 0.73988427, 0.74483568], atol=1e-7, rtol=1e-7)
        for center in roots:
            self.assertAlmostEqual(_flexible.metadata_from_center(center, 1.0, mixture)["centroid"],
                                   lag, delta=1e-11)

    def test_unresolvable_machine_precision_cusp_is_explicit_not_spurious_roots(self):
        critical = 0.6570950945940982
        ratio = N169_WEIGHTS[1] / (N169_WEIGHTS[1] + N169_WEIGHTS[2])

        def mixture(first):
            return np.array([first, (1.0 - first) * ratio, (1.0 - first) * (1.0 - ratio)])

        with self.assertRaises(_flexible.NumericalConvergenceError):
            _flexible.center_roots_for_lag(
                0.9977022953942492, 1.0, mixture(critical + 1e-10)
            )
        for first, lag, expected in (
            (critical - 1e-10, 0.9977022956205439, 1),
            (critical + 1e-8, 0.9977022841926619, 3),
        ):
            roots = _flexible.center_roots_for_lag(lag, 1.0, mixture(first))
            self.assertEqual(len(roots), expected)
            for center in roots:
                self.assertAlmostEqual(_flexible.metadata_from_center(center, 1.0, mixture(first))["centroid"],
                                       lag, delta=1e-10)

    def test_covariance_psd_and_independent_integral(self):
        for center, width, mixture, causal in (
            (2.0, 0.2, INITIAL_WEIGHTS, True),
            (-1.0, 5.0, INITIAL_WEIGHTS, True),
            (1.7, 1.0, np.array([0.7, 0.1, 0.2]), False),
        ):
            with self.subTest(center=center, width=width, causal=causal):
                metadata = _flexible.metadata_from_center(center, width, mixture,
                                                         enforce_nonnegative=causal)
                actual_cross, actual_auto = _flexible.ou_covariance_from_metadata(
                    TC, TL, 20.0, 0.8, metadata
                )
                reference_cross, reference_auto = independent_covariance(TC[:2], TL[:2], 20.0, 0.8, metadata)
                assert_allclose(actual_cross[:2, :2], reference_cross, atol=2e-7, rtol=2e-7)
                assert_allclose(actual_auto[:2, :2], reference_auto, atol=2e-7, rtol=2e-7)
                matrix = np.block([[core.ou_corr(TC, TC, 20.0), actual_cross],
                                   [actual_cross.T, actual_auto]])
                assert_allclose(matrix, matrix.T, atol=1e-12)
                assert_allclose(np.diag(matrix), 1.0, atol=1e-10)
                self.assertGreater(np.linalg.eigvalsh(matrix).min(), -2e-8)

    def test_accuracy_tightening_not_a_961_point_truth(self):
        for width in (0.05, 0.2, 1.0, 5.0, 20.0):
            with self.subTest(width=width):
                center = _flexible.center_roots_for_lag(2.0, width, INITIAL_WEIGHTS)[0]
                metadata = _flexible.metadata_from_center(center, width, INITIAL_WEIGHTS)
                default = _flexible.ou_covariance_from_metadata(TC, TL, 20.0, 0.8, metadata)
                tight = _flexible.ou_covariance_from_metadata(
                    TC, TL, 20.0, 0.8, metadata, rtol=1e-11, atol=1e-13
                )
                for actual, reference in zip(default, tight):
                    assert_allclose(actual, reference, rtol=1e-7, atol=1e-8)

    def test_top_hat_analytic_path_unchanged(self):
        matrix = core.build_reverberation_covariance(
            TC, TL, lag=2.0, tau_drw=20.0, rho=0.8,
            transfer="tophat", transfer_width=1.3, enforce_nonnegative_transfer=True,
        )
        assert_allclose(matrix[:2, 5:7],
                        [[0.7391956134442017, 0.7956668441051575],
                         [0.7137712880155557, 0.7809901877937659]], atol=2e-14, rtol=2e-14)
        self.assertAlmostEqual(matrix[5, 6], 0.9341686560709693, delta=2e-14)

    def test_original_model_hyperparameters_and_penalty_unchanged(self):
        config = core.build_config(transfer="flexible_gaussian3")
        self.assertEqual(config["flexible_sigma_ratio"], 0.45)
        self.assertEqual(config["flexible_smoothness"], 0.25)
        self.assertEqual(config["transfer_width_bounds"], (0.02, 20.0))
        for function in (core.build_config, core.run_mle):
            signature = inspect.signature(function)
            self.assertEqual(signature.parameters["flexible_sigma_ratio"].default, 0.45)
            self.assertEqual(signature.parameters["flexible_smoothness"].default, 0.25)
        self.assertEqual(len(core._parameter_names(config)), 6)
        parameters = {"lag": 2.0, "transfer_weights": N169_WEIGHTS, "tau_drw": 20.0,
                      "rho": 0.8, "transfer_width": N169_WIDTH}
        with patch.object(core, "_data_loglike_for_params", return_value=12.0), \
                patch.object(core, "log_prior_params", return_value=0.0):
            actual = core._objective_value(parameters, None, None, None, None, config)
        self.assertAlmostEqual(actual, 12.0 - 0.25 * float(sum(np.diff(N169_WEIGHTS) ** 2)), places=13)

    def test_joint_coordinates_and_profile_preserve_response_branches(self):
        config = core.build_config(transfer="flexible_gaussian3", lag_min=0.1, lag_max=10.0)
        config["total_duration"] = 20.0
        self.assertEqual(core._parameter_names(config)[0], "transfer_center")
        base = {"lag": N169_LAG, "tau_drw": 20.0, "rho": 0.8,
                "transfer_width": N169_WIDTH, "transfer_weights": N169_WEIGHTS,
                "transfer": "flexible_gaussian3", "jitter": config["jitter"]}
        candidates = core._flexible_params_at_lag(base, N169_LAG, config)
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            starts = core._params_to_joint_starts(candidate, config)
            self.assertEqual(len(starts), 1)
            reconstructed = core._x_to_params(starts[0], config)
            assert_allclose(reconstructed["transfer_component_centers"],
                            candidate["transfer_component_centers"], atol=1e-12)
            self.assertAlmostEqual(reconstructed["lag"], N169_LAG, delta=1e-8)
        nuisance = core._params_to_nuisance_x(base, config)
        with self.assertRaises(ValueError):
            core._x_to_params(nuisance, config, fixed_lag=N169_LAG)
        all_branches = core._x_to_params(nuisance, config, fixed_lag=N169_LAG,
                                        all_flexible_branches=True)
        self.assertEqual(len(all_branches), 3)

        # The fixed-lag envelope must compare the objective on every branch,
        # rather than use whichever root was found first.
        target_center = candidates[1]["transfer_component_centers"][1]

        def mock_objective(params, *unused):
            return -float(params["transfer_component_centers"][1] - target_center) ** 2

        def mock_minuit(objective, start, bounds, names, max_calls, cfg):
            objective(start)
            return SimpleNamespace(valid=True, values=dict(zip(names, start)))

        with patch.object(core, "_objective_value", side_effect=mock_objective), \
                patch.object(core, "_run_minuit", side_effect=mock_minuit):
            selected, score = core.optimize_nuisance_local(
                N169_LAG, [nuisance], None, None, TC, TL, config
            )
        self.assertAlmostEqual(selected["transfer_component_centers"][1], target_center, delta=1e-10)
        self.assertAlmostEqual(score, 0.0, places=13)

    def test_lag_bounds_apply_to_continuous_centroid_not_common_center(self):
        config = core.build_config(transfer="flexible_gaussian3", lag_min=0.1, lag_max=10.0)
        metadata = _flexible.metadata_from_center(-2.0, 20.0, INITIAL_WEIGHTS)
        params = {"lag": metadata["centroid"], "transfer_weights": INITIAL_WEIGHTS,
                  "tau_drw": 20.0, "rho": 0.8, "transfer_width": 20.0}
        with patch.object(core, "_data_loglike_for_params", return_value=12.0), \
                patch.object(core, "log_prior_params", return_value=0.0):
            self.assertEqual(core._objective_value(dict(params, lag=10.1), None, None, None, None, config), -np.inf)
            self.assertEqual(core._objective_value(dict(params, lag=0.05), None, None, None, None, config), -np.inf)
            self.assertTrue(np.isfinite(core._objective_value(dict(params, lag=2.0), None, None, None, None, config)))

    def test_fixed_scan_rebuilds_center_for_causal_and_untruncated_models(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                config = core.build_config(
                    transfer="flexible_gaussian3", enforce_nonnegative_transfer=causal,
                    lag_min=-2.0 if not causal else 0.1, lag_max=10.0,
                )
                width = N169_WIDTH if causal else 1.0
                mixture = N169_WEIGHTS if causal else np.array([0.7, 0.1, 0.2])
                initial_lag = N169_LAG if causal else 0.5
                base = {"lag": initial_lag, "tau_drw": 20.0, "rho": 0.8,
                        "transfer_width": width, "transfer_weights": mixture,
                        "transfer": "flexible_gaussian3", "jitter": config["jitter"]}
                p_hat = core._flexible_params_at_lag(base, initial_lag, config)[0]
                lag_grid = np.array([N169_LAG - .005, N169_LAG, N169_LAG + .005]) if causal else np.array([-1.0, 0.0, 1.0])

                def objective(params, *unused):
                    center = params["transfer_component_centers"][1]
                    metadata = _flexible.metadata_from_center(
                        center, width, mixture, enforce_nonnegative=causal
                    )
                    self.assertAlmostEqual(metadata["centroid"], params["lag"], delta=1e-8)
                    return float(center)

                with patch.object(core, "_objective_value", side_effect=objective):
                    scores, density, parameters = core.scan_lag_fixed(
                        None, None, TC, TL, lag_grid, p_hat, config
                    )
                self.assertTrue(np.all(np.isfinite(scores)))
                self.assertTrue(np.all(np.isfinite(density)))
                for lag, params in zip(lag_grid, parameters):
                    expected = _flexible.center_roots_for_lag(
                        lag, width, mixture, enforce_nonnegative=causal
                    )[-1]
                    self.assertAlmostEqual(params["transfer_component_centers"][1], expected, delta=1e-9)

    def test_nonuniform_nodes_cannot_use_uniform_grid_shortcut(self):
        with self.assertRaises(ValueError):
            core._weighted_ou_convolution_covariance(
                TC, TL, 2.0, 20.0, 0.8,
                np.array([0.0, 0.1, 1.0]), np.array([0.25, 0.5, 0.25]),
            )


if __name__ == "__main__":
    unittest.main()
