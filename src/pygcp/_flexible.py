"""Continuous three-Gaussian responses and error-controlled OU integrals.

The Gaussian centres, not a discretized centroid equation, define a response.
Quadrature resolution is a numerical accuracy setting: it never changes the
continuous response or selects a different solution of the centroid equation.
"""

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfcx, log_ndtr, logsumexp, ndtri_exp


class NumericalConvergenceError(RuntimeError):
    """A mathematically valid response could not be integrated accurately."""


class AmbiguousCentroidError(ValueError):
    """More than one continuous response has the requested centroid."""


_LOG_2PI = float(np.log(2.0 * np.pi))
_SQRT2 = float(np.sqrt(2.0))
_BASIS = np.array([-1.0, 0.0, 1.0])


def _validated(width, mixture_weights, sigma_ratio):
    width, sigma_ratio = float(width), float(sigma_ratio)
    weights = np.asarray(mixture_weights, dtype=float)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("flexible transfer width must be finite and positive")
    if not np.isfinite(sigma_ratio) or sigma_ratio <= 0.0:
        raise ValueError("flexible_sigma_ratio must be finite and positive")
    if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("flexible transfer needs three finite positive weights")
    weights = weights / np.max(weights)
    weights = weights / np.sum(weights)
    return width, weights, sigma_ratio


def _positive_normal_mean(alpha):
    """E[Z+alpha | Z+alpha >= 0], without negative-tail cancellation."""
    alpha = np.asarray(alpha, dtype=float)
    answer = np.empty_like(alpha)
    ordinary = alpha >= -30.0
    a = alpha[ordinary]
    mills = np.empty_like(a)
    negative = a < 0
    mills[negative] = np.sqrt(2.0 / np.pi) / erfcx(-a[negative] / _SQRT2)
    mills[~negative] = np.exp(-0.5 * a[~negative] ** 2 - 0.5 * _LOG_2PI
                               - log_ndtr(a[~negative]))
    answer[ordinary] = a + mills
    # Inverse Mills ratio minus x.  At x >= 30 this expansion avoids losing
    # every significant digit when the centre is far below the boundary.
    x = -alpha[~ordinary]
    y = 1.0 / (x * x)
    answer[~ordinary] = (1.0 / x) * (1.0 + y * (-2.0 + y * (10.0 + y *
        (-74.0 + y * (706.0 + y * (-8162.0 + y * (110410.0 + y *
        (-1708394.0 + y * 29752066.0))))))))
    return answer


def metadata_from_center(common_center, width, mixture_weights, sigma_ratio=0.45,
                         enforce_nonnegative=True):
    """Describe one continuous, normalized response with middle centre ``c``.

    ``mixture_weights`` are the *untruncated* basis weights.  ``effective_weights``
    include the different surviving masses of the three components.
    """
    width, weights, sigma_ratio = _validated(width, mixture_weights, sigma_ratio)
    center = float(common_center)
    if not np.isfinite(center):
        raise ValueError("flexible common centre must be finite")
    sigma = width * sigma_ratio
    centers = center + width * _BASIS
    if enforce_nonnegative:
        log_masses = log_ndtr(centers / sigma)
        log_mass = float(logsumexp(np.log(weights) + log_masses))
        effective = np.exp(np.log(weights) + log_masses - log_mass)
        means = sigma * _positive_normal_mean(centers / sigma)
    else:
        log_masses = np.zeros(3)
        log_mass = 0.0
        effective = weights.copy()
        means = centers.copy()
    centroid = float(np.dot(effective, means))
    return {
        "centroid": centroid,
        "lag": centroid,
        "common_center": center,
        "component_centers": centers,
        "component_sigma": sigma,
        "mixture_weights": weights,
        "effective_weights": effective,
        "component_means": means,
        "component_log_masses": log_masses,
        "log_mass": log_mass,
        "width": width,
        "sigma_ratio": sigma_ratio,
        "enforce_nonnegative": bool(enforce_nonnegative),
        "common_center_shift": float(np.dot(weights, centers) - centroid),
    }


def center_bounds(lag_min, lag_max, width_max, sigma_ratio=0.45):
    """Conservative common-centre bounds for a constrained centroid interval.

    Zero is only an unattained limiting centroid of a positive-width causal
    Gaussian response; its centre has no finite lower bound.
    """
    lo, hi, width = float(lag_min), float(lag_max), float(width_max)
    if not np.isfinite(hi) or not np.isfinite(width) or width <= 0:
        raise ValueError("finite upper lag and positive maximum width required")
    lower = None if lo <= 0 else -(float(sigma_ratio) * width) ** 2 / lo - width
    return lower, hi + width


def _center_roots_for_lag_uncached(lag, width, mixture_weights, sigma_ratio=0.45,
                                  enforce_nonnegative=True):
    """Return all resolved middle-centre roots, without assuming monotonicity.

    Search in units of the spacing.  Dense local grids cover each component's
    boundary crossing; separate tail grids avoid an enormous uniform bracket
    for a narrow response located far from zero.  Tangent roots are handled by
    extrema of the continuous centroid curve, as well as sign-changing roots.
    """
    width, weights, ratio = _validated(width, mixture_weights, sigma_ratio)
    lag = float(lag)
    if not np.isfinite(lag):
        raise ValueError("flexible lag must be finite")
    if not enforce_nonnegative:
        return np.array([lag - width * np.dot(weights, _BASIS)])
    if lag <= 0:
        raise ValueError("a positive-width non-negative response requires lag>0")
    target = lag / width

    def value_and_derivative(x):
        centers = np.asarray(x)[..., None] + _BASIS
        alpha = centers / ratio
        lm = log_ndtr(alpha)
        total = logsumexp(np.log(weights) + lm, axis=-1)
        effective = np.exp(np.log(weights) + lm - total[..., None])
        centroid = np.sum(effective * ratio * _positive_normal_mean(alpha), axis=-1)
        log_boundary = logsumexp(np.log(weights) - 0.5 * alpha**2
                                - np.log(ratio) - 0.5 * _LOG_2PI, axis=-1)
        # For a translated density truncated at zero, M'=1-M*f(0)/Z.
        derivative = -np.expm1(np.log(centroid) + log_boundary - total)
        return centroid - target, derivative

    def equation(x):
        return float(value_and_derivative(x)[0])

    def derivative(x):
        return float(value_and_derivative(x)[1])

    extent = max(12.0 * ratio, 1.0)
    lower = min(-1.0 - extent, -2.0 * ratio * ratio / target - 1.0)
    upper = max(1.0 + extent, target + 2.0 + extent)
    while equation(lower) >= 0:
        lower = 2.0 * lower - 1.0
    while equation(upper) <= 0:
        upper = 2.0 * upper + 1.0
    grids = [np.linspace(-1.0 - extent, 1.0 + extent, 193),
             np.linspace(target - 2.0 - extent, target + 2.0 + extent, 97)]
    grids += [b + ratio * np.linspace(-12.0, 12.0, 193) for b in _BASIS]
    # Negative-tail refinement is logarithmic in distance from the boundary.
    grids += [-np.geomspace(max(1.0, extent), max(-lower, 1.0, extent), 65),
              np.array([lower, upper])]
    x = np.unique(np.concatenate(grids))
    x = x[(x >= lower) & (x <= upper)]
    slopes = value_and_derivative(x)[1]
    stationary = []
    for k in range(len(x) - 1):
        if slopes[k] == 0.0:
            stationary.append(x[k])
        elif slopes[k] * slopes[k + 1] < 0:
            stationary.append(brentq(derivative, x[k], x[k + 1],
                                      xtol=2e-13, rtol=2e-14))
    # At a cusp a negative derivative dip can be narrower than *any* fixed
    # sampling grid: both stationary points then lie in one same-sign cell.
    # Refine sampled extrema of M' itself, not extrema of the target residual.
    # The latter can appear monotone at every sampled point and miss 3 roots.
    for k in range(1, len(x) - 1):
        local_min = (slopes[k] <= slopes[k - 1] and slopes[k] <= slopes[k + 1]
                     and slopes[k] >= 0.0
                     and (slopes[k] < slopes[k - 1] or slopes[k] < slopes[k + 1]))
        local_max = (slopes[k] >= slopes[k - 1] and slopes[k] >= slopes[k + 1]
                     and slopes[k] <= 0.0
                     and (slopes[k] > slopes[k - 1] or slopes[k] > slopes[k + 1]))
        if not (local_min or local_max):
            continue
        sign = 1.0 if local_min else -1.0
        optimum = minimize_scalar(lambda z: sign * derivative(z),
                                  bounds=(x[k - 1], x[k + 1]), method="bounded",
                                  options={"xatol": 2e-13})
        middle = float(optimum.x)
        middle_slope = derivative(middle)
        for edge, edge_slope in ((x[k - 1], slopes[k - 1]),
                                  (x[k + 1], slopes[k + 1])):
            if edge_slope * middle_slope < 0.0:
                stationary.append(brentq(derivative, min(edge, middle),
                                          max(edge, middle), xtol=2e-13, rtol=2e-14))
        if middle_slope == 0.0:
            stationary.append(middle)
    stationary.sort()
    distinct_stationary = []
    for point in stationary:
        if (not distinct_stationary or
                abs(point - distinct_stationary[-1]) > 2e-10 * max(1.0, abs(point))):
            distinct_stationary.append(point)
    stationary = distinct_stationary
    if len(stationary) > 1:
        residuals = value_and_derivative(np.asarray(stationary))[0]
        for first, second in zip(residuals[:-1], residuals[1:]):
            uncertainty = 64.0 * np.finfo(float).eps * max(
                1.0, abs(target), abs(target + first), abs(target + second))
            if (abs(first - second) <= uncertainty
                    and max(abs(first), abs(second)) <= uncertainty):
                raise NumericalConvergenceError(
                    "centroid branches are unresolved at floating-point precision "
                    "near a cusp; specify the continuous common centre explicitly")
    # Monotonic pieces are bracketed by the extrema of the continuous curve,
    # not by a target-dependent grid.  Two roots arbitrarily near a fold are
    # thus separated even when no sampling point lies between the two roots.
    x = np.unique(np.concatenate((x, np.asarray(stationary))))
    f = value_and_derivative(x)[0]
    roots = []
    for k in range(len(x) - 1):
        if f[k] == 0.0:
            roots.append(x[k])
        if f[k] * f[k + 1] < 0:
            roots.append(brentq(equation, x[k], x[k + 1], xtol=2e-13, rtol=2e-14))
    if f[-1] == 0.0:
        roots.append(x[-1])
    # A double root need not change sign: inspect the independently solved
    # stationary points, without re-minimizing a target-dependent objective.
    for z in stationary:
        if abs(equation(z)) <= 64.0 * np.finfo(float).eps * max(target, 1.0):
            roots.append(z)
    roots.sort()
    unique = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 5e-11 * max(1.0, abs(root)):
            unique.append(root)
    if not unique:
        raise NumericalConvergenceError("could not resolve continuous centroid roots")
    return width * np.asarray(unique)


@lru_cache(maxsize=256)
def _cached_center_roots(lag, width, weights, ratio, enforce_nonnegative):
    return tuple(_center_roots_for_lag_uncached(lag, width, weights, ratio,
                                               enforce_nonnegative))


def center_roots_for_lag(lag, width, mixture_weights, sigma_ratio=0.45,
                         enforce_nonnegative=True):
    """Return every resolved centre branch; cache shape-only profile work.

    A fresh array is returned so a caller cannot mutate a cached root list.
    Tau/rho changes do not require solving the same centroid equation again.
    """
    width, weights, ratio = _validated(width, mixture_weights, sigma_ratio)
    return np.asarray(_cached_center_roots(float(lag), width, tuple(weights), ratio,
                                           bool(enforce_nonnegative)), dtype=float)


@lru_cache(maxsize=16)
def _legendre(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return (nodes + 1.0) * 0.5, weights * 0.5


def _component_limits(metadata, tail=2e-15):
    centers, sigma = metadata["component_centers"], metadata["component_sigma"]
    if not metadata["enforce_nonnegative"]:
        return centers - 9.0 * sigma, centers + 9.0 * sigma
    lower = np.maximum(0.0, centers - 9.0 * sigma)
    upper = centers - sigma * ndtri_exp(np.log(tail) + metadata["component_log_masses"])
    return lower, upper


def quadrature_from_metadata(metadata, grid_size=61):
    """Positive nonuniform physical-delay quadrature of a *fixed* response.

    The requested size is a minimum budget, not a fixed global grid.  Each
    component is resolved on its own effective support; the continuous centres
    are never shifted in response to quadrature resolution.
    """
    n = max(32, int(np.ceil(int(grid_size) / 3.0)))
    lower, upper = _component_limits(metadata)
    centers, sigma = metadata["component_centers"], metadata["component_sigma"]
    for _ in range(5):
        u, w = _legendre(n)
        q = lower[:, None] + (upper - lower)[:, None] * u
        logw = (np.log(w)[None, :] - 0.5 * ((q - centers[:, None]) / sigma) ** 2)
        logw -= logsumexp(logw, axis=1)[:, None]
        weights = np.exp(logw) * metadata["effective_weights"][:, None]
        error = abs(float(np.sum(q * weights)) - metadata["centroid"])
        tolerance = 2e-11 * max(1.0, abs(metadata["centroid"]), metadata["width"])
        if error <= tolerance:
            order = np.argsort(q.ravel())
            nodes, weights = q.ravel()[order], weights.ravel()[order]
            positive = weights > 0
            weights = weights[positive]
            return nodes[positive], weights / np.sum(weights)
        n *= 2
    raise NumericalConvergenceError("three-Gaussian quadrature mean did not converge")


def _log_difference(log_a, log_b):
    """log(exp(a)-exp(b)), for a >= b, including infinite endpoints."""
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.minimum(np.asarray(log_b) - np.asarray(log_a), 0.0)
        result = np.asarray(log_a) + np.log(-np.expm1(delta))
    return np.where(np.isneginf(log_b), log_a, result)


def _log_normal_interval(lower, upper):
    lower, upper = np.broadcast_arrays(lower, upper)
    use_survival = lower > 0.0
    log_a = np.where(use_survival, log_ndtr(-lower), log_ndtr(upper))
    log_b = np.where(use_survival, log_ndtr(-upper), log_ndtr(lower))
    return _log_difference(log_a, log_b)


def _cross_raw(x, tau, metadata):
    shape = np.asarray(x).shape
    x = np.asarray(x, dtype=float).ravel()[:, None]
    centers = metadata["component_centers"][None, :]
    sigma = metadata["component_sigma"]
    k = 1.0 / tau
    lower = 0.0 if metadata["enforce_nonnegative"] else -np.inf
    split = np.maximum(x, lower)
    log_left = (-k * x + k * centers + 0.5 * (k * sigma) ** 2
                + _log_normal_interval((lower - centers - k * sigma**2) / sigma,
                                       (split - centers - k * sigma**2) / sigma))
    log_right = (k * x - k * centers + 0.5 * (k * sigma) ** 2
                 + log_ndtr((centers - split - k * sigma**2) / sigma))
    log_total = logsumexp(np.stack((log_left, log_right)), axis=0)
    log_total += np.log(metadata["mixture_weights"])[None, :] - metadata["log_mass"]
    return np.exp(logsumexp(log_total, axis=1)).reshape(shape)


def _difference_density(z, metadata):
    """Density of Q-Q', evaluated in stable logarithmic form."""
    z = np.asarray(z, dtype=float)
    original_shape = z.shape
    z = z.ravel()[:, None]
    centers = metadata["component_centers"]
    sigma = metadata["component_sigma"]
    delta = (centers[:, None] - centers[None, :]).ravel()
    log_weights = (np.log(metadata["mixture_weights"])[:, None]
                   + np.log(metadata["mixture_weights"])[None, :]).ravel()
    values = (-0.25 * ((z - delta) / sigma) ** 2
              - np.log(sigma) - 0.5 * np.log(4.0 * np.pi) + log_weights)
    if metadata["enforce_nonnegative"]:
        total = (centers[:, None] + centers[None, :]).ravel()
        values += log_ndtr((total - np.abs(z)) / (_SQRT2 * sigma))
        values -= 2.0 * metadata["log_mass"]
    return np.exp(logsumexp(values, axis=1)).reshape(original_shape)


def _ou_autocorrelation_interpolant(tau, metadata, grid_size, rtol, atol):
    lower, upper = _component_limits(metadata)
    limit = max(float(np.max(upper) - np.min(lower)), np.finfo(float).eps)
    # No interval spans many exponential decay lengths: fixed high-order
    # quadrature then resolves the kernel even for a very short DRW timescale.
    initial = max(int(grid_size), int(np.ceil(limit / (4.0 * tau))), 16)
    if initial > 200000:
        raise NumericalConvergenceError("OU integration requires an excessive initial grid")
    edges = np.linspace(0.0, limit, initial + 1)
    u, weights = _legendre(16)

    def interval_integrals(left, right):
        span = right - left
        nodes = left[:, None] + span[:, None] * u
        density = _difference_density(nodes, metadata)
        base = density * (span[:, None] * weights)
        toward_right = np.sum(base * np.exp(-(right[:, None] - nodes) / tau), axis=1)
        toward_left = np.sum(base * np.exp(-(nodes - left[:, None]) / tau), axis=1)
        return toward_right, toward_left

    for iteration in range(14):
        left, right = edges[:-1], edges[1:]
        a, b = interval_integrals(left, right)
        decay = np.exp(-(right - left) / tau)
        suffix = np.zeros(len(edges))
        for j in range(len(a) - 1, -1, -1):
            suffix[j] = b[j] + decay[j] * suffix[j + 1]
        prefix = np.empty(len(edges))
        prefix[0] = suffix[0]  # symmetry supplies the whole negative half-axis
        for j in range(len(a)):
            prefix[j + 1] = a[j] + decay[j] * prefix[j]
        values = prefix + suffix
        derivative = (suffix - prefix) / tau
        spline = CubicHermiteSpline(edges, values, derivative)
        fractions = np.array([0.25, 0.5, 0.75])
        points = left[:, None] + (right - left)[:, None] * fractions
        lp = np.repeat(left, 3)
        rp = np.repeat(right, 3)
        pointflat = points.ravel()
        a_part, _ = interval_integrals(lp, pointflat)
        _, b_part = interval_integrals(pointflat, rp)
        actual = (np.repeat(prefix[:-1], 3) * np.exp(-(pointflat - lp) / tau)
                  + a_part + np.repeat(suffix[1:], 3) * np.exp(-(rp - pointflat) / tau)
                  + b_part)
        error = np.abs(spline(pointflat) - actual).reshape(-1, 3)
        tolerance = (atol + rtol * np.maximum(np.abs(actual), values[0])).reshape(-1, 3)
        bad = np.any(error > tolerance * 0.25, axis=1)
        if not np.any(bad):
            return spline, float(values[0]), limit, float(values[-1]), len(edges)
        edges = np.sort(np.concatenate((edges, 0.5 * (left[bad] + right[bad]))))
        if len(edges) > 200000:
            break
    raise NumericalConvergenceError("continuous three-Gaussian OU integration did not converge")


def ou_covariance_from_metadata(t_c, t_l, tau_drw, rho, metadata, *, grid_size=61,
                                rtol=2e-9, atol=2e-11):
    """OU cross/line correlation matrices for one fixed continuous response.

    Cross-covariances use analytic truncated normal-exponential integrals.
    Line autocovariances use the analytic delay-difference density and a
    tolerance-controlled, piecewise-polynomial exponential convolution.  This
    does not discretize Q and Q' onto an equal-spacing grid or alter centres.
    """
    tau, rho = float(tau_drw), float(rho)
    t_c, t_l = np.asarray(t_c, dtype=float), np.asarray(t_l, dtype=float)
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("positive OU timescale and 0<=rho<1 required")
    if rtol <= 0 or atol <= 0:
        raise ValueError("integration tolerances must be positive")
    if t_c.ndim != 1 or t_l.ndim != 1 or np.any(~np.isfinite(t_c)) or np.any(~np.isfinite(t_l)):
        raise ValueError("sampling times must be finite one-dimensional arrays")
    spline, variance, limit, tail_value, nodes = _ou_autocorrelation_interpolant(
        tau, metadata, grid_size, float(rtol), float(atol))
    if not np.isfinite(variance) or variance <= 0:
        raise NumericalConvergenceError("continuous response variance is not positive")
    x = t_l[None, :] - t_c[:, None]
    cross = rho * _cross_raw(x, tau, metadata) / np.sqrt(variance)
    h = np.abs(t_l[:, None] - t_l[None, :])
    line_raw = spline(np.minimum(h, limit))
    outside = h > limit
    line_raw[outside] = tail_value * np.exp(-(h[outside] - limit) / tau)
    line = line_raw / variance
    line = 0.5 * (line + line.T)
    np.fill_diagonal(line, 1.0)
    metadata["integration_diagnostics"] = {
        "method": "continuous_normal_difference_ou",
        "rtol": float(rtol), "atol": float(atol),
        "autocovariance_nodes": nodes,
        "response_variance": variance,
        "difference_support_upper": limit,
    }
    if np.any(~np.isfinite(cross)) or np.any(~np.isfinite(line)):
        raise NumericalConvergenceError("non-finite continuous response covariance")
    return cross, line


# Descriptive aliases retained for callers that use the shorter numerical API.
response_state = metadata_from_center
centers_for_centroid = center_roots_for_lag
