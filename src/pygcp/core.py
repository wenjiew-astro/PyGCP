# -*- coding: utf-8 -*-
"""Core implementation of PyGCP's GCP-RM reverberation-mapping model.

This module contains the numerical model only. Most users should import the
public API from :mod:`pygcp` instead of importing this module directly.

Pipeline
--------
    flux, err
      -> KDE CDF Gaussianizing transform
      -> delta-method propagation of flux errors to Gaussian-score errors
      -> OU/DRW reverberation covariance convolved with a chosen transfer model
      -> K_obs = K_rev + D_meas + eps_jit I
      -> R_obs = corr(K_obs)
      -> Gaussian-copula log likelihood

Supported transfer models
-------------------------
    delta, tophat, isosceles_triangle, forward_triangle,
    long_tail, double_exponential, edge_on_ring, flexible_gaussian3

For all finite-width transfer functions, the response is inserted consistently
into BOTH the continuum-line cross-covariance and the line-line auto-covariance.
The top-hat keeps the original analytic OU expressions. Other transfer models use
an efficient one-dimensional quadrature; K_LL uses the autocorrelation of the
transfer weights, avoiding a Python M^2 double loop.

The high-level ``run_transfer_model_selection`` function:
  1. fits candidate transfer models in parallel using joint MLE;
  2. ranks them with blocked predictive likelihood (default) or AIC/AICc/BIC;
  3. runs a full lag profile only for the requested best/supported models;
  4. returns the best model without model averaging.

Public API
----------
The public package is :mod:`pygcp`. ``run_mle(..., transfer="tophat")`` is a
single-transfer fit, while
``run_transfer_model_selection`` performs multi-transfer automatic selection.

Each input file/table/array must have either two or three columns:
    time   flux [err]

The optional ``err`` column contains measurement errors. If it is omitted,
the data are treated as having no supplied measurement error (``err = 0``).
"""

import math
import os
import csv
import json
import copy
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import _flexible

try:
    from iminuit import Minuit
except Exception:
    Minuit = None

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:
    scipy_minimize = None


ND = NormalDist()
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# -----------------------------------------------------------------------------
# Basic normal functions and utilities
# -----------------------------------------------------------------------------

def normal_cdf_scalar(x):
    return 0.5 * (1.0 + math.erf(float(x) / SQRT2))


def normal_pdf_array(x):
    x = np.asarray(x, dtype=float)
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def normal_ppf_array(u):
    u = np.asarray(u, float)
    out = np.empty_like(u, dtype=float)
    flat_u = u.ravel()
    flat_out = out.ravel()
    for i, val in enumerate(flat_u):
        flat_out[i] = ND.inv_cdf(float(val))
    return out


def expit_scalar(x):
    x = float(x)
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def logit_scalar(p):
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def data_duration(t_c, t_l):
    t_min = float(min(np.min(t_c), np.min(t_l)))
    t_max = float(max(np.max(t_c), np.max(t_l)))
    return max(t_max - t_min, 1.0)


def ou_corr(a, b, tau):
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return np.exp(-np.abs(a - b) / float(tau))


# -----------------------------------------------------------------------------
# Transfer-function registry and efficient numerical convolution
# -----------------------------------------------------------------------------

TRANSFER_ALIASES = {
    "delta": "delta",
    "tophat": "tophat",
    "isosceles_triangle": "isosceles_triangle",
    "forward_triangle": "forward_triangle",
    "long_tail": "long_tail",
    "double_exponential": "double_exponential",
    "edge_on_ring": "edge_on_ring",
    "flexible_gaussian3": "flexible_gaussian3",
}

SUPPORTED_TRANSFERS = (
    "delta",
    "tophat",
    "isosceles_triangle",
    "forward_triangle",
    "long_tail",
    "double_exponential",
    "edge_on_ring",
    "flexible_gaussian3",
)

WIDTH_TRANSFERS = {
    "tophat",
    "isosceles_triangle",
    "forward_triangle",
    "long_tail",
    "double_exponential",
    "flexible_gaussian3",
}


def normalize_transfer_name(name):
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in TRANSFER_ALIASES:
        raise ValueError(
            f"unknown transfer={name!r}; supported: {', '.join(SUPPORTED_TRANSFERS)}"
        )
    return TRANSFER_ALIASES[key]


def transfer_has_width(name):
    return normalize_transfer_name(name) in WIDTH_TRANSFERS


def _reported_transfer_width(params):
    """Return a meaningful width for user-facing output, else ``None``."""
    if not transfer_has_width(params["transfer"]):
        return None
    return float(params["transfer_width"])


def _finite_number(name, value, minimum=None, strict=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        comparison = ">" if strict else ">="
        raise ValueError(f"{name} must be {comparison} {minimum}")
    return value


def _validated_bounds(name, values, positive=False):
    try:
        lower, upper = values
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two bounds") from exc
    lower = _finite_number(name, lower)
    upper = _finite_number(name, upper)
    if lower >= upper or (positive and lower <= 0):
        condition = "0 < lower < upper" if positive else "lower < upper"
        raise ValueError(f"{name} must satisfy {condition}")
    return lower, upper


def _positive_integer(name, value, minimum=1):
    value = _finite_number(name, value, minimum)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)




def _uniform_midpoints(left, right, n):
    n = max(int(n), 9)
    edges = np.linspace(float(left), float(right), n + 1)
    return 0.5 * (edges[:-1] + edges[1:]), float((right - left) / n)


def _normalize_transfer_weights(offsets, density):
    offsets = np.asarray(offsets, dtype=float)
    density = np.asarray(density, dtype=float)
    density = np.where(np.isfinite(density) & (density > 0.0), density, 0.0)
    total = float(np.sum(density))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("transfer quadrature has zero or non-finite total weight")
    weights = density / total
    # Enforce exactly zero weighted mean so that lag is the transfer centroid.
    offsets = offsets - float(np.sum(weights * offsets))
    return offsets, weights


def _make_discrete_transfer_quadrature(
    transfer,
    lag,
    width,
    grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    enforce_nonnegative=False,
):
    """Return delay offsets and normalized quadrature weights.

    Actual delays are ``lag + offsets``. The weighted mean of offsets is zero,
    so ``lag`` is the mean/centroid delay for every model.
    """
    transfer = normalize_transfer_name(transfer)
    lag = float(lag)
    width = float(width) if transfer_has_width(transfer) else 1.0
    m = max(int(grid_size), 9)
    if m % 2 == 0:
        m += 1

    if transfer == "delta":
        offsets = np.array([0.0])
        weights = np.array([1.0])

    elif transfer == "tophat":
        if width <= 0.0:
            raise ValueError("tophat width must be positive")
        offsets, _ = _uniform_midpoints(-0.5 * width, 0.5 * width, m)
        weights = np.full(m, 1.0 / m)

    elif transfer == "isosceles_triangle":
        if width <= 0.0:
            raise ValueError("triangle width must be positive")
        offsets, _ = _uniform_midpoints(-0.5 * width, 0.5 * width, m)
        density = np.maximum(1.0 - 2.0 * np.abs(offsets) / width, 0.0)
        offsets, weights = _normalize_transfer_weights(offsets, density)

    elif transfer == "forward_triangle":
        if width <= 0.0:
            raise ValueError("forward-triangle width must be positive")
        left, right = -width / 3.0, 2.0 * width / 3.0
        offsets, _ = _uniform_midpoints(left, right, m)
        density = np.maximum(right - offsets, 0.0)
        offsets, weights = _normalize_transfer_weights(offsets, density)

    elif transfer == "long_tail":
        if width <= 0.0:
            raise ValueError("long-tail short width must be positive")
        width_ratio = float(long_tail_width_ratio)
        ks = float(long_tail_scale_ratio)
        if width_ratio <= 1.0 or ks <= 0.0:
            raise ValueError("long-tail ratios require width_ratio>1 and scale_ratio>0")
        w1 = width
        w2 = width_ratio * width
        s1 = ks / (ks + 1.0)
        s2 = 1.0 / (ks + 1.0)
        lag1 = -(s2 / 3.0) * (w2 - w1)
        lag2 = +(s1 / 3.0) * (w2 - w1)
        left = min(lag1 - w1 / 3.0, lag2 - w2 / 3.0)
        right = max(lag1 + 2.0 * w1 / 3.0, lag2 + 2.0 * w2 / 3.0)
        offsets, _ = _uniform_midpoints(left, right, m)
        d1 = np.where(
            (offsets >= lag1 - w1 / 3.0) & (offsets <= lag1 + 2.0 * w1 / 3.0),
            np.maximum(lag1 + 2.0 * w1 / 3.0 - offsets, 0.0),
            0.0,
        )
        d2 = np.where(
            (offsets >= lag2 - w2 / 3.0) & (offsets <= lag2 + 2.0 * w2 / 3.0),
            np.maximum(lag2 + 2.0 * w2 / 3.0 - offsets, 0.0),
            0.0,
        )
        # Each unnormalized forward triangle integrates as width^2/2.
        density = s1 * d1 / (w1 * w1) + s2 * d2 / (w2 * w2)
        offsets, weights = _normalize_transfer_weights(offsets, density)

    elif transfer == "double_exponential":
        if width <= 0.0:
            raise ValueError("double-exponential rise width must be positive")
        w1 = width
        w2 = float(double_exp_decay_ratio) * width
        if w2 <= 0.0:
            raise ValueError("double-exponential decay width must be positive")
        # Yu et al. parameterization, centered so its exact mean is lag.
        t0 = -w2 - w1 * w2 / (w1 + w2)
        right = t0 + 10.0 * w2
        offsets, _ = _uniform_midpoints(t0, right, m)
        x = np.maximum((offsets - t0) / w1, 0.0)
        y = np.maximum((offsets - t0) / w2, 0.0)
        density = (1.0 - np.exp(-x)) * np.exp(-y)
        offsets, weights = _normalize_transfer_weights(offsets, density)

    elif transfer == "edge_on_ring":
        if lag <= 0.0:
            raise ValueError("edge-on-ring requires lag>0")
        # Uniform midpoint quadrature of the arcsine density on [-lag, lag].
        offsets, _ = _uniform_midpoints(-lag, lag, m)
        density = 1.0 / np.sqrt(np.maximum(lag * lag - offsets * offsets, 1e-15))
        offsets, weights = _normalize_transfer_weights(offsets, density)

    else:  # pragma: no cover
        raise AssertionError(transfer)

    if enforce_nonnegative and float(np.min(lag + offsets)) < -1e-10:
        raise ValueError("transfer support extends to negative physical delay")
    return offsets, weights


def _weighted_ou_convolution_covariance(t_c, t_l, lag, tau, rho, offsets, weights):
    """Efficient OU covariance for one discretized unit-area transfer function."""
    t_c = np.asarray(t_c, float)
    t_l = np.asarray(t_l, float)
    offsets = np.asarray(offsets, float)
    weights = np.asarray(weights, float)

    dt_cl = t_c[:, None] - t_l[None, :] + float(lag)
    raw_cl = np.zeros_like(dt_cl, dtype=float)
    for off, wt in zip(offsets, weights):
        raw_cl += wt * np.exp(-np.abs(dt_cl + off) / float(tau))

    dt_ll = t_l[:, None] - t_l[None, :]
    if len(offsets) == 1:
        raw_ll = np.exp(-np.abs(dt_ll) / float(tau))
        variance = 1.0
    else:
        spacing = np.diff(offsets)
        if np.any(spacing <= 0) or not np.allclose(
            spacing, spacing[0], rtol=1e-8, atol=1e-12,
        ):
            raise ValueError(
                "the autocorrelation shortcut requires equally spaced nodes; "
                "use the continuous flexible covariance for three-Gaussian responses"
            )
        # All quadratures above use a uniform grid. The transfer-weight
        # autocorrelation reduces the line-line double sum from M^2 to 2M-1.
        dq = float(np.median(np.diff(offsets)))
        coeff = np.correlate(weights, weights, mode="full")
        delay_diff = np.arange(-(len(weights) - 1), len(weights), dtype=float) * dq
        raw_ll = np.zeros_like(dt_ll, dtype=float)
        variance = 0.0
        for c, d in zip(coeff, delay_diff):
            raw_ll += c * np.exp(-np.abs(dt_ll - d) / float(tau))
            variance += float(c) * math.exp(-abs(float(d)) / float(tau))

    variance = max(float(variance), 1e-15)
    Kcl = float(rho) * raw_cl / math.sqrt(variance)
    Kll = raw_ll / variance
    Kll = 0.5 * (Kll + Kll.T)
    np.fill_diagonal(Kll, 1.0)
    return Kcl, np.clip(Kll, -1.0, 1.0)


# -----------------------------------------------------------------------------
# Analytic OU + normalized top-hat covariance
# -----------------------------------------------------------------------------

def _sinhc(x):
    """Return sinh(x)/x with a stable series near zero."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-5
    xs = x[small]
    out[small] = 1.0 + xs * xs / 6.0 + xs**4 / 120.0
    out[~small] = np.sinh(x[~small]) / x[~small]
    return out


def ou_tophat_variance(tau, width):
    """Variance of a unit-area top-hat convolution of a unit-variance OU GP.

    v_H = (1/w^2) int int exp(-|q1-q2|/tau) dq1 dq2
        = 2 [x + exp(-x) - 1] / x^2,  x=w/tau.
    """
    tau = float(tau)
    width = float(width)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if width <= 0.0:
        return 1.0
    x = width / tau
    if x < 1e-4:
        # 1 - x/3 + x^2/12 - x^3/60 + x^4/360 + O(x^5)
        return float(1.0 - x / 3.0 + x * x / 12.0 - x**3 / 60.0 + x**4 / 360.0)
    return float(2.0 * (x + math.expm1(-x)) / (x * x))


def ou_tophat_cross_corr(t_c, t_l, lag, tau, width):
    """Correlation between X(t) and the normalized top-hat response H(s).

    H(s) = [1/sqrt(v_H)] * (1/w) int_{-w/2}^{w/2}
           X[s - (lag + q)] dq.

    The normalization gives Var[H(s)] = 1 before measurement noise.
    """
    t_c = np.asarray(t_c, float)[:, None]
    t_l = np.asarray(t_l, float)[None, :]
    tau = float(tau)
    width = float(width)

    if width <= 0.0:
        return np.exp(-np.abs(t_c - (t_l - float(lag))) / tau)

    d = t_c - (t_l - float(lag))
    r = np.abs(d)
    half = 0.5 * width
    x = r / tau
    a = width / tau

    avg = np.empty_like(r)
    outside = r >= half

    # If the integration interval does not cross zero:
    # (1/w) int exp(-|d+q|/tau)dq
    # = exp(-|d|/tau) * sinh(a/2)/(a/2).
    avg[outside] = np.exp(-x[outside]) * float(_sinhc(np.array([0.5 * a]))[0])

    # If the interval crosses zero:
    # 2*tau/w * [1 - exp(-w/(2*tau))*cosh(|d|/tau)].
    inside = ~outside
    if np.any(inside):
        avg[inside] = (
            2.0 * tau / width
            * (1.0 - np.exp(-0.5 * a) * np.cosh(x[inside]))
        )

    v_h = ou_tophat_variance(tau, width)
    return avg / math.sqrt(max(v_h, 1e-15))


def ou_tophat_auto_corr(t1, t2, tau, width):
    """Correlation of two normalized top-hat responses of the same OU GP.

    This is the double convolution

        (1/w^2) int int k[(t1-q1),(t2-q2)] dq1 dq2,

    divided by its zero-lag value so that the diagonal is unity.
    """
    t1 = np.asarray(t1, float)[:, None]
    t2 = np.asarray(t2, float)[None, :]
    tau = float(tau)
    width = float(width)

    if width <= 0.0:
        return np.exp(-np.abs(t1 - t2) / tau)

    r = np.abs(t1 - t2)
    x = r / tau
    a = width / tau
    raw = np.empty_like(r)

    outside = r >= width
    # For |d| >= w the triangular convolution has a simple product form.
    factor = float(_sinhc(np.array([0.5 * a]))[0])
    raw[outside] = np.exp(-x[outside]) * factor * factor

    inside = ~outside
    if np.any(inside):
        ri = r[inside]
        xi = x[inside]
        # Stable form of
        # [2*tau*(w-r) + 2*tau^2*(exp(-w/tau)cosh(r/tau)-exp(-r/tau))]/w^2.
        exp_term = 0.5 * np.exp(-xi) * (
            np.expm1(2.0 * xi - a) + math.expm1(-a)
        )
        raw[inside] = (
            2.0 * tau * (width - ri) + 2.0 * tau * tau * exp_term
        ) / (width * width)

    v_h = ou_tophat_variance(tau, width)
    out = raw / max(v_h, 1e-15)
    # Numerical roundoff can create tiny deviations on the diagonal.
    return np.clip(out, -1.0, 1.0)




# -----------------------------------------------------------------------------
# KDE CDF transform and delta-method error propagation
# -----------------------------------------------------------------------------

def kde_cdf_pdf_at_samples(x, bandwidth=None):
    """Return Gaussian-KDE CDF and PDF evaluated at the sample points."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        raise ValueError("KDE transform needs at least 5 points")
    s = float(np.std(x, ddof=1))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("KDE transform needs non-constant finite data")

    h = (
        1.06 * s * n ** (-1.0 / 5.0)
        if bandwidth is None
        else _finite_number("kde_bandwidth", bandwidth, 0, strict=True)
    )
    h = max(h, 1e-12)

    u = np.empty(n, dtype=float)
    pdf = np.empty(n, dtype=float)
    for i in range(n):
        vals = (x[i] - x) / h
        u[i] = sum(normal_cdf_scalar(v) for v in vals) / n
        pdf[i] = float(np.mean(normal_pdf_array(vals)) / h)
    return u, pdf, h


def kde_gaussian_transform_with_error(
    flux,
    err,
    bandwidth=None,
    sigma_z_floor=1e-8,
    sigma_z_max=None,
):
    """KDE Gaussianizing transform plus first-order error propagation.

    z_i = Phi^{-1}[Ghat(y_i)]
    sigma_z_i ~= sigma_y_i * ghat(y_i) / phi(z_i)
    """
    flux = np.asarray(flux, dtype=float).reshape(-1)
    err = np.asarray(err, dtype=float).reshape(-1)
    if flux.shape != err.shape:
        raise ValueError("flux and err must have the same shape")
    if np.any(~np.isfinite(flux)) or np.any(~np.isfinite(err)):
        raise ValueError("flux and err must be finite")
    if np.any(err < 0.0):
        raise ValueError("measurement errors must be non-negative")

    n = len(flux)
    u_raw, pdf, bandwidth_used = kde_cdf_pdf_at_samples(flux, bandwidth=bandwidth)
    eps = 0.5 / (n + 1.0)
    u = np.clip(u_raw, eps, 1.0 - eps)
    z = normal_ppf_array(u)

    phi_z = np.clip(normal_pdf_array(z), 1e-300, np.inf)
    sigma_z = err * pdf / phi_z
    if sigma_z_floor is not None:
        sigma_z = np.maximum(sigma_z, float(sigma_z_floor))
    if sigma_z_max is not None:
        sigma_z = np.minimum(sigma_z, float(sigma_z_max))

    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(sigma_z)):
        raise ValueError("non-finite Gaussian scores or propagated score errors")

    return {
        "z": z,
        "sigma_z": sigma_z,
        "u": u,
        "u_raw": u_raw,
        "pdf": pdf,
        "bandwidth": bandwidth_used,
    }



def kde_cdf_pdf_from_reference(reference_flux, eval_flux, bandwidth=None):
    """Evaluate a Gaussian KDE fitted to ``reference_flux`` at ``eval_flux``.

    This out-of-sample form is used by blocked predictive validation so that
    held-out line points do not participate in their own marginal transform.
    """
    reference_flux = np.asarray(reference_flux, dtype=float).reshape(-1)
    eval_flux = np.asarray(eval_flux, dtype=float).reshape(-1)
    n_ref = len(reference_flux)
    if n_ref < 5:
        raise ValueError("KDE reference sample needs at least 5 points")
    if np.any(~np.isfinite(reference_flux)) or np.any(~np.isfinite(eval_flux)):
        raise ValueError("KDE reference/evaluation fluxes must be finite")
    s_ref = float(np.std(reference_flux, ddof=1))
    if not np.isfinite(s_ref) or s_ref <= 0.0:
        raise ValueError("KDE reference sample must be non-constant")

    h = (
        1.06 * s_ref * n_ref ** (-1.0 / 5.0)
        if bandwidth is None
        else _finite_number("kde_bandwidth", bandwidth, 0, strict=True)
    )
    h = max(h, 1e-12)
    u = np.empty(len(eval_flux), dtype=float)
    pdf = np.empty(len(eval_flux), dtype=float)
    for i, value in enumerate(eval_flux):
        vals = (value - reference_flux) / h
        u[i] = sum(normal_cdf_scalar(v) for v in vals) / n_ref
        pdf[i] = float(np.mean(normal_pdf_array(vals)) / h)
    return u, pdf, h


def kde_gaussian_transform_from_reference(
    reference_flux,
    eval_flux,
    eval_err,
    bandwidth=None,
    sigma_z_floor=1e-8,
    sigma_z_max=None,
):
    """Out-of-sample Gaussianizing transform using a training-only KDE."""
    eval_flux = np.asarray(eval_flux, dtype=float).reshape(-1)
    eval_err = np.asarray(eval_err, dtype=float).reshape(-1)
    if eval_flux.shape != eval_err.shape:
        raise ValueError("eval_flux and eval_err must have the same shape")
    if np.any(~np.isfinite(eval_err)) or np.any(eval_err < 0.0):
        raise ValueError("evaluation errors must be finite and non-negative")

    reference_flux = np.asarray(reference_flux, dtype=float).reshape(-1)
    u_raw, pdf, bandwidth_used = kde_cdf_pdf_from_reference(
        reference_flux, eval_flux, bandwidth=bandwidth
    )
    eps = 0.5 / (len(reference_flux) + 1.0)
    u = np.clip(u_raw, eps, 1.0 - eps)
    z = normal_ppf_array(u)
    phi_z = np.clip(normal_pdf_array(z), 1e-300, np.inf)
    sigma_z = eval_err * pdf / phi_z
    if sigma_z_floor is not None:
        sigma_z = np.maximum(sigma_z, float(sigma_z_floor))
    if sigma_z_max is not None:
        sigma_z = np.minimum(sigma_z, float(sigma_z_max))
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(sigma_z)):
        raise ValueError("non-finite out-of-sample Gaussian scores or errors")
    return {
        "z": z,
        "sigma_z": sigma_z,
        "u": u,
        "u_raw": u_raw,
        "pdf": pdf,
        "bandwidth": bandwidth_used,
    }


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(continuum, line, sort=True, drop_nan=True):
    """Load two light curves with columns time, flux, and optional err."""
    continuum_is_file = isinstance(continuum, (str, bytes, os.PathLike))
    line_is_file = isinstance(line, (str, bytes, os.PathLike))
    if continuum_is_file:
        continuum = np.loadtxt(continuum)
    if line_is_file:
        line = np.loadtxt(line)

    out = []
    for name, raw in (("continuum", continuum), ("line", line)):
        if hasattr(raw, "iloc"):
            arr = raw.iloc[:, :3].to_numpy(dtype=float)
        else:
            arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"{name} data must have at least columns: time, flux")

        t = np.asarray(arr[:, 0], dtype=float).reshape(-1)
        y = np.asarray(arr[:, 1], dtype=float).reshape(-1)
        if arr.shape[1] >= 3:
            e = np.asarray(arr[:, 2], dtype=float).reshape(-1)
        else:
            # Two-column input has no supplied measurement-error column.
            e = np.zeros_like(y)

        mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
        if drop_nan:
            t, y, e = t[mask], y[mask], e[mask]
        elif not np.all(mask):
            raise ValueError(f"{name} time, flux, and errors must be finite when drop_nan=False")

        if len(t) < 5:
            raise ValueError(f"{name} data needs at least 5 finite points")
        if np.any(e < 0.0):
            raise ValueError(f"{name} errors must be non-negative")

        if sort:
            order = np.argsort(t)
            t, y, e = t[order], y[order], e[order]
        out.append((t, y, e))

    return {
        "t_c": out[0][0],
        "t_l": out[1][0],
        "c_flux": out[0][1],
        "l_flux": out[1][1],
        "c_err": out[0][2],
        "l_err": out[1][2],
        "source": "observed_files" if continuum_is_file or line_is_file else "observed_arrays",
    }


# -----------------------------------------------------------------------------
# Error-aware convolved Gaussian-copula likelihood
# -----------------------------------------------------------------------------



def covariance_to_correlation(K):
    K = np.asarray(K, dtype=float)
    d = np.sqrt(np.clip(np.diag(K), 1e-15, np.inf))
    R = K / np.outer(d, d)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def gaussian_copula_loglike_from_R(
    z,
    R,
    cholesky_jitter=1e-8,
    jitter_growth=10.0,
    max_tries=8,
):
    """Gaussian copula log density for standard-normal scores z."""
    z = np.asarray(z, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(z)
    eye = np.eye(n)
    jitter = float(cholesky_jitter)

    mvn_loglike = -np.inf
    for _ in range(int(max_tries)):
        try:
            L = np.linalg.cholesky(R + jitter * eye)
            if float(np.min(np.diag(L))) < 1e-12:
                return -np.inf
            y = np.linalg.solve(L, z)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            mvn_loglike = float(-0.5 * (y @ y + logdet + n * math.log(2.0 * math.pi)))
            break
        except np.linalg.LinAlgError:
            jitter *= float(jitter_growth)

    if not np.isfinite(mvn_loglike):
        return -np.inf

    marginal_normal_loglike = float(
        np.sum(-0.5 * z**2 - 0.5 * math.log(2.0 * math.pi))
    )
    val = mvn_loglike - marginal_normal_loglike
    return float(val) if np.isfinite(val) else -np.inf




def log_prior_params(tau_drw, rho, transfer_width, cfg):
    if not cfg["use_prior"]:
        return 0.0

    tau0 = cfg["tau_drw_prior_frac"] * cfg["total_duration"]
    sd = cfg["log_tau_prior_sd"]
    lp = -0.5 * ((math.log(tau_drw) - math.log(tau0)) / sd) ** 2

    r = np.clip(rho / cfg["rho_max"], 1e-12, 1.0 - 1e-12)
    lp += (cfg["rho_prior_a"] - 1.0) * math.log(r)
    lp += (cfg["rho_prior_b"] - 1.0) * math.log(1.0 - r)

    if cfg["fit_transfer_width"] and cfg["transfer_width_prior"] is not None:
        w0 = cfg["transfer_width_prior"]
        sw = cfg["log_width_prior_sd"]
        lp += -0.5 * ((math.log(transfer_width) - math.log(w0)) / sw) ** 2
    return float(lp)


# -----------------------------------------------------------------------------
# Parameter transforms and optimization
# -----------------------------------------------------------------------------







def _finite_float_or_none(value):
    """Return a JSON-safe finite float, otherwise ``None``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _physical_parameter_values(p_hat, cfg):
    """Physical fitted/derived parameters exposed in uncertainty outputs."""
    values = {
        "lag": float(p_hat["lag"]),
        "tau_drw": float(p_hat["tau_drw"]),
        "rho": float(p_hat["rho"]),
        "jitter": float(p_hat["jitter"]),
    }
    fitted = {
        "lag": True,
        "tau_drw": True,
        "rho": True,
        "jitter": False,
    }

    if transfer_has_width(p_hat["transfer"]):
        values["transfer_width"] = float(p_hat["transfer_width"])
        fitted["transfer_width"] = bool(cfg["fit_transfer_width"])

    weights = p_hat.get("transfer_weights")
    centers = p_hat.get("transfer_component_centers")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        centers = np.asarray(centers, dtype=float)
        for index in range(3):
            values[f"flexible_weight_{index + 1}"] = float(weights[index])
            values[f"flexible_center_{index + 1}"] = float(centers[index])
            fitted[f"flexible_weight_{index + 1}"] = True
            fitted[f"flexible_center_{index + 1}"] = True
        values["flexible_component_sigma"] = float(
            p_hat["transfer_component_sigma"]
        )
        values["flexible_common_center_shift"] = float(
            p_hat["transfer_common_center_shift"]
        )
        fitted["flexible_component_sigma"] = True
        fitted["flexible_common_center_shift"] = True
    return values, fitted


def _empty_parameter_uncertainties(p_hat, cfg, status, message=None):
    values, fitted = _physical_parameter_values(p_hat, cfg)
    return {
        "method": "iminuit_hesse_delta_method",
        "available": False,
        "status": str(status),
        "message": None if message is None else str(message),
        "diagnostics": {
            "is_valid": None,
            "has_accurate_covar": None,
            "has_posdef_covar": None,
            "has_parameters_at_limit": None,
            "edm": None,
        },
        "internal_parameters": {},
        "internal_covariance": None,
        "physical_parameters": {
            name: {
                "value": _finite_float_or_none(value),
                "error": None,
                "fitted": bool(fitted[name]),
            }
            for name, value in values.items()
        },
    }


def estimate_hesse_parameter_uncertainties(minuit_result, p_hat, cfg, enabled=True):
    """Run HESSE and propagate its covariance to physical parameters.

    Minuit optimizes several transformed variables (log time-scales, a logit
    correlation, a feasible top-hat width fraction, and softmax logits).
    A numerical Jacobian of the optimizer-to-physical transformation
    propagates the full internal covariance matrix, preserving correlations and
    also covering the derived flexible-Gaussian weights, centers, and sigma.
    Failures are returned as structured diagnostics and never invalidate an
    otherwise successful maximum-likelihood fit.
    """
    if not enabled:
        return _empty_parameter_uncertainties(p_hat, cfg, "disabled")
    if minuit_result is None or not hasattr(minuit_result, "hesse"):
        return _empty_parameter_uncertainties(
            p_hat,
            cfg,
            "unavailable_backend",
            "iminuit is unavailable; optimizer used the SciPy fallback",
        )

    try:
        minuit_result.hesse()
        names = list(minuit_result.parameters)
        x_hat = np.asarray([minuit_result.values[name] for name in names], float)
        covariance_view = minuit_result.covariance
        if covariance_view is None:
            return _empty_parameter_uncertainties(
                p_hat, cfg, "covariance_unavailable", "HESSE returned no covariance"
            )
        covariance = np.asarray(
            [[covariance_view[a, b] for b in names] for a in names],
            dtype=float,
        )
        if covariance.shape != (len(names), len(names)) or not np.all(
            np.isfinite(covariance)
        ):
            return _empty_parameter_uncertainties(
                p_hat, cfg, "covariance_invalid", "non-finite HESSE covariance"
            )

        base_values, fitted = _physical_parameter_values(p_hat, cfg)
        physical_names = list(base_values)
        jacobian = np.zeros((len(physical_names), len(names)), dtype=float)

        def physical_vector(x):
            decode = getattr(minuit_result, "_gcp_rm_to_model_x", None)
            if decode is not None:
                x = decode(x)
            params = _x_to_params(x, cfg)
            values, _ = _physical_parameter_values(params, cfg)
            return np.asarray([values[name] for name in physical_names], dtype=float)

        base_vector = physical_vector(x_hat)
        for column, name in enumerate(names):
            scale = max(abs(float(x_hat[column])), 1.0)
            h = max(1e-6 * scale, 1e-7)
            lower, upper = minuit_result.limits[name]
            can_step_down = lower is None or x_hat[column] - h >= lower
            can_step_up = upper is None or x_hat[column] + h <= upper
            if can_step_down and can_step_up:
                x_low = x_hat.copy()
                x_high = x_hat.copy()
                x_low[column] -= h
                x_high[column] += h
                derivative = (physical_vector(x_high) - physical_vector(x_low)) / (2.0 * h)
            elif can_step_up:
                x_high = x_hat.copy()
                x_high[column] += h
                derivative = (physical_vector(x_high) - base_vector) / h
            elif can_step_down:
                x_low = x_hat.copy()
                x_low[column] -= h
                derivative = (base_vector - physical_vector(x_low)) / h
            else:
                derivative = np.full(len(physical_names), np.nan)
            jacobian[:, column] = derivative

        covariance_physical = jacobian @ covariance @ jacobian.T
        variances = np.diag(covariance_physical)
        errors = np.where(
            np.isfinite(variances) & (variances >= -1e-12),
            np.sqrt(np.clip(variances, 0.0, None)),
            np.nan,
        )

        fmin = minuit_result.fmin
        diagnostics = {
            "is_valid": bool(fmin.is_valid),
            "has_accurate_covar": bool(fmin.has_accurate_covar),
            "has_posdef_covar": bool(fmin.has_posdef_covar),
            "has_parameters_at_limit": bool(fmin.has_parameters_at_limit),
            "edm": _finite_float_or_none(fmin.edm),
        }
        available = bool(
            np.all(np.isfinite(errors[[fitted[name] for name in physical_names]]))
        )
        status = (
            "ok"
            if available
            and diagnostics["is_valid"]
            and diagnostics["has_accurate_covar"]
            and diagnostics["has_posdef_covar"]
            else "warning"
        )
        message = None
        if (getattr(minuit_result, "_gcp_rm_to_model_x", None) is not None
                and diagnostics["has_parameters_at_limit"]):
            status = "warning"
            message = (
                "The fit is at a parameter/transfer-support boundary; symmetric "
                "HESSE errors should not be treated as validated confidence intervals."
            )
        return {
            "method": "iminuit_hesse_delta_method",
            "available": available,
            "status": status,
            "message": message,
            "diagnostics": diagnostics,
            "internal_parameters": {
                name: {
                    "value": _finite_float_or_none(minuit_result.values[name]),
                    "error": _finite_float_or_none(minuit_result.errors[name]),
                }
                for name in names
            },
            "internal_covariance": covariance.tolist(),
            "physical_parameters": {
                name: {
                    "value": _finite_float_or_none(base_values[name]),
                    "error": (
                        _finite_float_or_none(errors[index])
                        if fitted[name]
                        else None
                    ),
                    "fitted": bool(fitted[name]),
                }
                for index, name in enumerate(physical_names)
            },
        }
    except Exception as exc:
        return _empty_parameter_uncertainties(
            p_hat, cfg, "hesse_failed", repr(exc)
        )


def physical_parameter_error(parameter_uncertainties, name):
    """Read one JSON-safe physical HESSE error from an uncertainty report."""
    item = (parameter_uncertainties.get("physical_parameters") or {}).get(name) or {}
    return _finite_float_or_none(item.get("error"))


def _add_flexible_uncertainty_errors(summary, parameter_uncertainties, params):
    """Add flexible-transfer-only fields without polluting other results."""
    if params.get("transfer_weights") is None:
        return
    names = [
        "flexible_weight_1",
        "flexible_weight_2",
        "flexible_weight_3",
        "flexible_center_1",
        "flexible_center_2",
        "flexible_center_3",
        "flexible_component_sigma",
        "flexible_common_center_shift",
    ]
    for name in names:
        summary[f"{name}_error"] = physical_parameter_error(
            parameter_uncertainties, name
        )


def _fit_score_kind(cfg):
    if cfg["use_prior"] or (
        cfg["transfer"] == "flexible_gaussian3" and cfg["flexible_smoothness"] > 0
    ):
        return "penalized_objective"
    return "data_loglike"


def _summarize_scan(lag_grid, scores, cfg):
    summary = likelihood_summary(lag_grid, scores)
    summary["scan_score_kind"] = _fit_score_kind(cfg)
    if summary["scan_score_kind"] != "data_loglike":
        # A penalized objective profile is not an ordinary likelihood-ratio CI.
        summary["lr68_regions"] = []
        summary["lr95_regions"] = []
    else:
        for name in ("lr68_regions", "lr95_regions"):
            summary[name] = [
                (max(left, cfg["lag_min"]), min(right, cfg["lag_max"]))
                for left, right in summary[name]
            ]
    return summary






class _ScipyMinuitAdapter:
    def __init__(self, x, names, result):
        self.values = {name: float(val) for name, val in zip(names, x)}
        self.valid = bool(getattr(result, "success", False))
        self.fval = float(getattr(result, "fun", np.nan))
        self.message = str(getattr(result, "message", ""))
        self.backend = "scipy_powell"
        self.nfcn = int(getattr(result, "nfev", 0))


class _FeasibleCoordinateFit:
    """Delegate optimizer diagnostics while retaining its coordinate mapping."""

    def __init__(self, fit, decode, names):
        self._fit = fit
        self._gcp_rm_to_model_x = decode
        self._gcp_rm_optimizer_names = tuple(names)

    def __getattr__(self, name):
        return getattr(self._fit, name)


def _uses_feasible_tophat_coordinates(cfg):
    return (cfg["transfer"] == "tophat" and cfg["enforce_nonnegative_transfer"]
            and cfg["fit_transfer_width"])


def _feasible_tophat_lag_bounds(cfg):
    lower = max(cfg["lag_min"], 0.5 * cfg["transfer_width_bounds"][0])
    upper = cfg["lag_max"]
    if lower >= upper:
        raise ValueError("lag bounds leave no non-degenerate non-negative tophat support")
    return float(lower), float(upper)


def _log_width_inside_support(width, maximum):
    """Keep log/exp roundoff from putting an endpoint outside its support."""
    value = math.log(float(width))
    while math.exp(value) > maximum:
        value = float(np.nextafter(value, -np.inf))
    return value


def _joint_tophat_coordinate_setup(start, bounds, names, cfg):
    """Represent the same feasible width interval without an interior penalty.

    The fraction ranges from zero to one, mapping to
    [width_min, min(width_max, 2*lag)]. No transfer formula, prior, physical
    bound, or number of fitted parameters is changed.
    """
    if not (_uses_feasible_tophat_coordinates(cfg) and "lag" in names):
        return start, bounds, names, None
    names = list(names)
    bounds = list(bounds)
    start = np.asarray(start, float).copy()
    lag_index = names.index("lag")
    width_index = names.index("log_transfer_width")
    width_lower, width_upper = cfg["transfer_width_bounds"]
    bounds[lag_index] = _feasible_tophat_lag_bounds(cfg)
    start[lag_index] = np.clip(start[lag_index], *bounds[lag_index])
    maximum = min(width_upper, 2.0 * start[lag_index])
    span = maximum - width_lower
    width = math.exp(float(start[width_index]))
    start[width_index] = np.clip((width - width_lower) / span, 0, 1) if span > 0 else 0.5
    names[width_index] = "transfer_width_fraction"
    bounds[width_index] = (0.0, 1.0)

    def decode(values):
        values = np.asarray(values, float).copy()
        maximum = min(width_upper, 2.0 * values[lag_index])
        width = width_lower + (maximum - width_lower) * values[width_index]
        # Protect exact support endpoints from addition/subtraction roundoff.
        width = np.clip(width, width_lower, maximum)
        values[width_index] = _log_width_inside_support(width, maximum)
        return values

    return start, bounds, names, decode


def _run_minuit(objective, start, bounds, names, max_calls, cfg):
    """Run bounded MIGRAD, with a SciPy Powell fallback when iminuit is absent."""
    start, bounds, names, decode = _joint_tophat_coordinate_setup(
        start, bounds, names, cfg
    )
    if decode is not None:
        model_objective = objective

        def objective(values):
            return model_objective(decode(values))

    start = np.asarray(start, dtype=float)
    start = np.array(
        [np.clip(start[i], bounds[i][0], bounds[i][1]) for i in range(len(bounds))],
        dtype=float,
    )

    if Minuit is None:
        if scipy_minimize is None:
            raise RuntimeError("Install iminuit (preferred) or scipy for optimization")
        result = scipy_minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={"maxfev": int(max_calls), "xtol": 1e-4, "ftol": 1e-4},
        )
        fit = _ScipyMinuitAdapter(result.x, names, result)
        return _FeasibleCoordinateFit(fit, decode, names) if decode is not None else fit

    def positional_objective(*values):
        return objective(np.asarray(values, dtype=float))

    minuit = Minuit(positional_objective, *start, name=tuple(names))
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.strategy = int(cfg["minuit_strategy"])
    minuit.tol = float(cfg["minuit_tol"])
    for name, (lower, upper) in zip(names, bounds):
        minuit.limits[name] = (lower, upper)
        span = upper - lower
        minuit.errors[name] = (
            max(0.02 * span, 1e-3) if np.isfinite(span)
            else max(0.1 * cfg["transfer_width"], 1e-3)
        )
    if decode is not None:
        # Use a physical width scale, independent of the requested lag range.
        minuit.errors["lag"] = max(0.1 * cfg["transfer_width"], 1e-3)
    if "transfer_center" in names:
        width = (math.exp(float(start[names.index("log_transfer_width")]))
                 if "log_transfer_width" in names else cfg["transfer_width"])
        minuit.errors["transfer_center"] = max(0.1 * width, 1e-3)
    minuit.migrad(ncall=int(max_calls))
    return _FeasibleCoordinateFit(minuit, decode, names) if decode is not None else minuit


def _minuit_values(minuit, names):
    decode = getattr(minuit, "_gcp_rm_to_model_x", None)
    if decode is not None:
        values = np.asarray([minuit.values[name]
                             for name in minuit._gcp_rm_optimizer_names], float)
        return decode(values)
    return np.asarray([minuit.values[name] for name in names], dtype=float)


def _optimizer_diagnostics(fit):
    """Preserve convergence diagnostics even when HESSE is disabled."""
    fmin = getattr(fit, "fmin", None)
    return {
        "backend": getattr(fit, "backend", "iminuit_migrad"),
        "converged": bool(getattr(fit, "valid", False)),
        "has_reached_call_limit": bool(getattr(fmin, "has_reached_call_limit", False)),
        "has_parameters_at_limit": bool(getattr(fmin, "has_parameters_at_limit", False)),
        "edm": _finite_float_or_none(getattr(fmin, "edm", None)),
        "n_function_calls": int(getattr(fit, "nfcn", 0)),
        "message": getattr(fit, "message", None),
    }








# -----------------------------------------------------------------------------
# Lag scans and summaries
# -----------------------------------------------------------------------------

def make_lag_grid(lag_min=-10.0, lag_max=100.0, lag_step=1.0):
    """Cover the requested bounds without extrapolating past either endpoint."""
    lag_min, lag_max = _validated_bounds("lag bounds", (lag_min, lag_max))
    lag_step = _finite_number("lag_step", lag_step, 0, strict=True)
    lag_grid = np.arange(lag_min, lag_max, lag_step, dtype=float)
    lag_grid = lag_grid[lag_grid <= lag_max]
    tolerance = 4 * np.spacing(max(abs(lag_min), abs(lag_max)))
    if len(lag_grid) > 1 and lag_max - lag_grid[-1] <= tolerance:
        lag_grid[-1] = lag_max
    else:
        lag_grid = np.r_[lag_grid, lag_max]
    if len(lag_grid) < 2 or np.any(np.diff(lag_grid) <= 0):
        raise ValueError("lag_step must produce distinct representable grid points")
    return lag_grid


def grid_edges(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        raise ValueError("Need at least two grid points")
    e = np.empty(len(x) + 1, dtype=float)
    e[1:-1] = 0.5 * (x[:-1] + x[1:])
    e[0] = x[0] - 0.5 * (x[1] - x[0])
    e[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return e


def grid_cell_widths(x):
    return np.diff(grid_edges(x))


def normalize_log_density(logscore, grid):
    logscore = np.asarray(logscore, float)
    if not np.any(np.isfinite(logscore)):
        return np.full_like(logscore, np.nan, dtype=float)
    q = np.exp(logscore - np.nanmax(logscore))
    area = np.sum(q * grid_cell_widths(grid))
    if not np.isfinite(area) or area <= 0.0:
        return np.full_like(q, np.nan, dtype=float)
    return q / area


def _normalize_scan_density(logscore, grid, cfg):
    """Normalize displayed scan density on the requested finite interval."""
    logscore = np.asarray(logscore, float)
    if not np.any(np.isfinite(logscore)):
        return np.full_like(logscore, np.nan)
    edges = np.clip(grid_edges(grid), cfg["lag_min"], cfg["lag_max"])
    relative = np.exp(logscore - np.nanmax(logscore))
    area = np.sum(relative * np.diff(edges))
    if not np.isfinite(area) or area <= 0:
        return np.full_like(relative, np.nan)
    return relative / area


def scan_lag_fixed(z, sigma_z, t_c, t_l, lag_grid, p_hat, cfg):
    """Lag scan with nuisance parameters fixed at the joint MLE."""
    loglike = np.empty_like(lag_grid, dtype=float)
    params_at_lag = []
    for i, lag in enumerate(lag_grid):
        p = dict(p_hat)
        p["lag"] = float(lag)
        if _is_flexible_cfg(cfg):
            candidates = _flexible_params_at_lag(p_hat, float(lag), cfg)
            if candidates:
                scored = [(_objective_value(candidate, z, sigma_z, t_c, t_l, cfg),
                           candidate) for candidate in candidates]
                score, p = max(scored, key=lambda item: item[0])
            else:
                score = -np.inf
            params_at_lag.append(p)
            loglike[i] = score
            continue
        params_at_lag.append(p)
        loglike[i] = _objective_value(p, z, sigma_z, t_c, t_l, cfg)
    if not np.any(np.isfinite(loglike)):
        raise RuntimeError("fixed scan has no physically feasible finite lag point")
    return loglike, _normalize_scan_density(loglike, lag_grid, cfg), params_at_lag


def scan_lag_profile_local(z, sigma_z, t_c, t_l, lag_grid, p_joint, cfg):
    """Profile tau_drw, rho, and optional width at each lag."""
    # Some grid points can be outside the physical support of a constrained
    # transfer function.  Those points have zero likelihood density; they must
    # not abort the complete profile scan.
    loglike = np.full_like(lag_grid, -np.inf, dtype=float)
    duration = data_duration(t_c, t_l)
    tau0 = np.clip(cfg["default_tau_drw_frac"] * duration, *cfg["tau_drw_bounds"])
    rho0 = min(max(cfg["default_rho"], 1e-8), cfg["rho_max"] * (1.0 - 1e-8))

    default_p = {
        "tau_drw": float(tau0),
        "rho": float(rho0),
        "transfer_width": cfg["transfer_width"],
    }
    default_start = _params_to_nuisance_x(default_p, cfg)
    joint_start = _params_to_nuisance_x(p_joint, cfg)
    prev_start = joint_start.copy()

    center = int(np.argmin(np.abs(lag_grid - p_joint["lag"])))
    order = [center]
    for k in range(1, len(lag_grid)):
        left, right = center - k, center + k
        if left >= 0:
            order.append(left)
        if right < len(lag_grid):
            order.append(right)

    param_by_idx = {}
    x_by_idx = {}
    for count, idx in enumerate(order):
        lag = float(lag_grid[idx])
        starts = [joint_start, default_start]
        neighbors = [j for j in (idx - 1, idx + 1) if j in x_by_idx]
        if neighbors:
            nearest = min(neighbors, key=lambda j: abs(j - idx))
            starts.insert(0, x_by_idx[nearest])
        else:
            starts.insert(0, prev_start)

        try:
            p_best, f_best = optimize_nuisance_local(
                lag, starts, z, sigma_z, t_c, t_l, cfg
            )
            profile_status = "ok"
            profile_message = None
        except ProfileLagInfeasible as exc:
            p_best = dict(p_joint)
            p_best["lag"] = lag
            f_best = -np.inf
            profile_status = "physically_infeasible"
            profile_message = str(exc)
        except Exception as exc:
            # Preserve the rest of the scan and make the failed point explicit.
            # A finite profile can still be summarized, while diagnostics retain
            # the distinction from a physically forbidden lag.
            p_best = dict(p_joint)
            p_best["lag"] = lag
            f_best = -np.inf
            profile_status = "optimization_failed"
            profile_message = repr(exc)
        p_best["profile_status"] = profile_status
        p_best["profile_message"] = profile_message
        loglike[idx] = f_best
        param_by_idx[idx] = p_best
        if np.isfinite(f_best):
            x_by_idx[idx] = _params_to_nuisance_x(p_best, cfg)
            prev_start = x_by_idx[idx]

        if cfg["verbose"] and (count % 20 == 0 or count == len(order) - 1):
            print(
                f"  profiled {count + 1}/{len(order)} lags; "
                f"lag={lag:.4f}, loglike={f_best:.3f}"
            )

    if not np.any(np.isfinite(loglike)):
        raise RuntimeError("profile scan has no physically feasible finite lag point")
    params_at_lag = [param_by_idx[i] for i in range(len(lag_grid))]
    failed = sum(p.get("profile_status") == "optimization_failed" for p in params_at_lag)
    if failed:
        warnings.warn(
            f"profile optimization failed at {failed} lag grid points; inspect "
            "params_at_lag before interpreting the profile",
            RuntimeWarning,
            stacklevel=2,
        )
    return loglike, _normalize_scan_density(loglike, lag_grid, cfg), params_at_lag


def profile_scan_diagnostics(loglike, params_at_lag):
    """Compact diagnostics that distinguish support masking from fit failure."""
    statuses = [p.get("profile_status") for p in params_at_lag]
    return {
        "profile_grid_points": int(len(loglike)),
        "profile_finite_points": int(np.sum(np.isfinite(loglike))),
        "profile_physically_infeasible_points": int(
            sum(status == "physically_infeasible" for status in statuses)
        ),
        "profile_optimization_failed_points": int(
            sum(status == "optimization_failed" for status in statuses)
        ),
    }


def likelihood_ratio_regions(lag_grid, loglike, delta_chi2):
    lag_grid = np.asarray(lag_grid, float)
    loglike = np.asarray(loglike, float)
    if len(lag_grid) < 2 or not np.any(np.isfinite(loglike)):
        return []

    threshold = float(np.nanmax(loglike) - 0.5 * delta_chi2)
    keep = np.isfinite(loglike) & (loglike >= threshold)
    edges = grid_edges(lag_grid)
    regions = []
    i = 0
    while i < len(lag_grid):
        if not keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(lag_grid) and keep[j + 1]:
            j += 1
        regions.append((float(edges[i]), float(edges[j + 1])))
        i = j + 1
    return regions


def likelihood_summary(lag_grid, loglike):
    lag_grid = np.asarray(lag_grid, float)
    loglike = np.asarray(loglike, float)
    if not np.any(np.isfinite(loglike)):
        return {
            "lag_mle": np.nan,
            "max_loglike": -np.inf,
            "lr68_regions": [],
            "lr95_regions": [],
        }
    imax = int(np.nanargmax(loglike))
    return {
        "lag_mle": float(lag_grid[imax]),
        "max_loglike": float(loglike[imax]),
        "delta_loglike": loglike - loglike[imax],
        "lr68_regions": likelihood_ratio_regions(lag_grid, loglike, 1.0),
        "lr95_regions": likelihood_ratio_regions(lag_grid, loglike, 3.84),
    }


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------


def _save_base_outputs(
    prefix,
    lag_grid,
    loglike,
    rel_density,
    p_hat,
    params_at_lag,
    scan_mode,
    summary,
    transform_info,
):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    delta = (
        loglike - np.nanmax(loglike)
        if np.any(np.isfinite(loglike))
        else np.full_like(loglike, np.nan)
    )
    has_width = transfer_has_width(p_hat["transfer"])
    width_at_lag = np.array(
        [p["transfer_width"] for p in params_at_lag], dtype=float
    )
    if not has_width:
        width_at_lag[:] = np.nan
    if len(loglike):
        arr = np.column_stack(
            [
                lag_grid, loglike, delta, rel_density,
                np.array([p["tau_drw"] for p in params_at_lag]),
                np.array([p["rho"] for p in params_at_lag]),
                width_at_lag,
            ]
        )
        np.savetxt(
            f"{prefix}_mle_scan.txt",
            arr,
            header=(
                "lag loglike delta_loglike relative_likelihood_density "
                "tau_drw_at_lag rho_at_lag transfer_width_at_lag"
            ),
            comments="",
        )

    with open(f"{prefix}_mle_params.txt", "w", encoding="utf-8") as f:
        f.write(f"scan_mode {scan_mode}\n")
        f.write("model joint_gaussian_copula_reverberation_mapping\n")
        f.write("gaussianization kde\n")
        f.write(f"transfer {p_hat['transfer']}\n")
        f.write(f"fit_transfer_width {int(summary['fit_transfer_width'])}\n")
        f.write(f"joint_mle_lag {p_hat['lag']}\n")
        f.write(f"joint_mle_tau_drw {p_hat['tau_drw']}\n")
        f.write(f"joint_mle_rho {p_hat['rho']}\n")
        f.write(f"joint_loglike {summary['joint_mle_loglike']}\n")
        f.write(f"joint_objective {summary['joint_objective']}\n")
        f.write(f"fit_score_kind {summary['fit_score_kind']}\n")
        f.write(f"optimizer_converged {int(summary['optimizer_converged'])}\n")
        if has_width:
            f.write(f"joint_mle_transfer_width {p_hat['transfer_width']}\n")
        f.write(f"jitter {p_hat['jitter']}\n")
        f.write(f"continuum_kde_bandwidth {transform_info['c']['bandwidth']}\n")
        f.write(f"line_kde_bandwidth {transform_info['l']['bandwidth']}\n")
        f.write(f"median_sigma_z_continuum {np.median(transform_info['c']['sigma_z'])}\n")
        f.write(f"median_sigma_z_line {np.median(transform_info['l']['sigma_z'])}\n")

    with open(f"{prefix}_mle_summary.txt", "w", encoding="utf-8") as f:
        for key, val in summary.items():
            if key == "delta_loglike":
                continue
            f.write(f"{key} {val}\n")


def finish_figure(fig, filename=None, save=True, show=False, dpi=180):
    import matplotlib.pyplot as plt
    if save and filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def make_plots(
    prefix,
    data,
    z_c,
    z_l,
    sigma_z_c,
    sigma_z_l,
    lag_grid,
    loglike,
    rel_density,
    p_hat,
    save=True,
    show=False,
    score_kind="data_loglike",
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is a required dependency of pygcp-rm; "
            "install matplotlib or reinstall pygcp-rm with dependencies"
        ) from exc
    if not (save or show):
        return

    t_c, t_l = data["t_c"], data["t_l"]
    c_flux, l_flux = data["c_flux"], data["l_flux"]
    c_err, l_err = data["c_err"], data["l_err"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(t_c, c_flux, yerr=c_err, fmt="o", ms=4, lw=0.8, capsize=2, label="continuum")
    ax.errorbar(t_l, l_flux, yerr=l_err, fmt="o", ms=4, lw=0.8, capsize=2, label="line")
    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    ax.set_title("Observed light curves with flux errors")
    ax.legend()
    fig.tight_layout()
    finish_figure(fig, f"{prefix}_mle_lightcurves.png", save=save, show=show)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(t_c, z_c, yerr=sigma_z_c, fmt="o", ms=4, lw=0.8, capsize=2, label="continuum score")
    ax.errorbar(t_l, z_l, yerr=sigma_z_l, fmt="o", ms=4, lw=0.8, capsize=2, label="line score")
    ax.set_xlabel("time")
    ax.set_ylabel("Gaussian score")
    ax.set_title("KDE Gaussian scores with propagated errors")
    ax.legend()
    fig.tight_layout()
    finish_figure(fig, f"{prefix}_mle_scores_with_errors.png", save=save, show=show)

    if not len(loglike):
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    score_label = "likelihood" if score_kind == "data_loglike" else "objective"
    ax.plot(lag_grid, rel_density, lw=2, label=f"relative {score_label} density")
    ax.axvline(p_hat["lag"], ls=":", lw=1.5, label=f"joint fit lag = {p_hat['lag']:.3f}")
    ax.set_xlabel("lag")
    ax.set_ylabel(f"normalized relative {score_label}")
    ax.legend(loc="upper left", fontsize=8)
    title = f"Error-aware lag scan: {p_hat['transfer']}"
    if transfer_has_width(p_hat["transfer"]):
        title += f", width={p_hat['transfer_width']:.3g}"
    ax.set_title(title)
    fig.tight_layout()
    finish_figure(fig, f"{prefix}_mle_lag_scan.png", save=save, show=show)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    delta = loglike - np.nanmax(loglike)
    ax.plot(lag_grid, -2.0 * delta, lw=2)
    if score_kind == "data_loglike":
        ax.axhline(1.0, color="0.4", ls="dashed", lw=1.0, label="Delta chi2 = 1")
        ax.axhline(3.84, color="0.6", ls=":", lw=1.0, label="Delta chi2 = 3.84")
        ax.legend(fontsize=8)
    ax.set_xlabel("lag")
    ax.set_ylabel(f"-2 Delta {score_label}")
    ax.set_title("Likelihood-ratio profile" if score_kind == "data_loglike"
                 else "Penalized-objective profile")
    fig.tight_layout()
    finish_figure(fig, f"{prefix}_mle_likelihood_ratio.png", save=save, show=show)




# -----------------------------------------------------------------------------
# Internal data preparation
# -----------------------------------------------------------------------------

def _prepare_mle_data(
    continuum,
    line,
    kde_bandwidth=None,
    sigma_z_floor=1e-8,
    sigma_z_max=None,
):
    """Load and Gaussianize the two light curves once for reuse across models."""
    data = load_data(continuum, line)
    c_trans = kde_gaussian_transform_with_error(
        data["c_flux"],
        data["c_err"],
        bandwidth=kde_bandwidth,
        sigma_z_floor=sigma_z_floor,
        sigma_z_max=sigma_z_max,
    )
    l_trans = kde_gaussian_transform_with_error(
        data["l_flux"],
        data["l_err"],
        bandwidth=kde_bandwidth,
        sigma_z_floor=sigma_z_floor,
        sigma_z_max=sigma_z_max,
    )
    z_c, z_l = c_trans["z"], l_trans["z"]
    sigma_z_c, sigma_z_l = c_trans["sigma_z"], l_trans["sigma_z"]
    return {
        "data": data,
        "c_trans": c_trans,
        "l_trans": l_trans,
        "z_c": z_c,
        "z_l": z_l,
        "sigma_z_c": sigma_z_c,
        "sigma_z_l": sigma_z_l,
        "z": np.r_[z_c, z_l],
        "sigma_z": np.r_[sigma_z_c, sigma_z_l],
    }


# -----------------------------------------------------------------------------
# Configuration and one-call entry point
# -----------------------------------------------------------------------------

def build_config(
    lag_min=-10.0,
    lag_max=100.0,
    lag_step=1.0,
    rho_max=1.0,
    jitter=1e-8,
    cholesky_jitter=1e-8,
    tau_drw_bounds=(0.1, 1000.0),
    rho_logit_bounds=(-8.0, 12.0),
    default_tau_drw_frac=0.25,
    default_rho=0.70,
    transfer="tophat",
    transfer_width=None,
    fit_transfer_width=True,
    transfer_width_bounds=(0.02, 20.0),
    transfer_grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    flexible_sigma_ratio=0.45,
    flexible_smoothness=0.25,
    flexible_weight_logit_bound=5.0,
    flexible_initial_weights=(0.25, 0.50, 0.25),
    enforce_nonnegative_transfer=True,
    use_prior=False,
    log_tau_prior_sd=1.0,
    tau_drw_prior_frac=0.25,
    rho_prior_a=2.0,
    rho_prior_b=1.5,
    transfer_width_prior=None,
    log_width_prior_sd=0.7,
    scan_mode="profile-local",
    minuit_max_calls=2000,
    minuit_n_starts=10,
    minuit_tol=0.1,
    minuit_strategy=1,
    profile_max_calls=500,
    sigma_z_floor=1e-8,
    sigma_z_max=None,
    kde_bandwidth=None,
    verbose=False,
):
    """Validate fitting settings before any data processing or optimization."""
    lag_min, lag_max = _validated_bounds("lag bounds", (lag_min, lag_max))
    lag_step = _finite_number("lag_step", lag_step, 0, strict=True)
    rho_max = _finite_number("rho_max", rho_max, 0, strict=True)
    if rho_max > 1.0:
        raise ValueError("rho_max must satisfy 0 < rho_max <= 1")
    tau_drw_bounds = _validated_bounds("tau_drw_bounds", tau_drw_bounds, positive=True)
    rho_logit_bounds = _validated_bounds("rho_logit_bounds", rho_logit_bounds)
    jitter = _finite_number("jitter", jitter, 0)
    cholesky_jitter = _finite_number("cholesky_jitter", cholesky_jitter, 0)
    default_tau_drw_frac = _finite_number(
        "default_tau_drw_frac", default_tau_drw_frac, 0, strict=True
    )
    default_rho = _finite_number("default_rho", default_rho, 0, strict=True)
    if default_rho >= 1.0:
        raise ValueError("default_rho must satisfy 0 < default_rho < 1")
    for name, value in (
        ("log_tau_prior_sd", log_tau_prior_sd),
        ("tau_drw_prior_frac", tau_drw_prior_frac),
        ("rho_prior_a", rho_prior_a),
        ("rho_prior_b", rho_prior_b),
        ("log_width_prior_sd", log_width_prior_sd),
    ):
        _finite_number(name, value, 0, strict=True)
    if kde_bandwidth is not None:
        kde_bandwidth = _finite_number("kde_bandwidth", kde_bandwidth, 0, strict=True)
    if sigma_z_floor is not None:
        sigma_z_floor = _finite_number("sigma_z_floor", sigma_z_floor, 0)
    if sigma_z_max is not None:
        sigma_z_max = _finite_number("sigma_z_max", sigma_z_max, 0, strict=True)
        if sigma_z_floor is not None and sigma_z_max < sigma_z_floor:
            raise ValueError("sigma_z_max must be >= sigma_z_floor")
    transfer_grid_size = _positive_integer("transfer_grid_size", transfer_grid_size, 9)
    minuit_max_calls = _positive_integer("minuit_max_calls", minuit_max_calls)
    minuit_n_starts = _positive_integer("minuit_n_starts", minuit_n_starts)
    profile_max_calls = _positive_integer("profile_max_calls", profile_max_calls)
    minuit_tol = _finite_number("minuit_tol", minuit_tol, 0, strict=True)
    flexible_sigma_ratio = _finite_number(
        "flexible_sigma_ratio", flexible_sigma_ratio, 0, strict=True
    )
    flexible_smoothness = _finite_number("flexible_smoothness", flexible_smoothness, 0)
    flexible_weight_logit_bound = _finite_number(
        "flexible_weight_logit_bound", flexible_weight_logit_bound, 0, strict=True
    )
    long_tail_width_ratio = _finite_number(
        "long_tail_width_ratio", long_tail_width_ratio, 1, strict=True
    )
    long_tail_scale_ratio = _finite_number(
        "long_tail_scale_ratio", long_tail_scale_ratio, 0, strict=True
    )
    double_exp_decay_ratio = _finite_number(
        "double_exp_decay_ratio", double_exp_decay_ratio, 0, strict=True
    )
    if scan_mode not in ("fixed", "profile-local"):
        raise ValueError("scan_mode must be 'fixed' or 'profile-local'")
    transfer = normalize_transfer_name(transfer)
    width_lower, width_upper = _validated_bounds(
        "transfer_width_bounds", transfer_width_bounds, positive=True
    )
    width_product = width_lower * width_upper
    automatic_width = (
        math.sqrt(width_product)
        if math.isfinite(width_product) and width_product > 0
        else math.exp(0.5 * (math.log(width_lower) + math.log(width_upper)))
    )
    if not transfer_has_width(transfer):
        if transfer_width is not None:
            raise ValueError(f"{transfer} does not use transfer_width")
        fit_transfer_width = False
    elif transfer_width is None:
        if not fit_transfer_width:
            raise ValueError(
                "transfer_width is required when fit_transfer_width=False"
            )
        transfer_width = automatic_width
    if transfer_width is not None:
        transfer_width = _finite_number("transfer_width", transfer_width, 0, strict=True)
    if transfer_has_width(transfer) and transfer_width <= 0.0:
        raise ValueError("finite-width transfer models require transfer_width>0")
    if transfer_width_prior is not None:
        transfer_width_prior = _finite_number(
            "transfer_width_prior", transfer_width_prior, 0, strict=True
        )
    flexible_initial_weights = _parse_flexible_initial_weights(
        flexible_initial_weights
    )
    if minuit_strategy not in (0, 1, 2):
        raise ValueError("minuit_strategy must be 0, 1, or 2")

    return {
        "lag_min": float(lag_min),
        "lag_max": float(lag_max),
        "lag_step": float(lag_step),
        "rho_max": float(rho_max),
        "jitter": float(jitter),
        "cholesky_jitter": float(cholesky_jitter),
        "tau_drw_bounds": tuple(float(x) for x in tau_drw_bounds),
        "rho_logit_bounds": tuple(float(x) for x in rho_logit_bounds),
        "default_tau_drw_frac": float(default_tau_drw_frac),
        "default_rho": float(default_rho),
        "transfer": transfer,
        "transfer_width": transfer_width,
        "fit_transfer_width": bool(fit_transfer_width),
        "transfer_width_bounds": (width_lower, width_upper),
        "transfer_grid_size": int(transfer_grid_size),
        "long_tail_width_ratio": float(long_tail_width_ratio),
        "long_tail_scale_ratio": float(long_tail_scale_ratio),
        "double_exp_decay_ratio": float(double_exp_decay_ratio),
        "enforce_nonnegative_transfer": bool(enforce_nonnegative_transfer),
        "use_prior": bool(use_prior),
        "log_tau_prior_sd": float(log_tau_prior_sd),
        "tau_drw_prior_frac": float(tau_drw_prior_frac),
        "rho_prior_a": float(rho_prior_a),
        "rho_prior_b": float(rho_prior_b),
        "transfer_width_prior": transfer_width_prior,
        "log_width_prior_sd": float(log_width_prior_sd),
        "flexible_sigma_ratio": float(flexible_sigma_ratio),
        "flexible_smoothness": float(flexible_smoothness),
        "flexible_weight_logit_bounds": (
            -float(flexible_weight_logit_bound),
            float(flexible_weight_logit_bound),
        ),
        "flexible_initial_weights": flexible_initial_weights.tolist(),
        "likelihood_mode": "joint_gaussian_copula",
        "total_duration": None,
        "scan_mode": scan_mode,
        "optimizer": "iminuit_migrad" if Minuit is not None else "scipy_powell_fallback",
        "minuit_max_calls": int(minuit_max_calls),
        "minuit_n_starts": int(minuit_n_starts),
        "minuit_tol": float(minuit_tol),
        "minuit_strategy": int(minuit_strategy),
        "profile_max_calls": int(profile_max_calls),
        "sigma_z_floor": None if sigma_z_floor is None else float(sigma_z_floor),
        "sigma_z_max": None if sigma_z_max is None else float(sigma_z_max),
        "kde_bandwidth": kde_bandwidth,
        "verbose": bool(verbose),
    }


def run_mle(
    continuum,
    line,
    lag_min=-10.0,
    lag_max=100.0,
    lag_step=1.0,
    rho_max=1.0,
    jitter=1e-8,
    cholesky_jitter=1e-8,
    tau_drw_bounds=(0.1, 1000.0),
    rho_logit_bounds=(-8.0, 12.0),
    default_tau_drw_frac=0.25,
    default_rho=0.70,
    transfer="tophat",
    transfer_width=None,
    fit_transfer_width=True,
    transfer_width_bounds=(0.02, 20.0),
    transfer_grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    flexible_sigma_ratio=0.45,
    flexible_smoothness=0.25,
    flexible_weight_logit_bound=5.0,
    flexible_initial_weights=(0.25, 0.50, 0.25),
    enforce_nonnegative_transfer=True,
    use_prior=False,
    log_tau_prior_sd=1.0,
    tau_drw_prior_frac=0.25,
    rho_prior_a=2.0,
    rho_prior_b=1.5,
    transfer_width_prior=None,
    log_width_prior_sd=0.7,
    scan_mode="profile-local",
    minuit_max_calls=2000,
    minuit_n_starts=10,
    minuit_tol=0.1,
    minuit_strategy=1,
    profile_max_calls=500,
    sigma_z_floor=1e-8,
    sigma_z_max=None,
    kde_bandwidth=None,
    compute_hesse_errors=False,
    run_scan=True,
    save=False,
    make_figures=False,
    show_figures=False,
    output_prefix="gcp_rm",
    verbose=False,
):
    """Run the error-aware, optionally convolved GCP-RM MLE pipeline.

    Transfer support is constrained to non-negative delays by default. Set
    ``enforce_nonnegative_transfer=False`` to allow negative-delay support.
    """
    cfg = build_config(
        lag_min=lag_min,
        lag_max=lag_max,
        lag_step=lag_step,
        rho_max=rho_max,
        jitter=jitter,
        cholesky_jitter=cholesky_jitter,
        tau_drw_bounds=tau_drw_bounds,
        rho_logit_bounds=rho_logit_bounds,
        default_tau_drw_frac=default_tau_drw_frac,
        default_rho=default_rho,
        transfer=transfer,
        transfer_width=transfer_width,
        fit_transfer_width=fit_transfer_width,
        transfer_width_bounds=transfer_width_bounds,
        transfer_grid_size=transfer_grid_size,
        long_tail_width_ratio=long_tail_width_ratio,
        long_tail_scale_ratio=long_tail_scale_ratio,
        double_exp_decay_ratio=double_exp_decay_ratio,
        flexible_sigma_ratio=flexible_sigma_ratio,
        flexible_smoothness=flexible_smoothness,
        flexible_weight_logit_bound=flexible_weight_logit_bound,
        flexible_initial_weights=flexible_initial_weights,
        enforce_nonnegative_transfer=enforce_nonnegative_transfer,
        use_prior=use_prior,
        log_tau_prior_sd=log_tau_prior_sd,
        tau_drw_prior_frac=tau_drw_prior_frac,
        rho_prior_a=rho_prior_a,
        rho_prior_b=rho_prior_b,
        transfer_width_prior=transfer_width_prior,
        log_width_prior_sd=log_width_prior_sd,
        scan_mode=scan_mode,
        minuit_max_calls=minuit_max_calls,
        minuit_n_starts=minuit_n_starts,
        minuit_tol=minuit_tol,
        minuit_strategy=minuit_strategy,
        profile_max_calls=profile_max_calls,
        sigma_z_floor=sigma_z_floor,
        sigma_z_max=sigma_z_max,
        kde_bandwidth=kde_bandwidth,
        verbose=verbose,
    )

    if cfg["transfer"] == "flexible_gaussian3":
        cfg["flexible_numerics"] = "continuous_gaussian_response_v1"
        cfg["flexible_lag_definition"] = "continuous_response_centroid"

    prepared = _prepare_mle_data(
        continuum,
        line,
        kde_bandwidth=kde_bandwidth,
        sigma_z_floor=sigma_z_floor,
        sigma_z_max=sigma_z_max,
    )
    data = prepared["data"]
    c_trans = prepared["c_trans"]
    l_trans = prepared["l_trans"]
    z_c, z_l = prepared["z_c"], prepared["z_l"]
    sigma_z_c, sigma_z_l = prepared["sigma_z_c"], prepared["sigma_z_l"]
    z = prepared["z"]
    sigma_z = prepared["sigma_z"]
    cfg["total_duration"] = data_duration(data["t_c"], data["t_l"])
    lag_grid = (
        make_lag_grid(cfg["lag_min"], cfg["lag_max"], cfg["lag_step"])
        if run_scan else np.array([], dtype=float)
    )

    p_hat, joint_objective, minuit_result = get_optimal_x(
        z, sigma_z, data["t_c"], data["t_l"], cfg
    )
    optimizer_diagnostics = _optimizer_diagnostics(minuit_result)
    joint_loglike = _data_loglike_for_params(
        p_hat, z, sigma_z, data["t_c"], data["t_l"], cfg
    )
    parameter_uncertainties = estimate_hesse_parameter_uncertainties(
        minuit_result,
        p_hat,
        cfg,
        enabled=bool(compute_hesse_errors),
    )

    if run_scan:
        if scan_mode == "fixed":
            loglike, rel_density, params_at_lag = scan_lag_fixed(
                z, sigma_z, data["t_c"], data["t_l"], lag_grid, p_hat, cfg
            )
        else:
            loglike, rel_density, params_at_lag = scan_lag_profile_local(
                z, sigma_z, data["t_c"], data["t_l"], lag_grid, p_hat, cfg
            )
        summary = _summarize_scan(lag_grid, loglike, cfg)
        summary.update(profile_scan_diagnostics(loglike, params_at_lag))
    else:
        loglike = np.array([], dtype=float)
        rel_density = np.array([], dtype=float)
        params_at_lag = []
        summary = {
            "lag_mle": None,
            "max_loglike": None,
            "delta_loglike": np.array([], dtype=float),
            "lr68_regions": [],
            "lr95_regions": [],
            "scan_score_kind": _fit_score_kind(cfg),
        }
    summary.update(
        {
            "joint_mle_lag": p_hat["lag"],
            "joint_mle_loglike": joint_loglike,
            "joint_objective": joint_objective,
            "fit_score_kind": _fit_score_kind(cfg),
            "optimizer_converged": optimizer_diagnostics["converged"],
            "optimizer_diagnostics": optimizer_diagnostics,
            "joint_mle_tau_drw": p_hat["tau_drw"],
            "joint_mle_rho": p_hat["rho"],
            "joint_mle_transfer_width": _reported_transfer_width(p_hat),
            "joint_mle_lag_error": physical_parameter_error(
                parameter_uncertainties, "lag"
            ),
            "joint_mle_tau_drw_error": physical_parameter_error(
                parameter_uncertainties, "tau_drw"
            ),
            "joint_mle_rho_error": physical_parameter_error(
                parameter_uncertainties, "rho"
            ),
            "joint_mle_transfer_width_error": physical_parameter_error(
                parameter_uncertainties, "transfer_width"
            ),
            "hesse_available": bool(parameter_uncertainties["available"]),
            "hesse_status": parameter_uncertainties["status"],
            "hesse_valid": parameter_uncertainties["diagnostics"]["is_valid"],
            "hesse_accurate_covariance": parameter_uncertainties["diagnostics"][
                "has_accurate_covar"
            ],
            "hesse_positive_definite": parameter_uncertainties["diagnostics"][
                "has_posdef_covar"
            ],
            "hesse_parameters_at_limit": parameter_uncertainties["diagnostics"][
                "has_parameters_at_limit"
            ],
            "hesse_edm": parameter_uncertainties["diagnostics"]["edm"],
            "transfer": p_hat["transfer"],
            "fit_transfer_width": bool(cfg["fit_transfer_width"]),
            "median_sigma_z_continuum": float(np.median(sigma_z_c)),
            "median_sigma_z_line": float(np.median(sigma_z_l)),
        }
    )
    _add_flexible_uncertainty_errors(summary, parameter_uncertainties, p_hat)

    transform_info = {"c": c_trans, "l": l_trans}
    result = {
        "config": cfg,
        "data": data,
        "transform_info": transform_info,
        "z_c": z_c,
        "z_l": z_l,
        "z": z,
        "sigma_z_c": sigma_z_c,
        "sigma_z_l": sigma_z_l,
        "sigma_z": sigma_z,
        "lag_grid": lag_grid,
        "p_hat": p_hat,
        "joint_loglike": joint_loglike,
        "joint_objective": joint_objective,
        "fit_score_kind": _fit_score_kind(cfg),
        "optimizer_diagnostics": optimizer_diagnostics,
        "minuit_result": minuit_result,
        "parameter_uncertainties": parameter_uncertainties,
        "loglike": loglike,
        "relative_likelihood_density": rel_density,
        "params_at_lag": params_at_lag,
        "summary": summary,
        "run_scan": bool(run_scan),
    }

    if save:
        uncertainty_path = Path(
            f"{output_prefix}_parameter_uncertainties.json"
        )
        uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
        uncertainty_path.write_text(
            json.dumps(parameter_uncertainties, indent=2),
            encoding="utf-8",
        )

    if save:
        save_outputs(
            output_prefix,
            lag_grid if run_scan else np.array([], dtype=float),
            loglike,
            rel_density,
            p_hat,
            params_at_lag,
            scan_mode,
            summary,
            transform_info,
        )

    if make_figures:
        make_plots(
            output_prefix,
            data,
            z_c,
            z_l,
            sigma_z_c,
            sigma_z_l,
            lag_grid if run_scan else np.array([], dtype=float),
            loglike,
            rel_density,
            p_hat,
            save=True,
            show=show_figures,
            score_kind=_fit_score_kind(cfg),
        )

    if verbose:
        print(format_report(result))
    return result



# -----------------------------------------------------------------------------
# Contiguous time-block predictive validation for transfer-model selection
# -----------------------------------------------------------------------------

def conditional_gaussian_copula_logscore_from_R(
    z,
    R,
    train_indices,
    test_indices,
    cholesky_jitter=1e-8,
    jitter_growth=10.0,
    max_tries=8,
):
    """Conditional copula log score of held-out variables given training data.

    For Gaussian scores, the conditional Gaussian density is evaluated and the
    standard-normal marginal log densities of the held-out variables are
    subtracted. The result is therefore a conditional Gaussian-copula score,
    consistent with ``gaussian_copula_loglike_from_R``.
    """
    z = np.asarray(z, dtype=float)
    R = np.asarray(R, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if train_indices.size == 0 or test_indices.size == 0:
        return -np.inf
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("training and test indices must be disjoint")

    z_train = z[train_indices]
    z_test = z[test_indices]
    R_tt = R[np.ix_(train_indices, train_indices)]
    R_ht = R[np.ix_(test_indices, train_indices)]
    R_hh = R[np.ix_(test_indices, test_indices)]

    train_eye = np.eye(len(train_indices))
    jitter = float(cholesky_jitter)
    L_train = None
    for _ in range(int(max_tries)):
        try:
            L_train = np.linalg.cholesky(R_tt + jitter * train_eye)
            break
        except np.linalg.LinAlgError:
            jitter *= float(jitter_growth)
    if L_train is None:
        return -np.inf

    alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, z_train))
    conditional_mean = R_ht @ alpha
    projected = np.linalg.solve(L_train, R_ht.T)
    conditional_cov = R_hh - projected.T @ projected
    conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)

    test_eye = np.eye(len(test_indices))
    test_jitter = float(cholesky_jitter)
    L_test = None
    for _ in range(int(max_tries)):
        try:
            L_test = np.linalg.cholesky(conditional_cov + test_jitter * test_eye)
            break
        except np.linalg.LinAlgError:
            test_jitter *= float(jitter_growth)
    if L_test is None:
        return -np.inf

    residual = z_test - conditional_mean
    whitened = np.linalg.solve(L_test, residual)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L_test))))
    n_test = len(test_indices)
    conditional_normal_loglike = float(
        -0.5 * (whitened @ whitened + logdet + n_test * math.log(2.0 * math.pi))
    )
    standard_normal_loglike = float(
        np.sum(-0.5 * z_test**2 - 0.5 * math.log(2.0 * math.pi))
    )
    score = conditional_normal_loglike - standard_normal_loglike
    return float(score) if np.isfinite(score) else -np.inf


def make_contiguous_line_folds(n_line, n_folds=5, gap_points=0, min_train_line=10):
    """Split sorted line observations into contiguous held-out time blocks."""
    n_line = int(n_line)
    n_folds = int(n_folds)
    gap_points = int(gap_points)
    min_train_line = int(min_train_line)
    if n_folds < 2:
        raise ValueError("blocked CV needs at least 2 folds")
    if n_line < 6:
        raise ValueError("blocked CV needs at least 6 line observations")
    if gap_points < 0:
        raise ValueError("cv_gap_points must be non-negative")
    if min_train_line < 5:
        min_train_line = 5
    if n_folds > n_line:
        n_folds = n_line

    all_indices = np.arange(n_line, dtype=int)
    folds = []
    for fold_index, test_indices in enumerate(np.array_split(all_indices, n_folds)):
        test_indices = np.asarray(test_indices, dtype=int)
        if test_indices.size == 0:
            continue
        excluded = np.zeros(n_line, dtype=bool)
        left = max(0, int(test_indices[0]) - gap_points)
        right = min(n_line, int(test_indices[-1]) + gap_points + 1)
        excluded[left:right] = True
        train_indices = all_indices[~excluded]
        if len(train_indices) < min_train_line:
            raise ValueError(
                f"fold {fold_index} leaves only {len(train_indices)} training line points; "
                f"need at least {min_train_line}"
            )
        folds.append(
            {
                "fold": int(fold_index),
                "train_indices": train_indices,
                "test_indices": test_indices,
                "excluded_start": int(left),
                "excluded_stop": int(right),
            }
        )
    if len(folds) < 2:
        raise ValueError("blocked CV produced fewer than 2 usable folds")
    return folds




# -----------------------------------------------------------------------------
# Internal consistency checks
# -----------------------------------------------------------------------------

def kernel_self_check():
    """Return PSD/symmetry diagnostics for every supported transfer model."""
    t_c = np.array([0.0, 1.1, 2.7, 5.0])
    t_l = np.array([0.4, 1.8, 3.3, 6.2])
    tau = 20.0
    lag = 2.4
    width = 0.6
    out = {}
    for transfer in SUPPORTED_TRANSFERS:
        K = build_reverberation_covariance(
            t_c,
            t_l,
            lag,
            tau,
            0.8,
            transfer=transfer,
            transfer_width=width,
            transfer_grid_size=61,
            enforce_nonnegative_transfer=False,
        )
        eig = np.linalg.eigvalsh(K)
        out[transfer] = {
            "max_symmetry_error": float(np.max(np.abs(K - K.T))),
            "min_eigenvalue": float(np.min(eig)),
            "max_diagonal_error": float(np.max(np.abs(np.diag(K) - 1.0))),
        }
    K_delta = build_reverberation_covariance(
        t_c, t_l, lag, tau, 0.8, transfer="delta", transfer_width=0.0
    )
    K_small = build_reverberation_covariance(
        t_c, t_l, lag, tau, 0.8, transfer="tophat", transfer_width=1e-6
    )
    out["tophat_delta_limit"] = {
        "max_abs_error": float(np.max(np.abs(K_delta - K_small)))
    }
    return out






def _fit_transfer_worker(payload):
    """ProcessPool worker; payload contains arrays/paths plus run_mle keyword args."""
    transfer = payload["transfer"]
    kwargs = dict(payload["kwargs"])
    kwargs.update(
        transfer=transfer,
        run_scan=bool(payload.get("run_scan", False)),
        save=False,
        make_figures=False,
        show_figures=False,
        verbose=False,
    )
    result = run_mle(
        payload["continuum"],
        payload["line"],
        **kwargs,
    )
    if (payload.get("selection_metric", "aic") == "blocked_cv"
            and result["fit_score_kind"] != "data_loglike"):
        # Predictive validation scores the requested model, including smoothing.
        # No extra likelihood-only fit is needed when AIC/BIC are not requested.
        criteria = {
            "transfer": transfer,
            "likelihood_mode": "joint_gaussian_copula",
            "n_parameters": 3 + transfer_parameter_count(
                transfer, result["config"]["fit_transfer_width"]
            ),
            "n_observations": len(result["z"]),
            "data_loglike_at_fit": float(result["joint_loglike"]),
            "max_data_loglike": None,
            "aic": None, "aicc": None, "bic": None,
            "information_criteria_status": "not_requested_penalized_fit",
            "information_criteria_refit": False,
        }
    else:
        criteria = model_information_criteria(result)
    cv_config = dict(payload.get("cv_config") or {})
    if cv_config.get("enabled", False):
        try:
            cv_result = blocked_time_predictive_validation(
                payload["continuum"],
                payload["line"],
                transfer=transfer,
                run_mle_kwargs=kwargs,
                n_folds=cv_config["n_folds"],
                gap_points=cv_config["gap_points"],
                min_train_line=cv_config["min_train_line"],
                cv_minuit_max_calls=cv_config.get("minuit_max_calls"),
                cv_minuit_n_starts=cv_config.get("minuit_n_starts"),
            )
            criteria.update(cv_result)
            criteria["blocked_cv_error"] = None
        except Exception as exc:
            criteria.update(
                blocked_cv_logscore=float("-inf"),
                blocked_cv_mean_logscore=float("-inf"),
                blocked_cv_n_test=0,
                blocked_cv_n_folds=0,
                blocked_cv_failed_folds=int(cv_config.get("n_folds", 0)),
                blocked_cv_folds=[],
                blocked_cv_error=repr(exc),
            )
    # iminuit objects are not guaranteed to be pickleable across processes.
    result["minuit_result"] = None
    return transfer, result, criteria


def _scan_existing_result(result, scan_mode="fixed"):
    """Add a lag scan to a stage-1 joint-MLE result without refitting the joint MLE.

    Parameters
    ----------
    result : dict
        A ``run_mle(..., run_scan=False)`` result.  It already contains the
        Gaussianized data, covariance configuration, and joint-MLE nuisance
        parameters.
    scan_mode : {"fixed", "profile-local"}
        ``fixed`` keeps tau_drw, rho, and transfer width at their stage-1 joint
        MLE values.  ``profile-local`` re-optimizes only the nuisance parameters
        at each lag, but still reuses the stage-1 joint MLE as initialization.

    This is the key speed optimization for model selection: the expensive
    candidate-model joint fits are not repeated before the lag-density scan.
    """
    if scan_mode not in {"fixed", "profile-local"}:
        raise ValueError("scan_mode must be fixed or profile-local")

    out = copy.deepcopy(result)
    cfg = dict(out["config"])
    cfg["scan_mode"] = scan_mode
    out["config"] = cfg

    lag_grid = make_lag_grid(cfg["lag_min"], cfg["lag_max"], cfg["lag_step"])
    out["lag_grid"] = lag_grid
    z = np.asarray(out["z"], dtype=float)
    sigma_z = np.asarray(out["sigma_z"], dtype=float)
    t_c = np.asarray(out["data"]["t_c"], dtype=float)
    t_l = np.asarray(out["data"]["t_l"], dtype=float)
    p_hat = dict(out["p_hat"])

    if scan_mode == "fixed":
        loglike, rel_density, params_at_lag = scan_lag_fixed(
            z, sigma_z, t_c, t_l, lag_grid, p_hat, cfg
        )
    else:
        loglike, rel_density, params_at_lag = scan_lag_profile_local(
            z, sigma_z, t_c, t_l, lag_grid, p_hat, cfg
        )

    summary = _summarize_scan(lag_grid, loglike, cfg)
    summary.update(profile_scan_diagnostics(loglike, params_at_lag))
    parameter_uncertainties = out.get("parameter_uncertainties") or {}
    summary.update(
        {
            "joint_mle_lag": p_hat["lag"],
            "joint_mle_loglike": out["joint_loglike"],
            "joint_objective": out["joint_objective"],
            "fit_score_kind": out["fit_score_kind"],
            "optimizer_converged": out["optimizer_diagnostics"]["converged"],
            "optimizer_diagnostics": out["optimizer_diagnostics"],
            "joint_mle_tau_drw": p_hat["tau_drw"],
            "joint_mle_rho": p_hat["rho"],
            "joint_mle_transfer_width": _reported_transfer_width(p_hat),
            "joint_mle_lag_error": physical_parameter_error(
                parameter_uncertainties, "lag"
            ),
            "joint_mle_tau_drw_error": physical_parameter_error(
                parameter_uncertainties, "tau_drw"
            ),
            "joint_mle_rho_error": physical_parameter_error(
                parameter_uncertainties, "rho"
            ),
            "joint_mle_transfer_width_error": physical_parameter_error(
                parameter_uncertainties, "transfer_width"
            ),
            "hesse_available": bool(parameter_uncertainties.get("available", False)),
            "hesse_status": parameter_uncertainties.get("status"),
            "hesse_valid": (parameter_uncertainties.get("diagnostics") or {}).get(
                "is_valid"
            ),
            "hesse_accurate_covariance": (
                parameter_uncertainties.get("diagnostics") or {}
            ).get("has_accurate_covar"),
            "hesse_positive_definite": (
                parameter_uncertainties.get("diagnostics") or {}
            ).get("has_posdef_covar"),
            "hesse_parameters_at_limit": (
                parameter_uncertainties.get("diagnostics") or {}
            ).get("has_parameters_at_limit"),
            "hesse_edm": (parameter_uncertainties.get("diagnostics") or {}).get(
                "edm"
            ),
            "transfer": p_hat["transfer"],
            "fit_transfer_width": bool(cfg["fit_transfer_width"]),
            "median_sigma_z_continuum": float(np.median(out["sigma_z_c"])),
            "median_sigma_z_line": float(np.median(out["sigma_z_l"])),
            "scan_reused_joint_mle": True,
            "scan_mode": scan_mode,
        }
    )
    _add_flexible_uncertainty_errors(summary, parameter_uncertainties, p_hat)

    out["loglike"] = loglike
    out["relative_likelihood_density"] = rel_density
    out["params_at_lag"] = params_at_lag
    out["summary"] = summary
    out["run_scan"] = True
    out["scan_reused_joint_mle"] = True
    out["minuit_result"] = None
    return out


def _scan_existing_result_worker(payload):
    """ProcessPool worker for lag scans that reuse stage-1 joint-MLE results."""
    result = _scan_existing_result(payload["result"], payload["scan_mode"])
    return result["p_hat"]["transfer"], result







def run_transfer_model_selection(
    continuum,
    line,
    candidate_transfers=None,
    selection_metric="blocked_cv",
    delta_keep=6.0,
    cv_folds=5,
    cv_gap_points=0,
    cv_min_train_line=10,
    cv_minuit_max_calls=1000,
    cv_minuit_n_starts=3,
    scan_models="best",
    parallel=True,
    max_workers=None,
    save=True,
    make_figures=False,
    output_prefix="gcp_rm_auto",
    **run_mle_kwargs,
):
    """Fit and rank transfer-covariance models, then return the best model.

    ``scan_models`` is one of ``none``, ``best``, ``supported``, or ``all``.
    Model selection uses either a classical information criterion or contiguous
    held-out line-block predictive likelihood; no transfer-model averaging is
    performed.
    """
    metric = str(selection_metric).lower()
    if metric not in {"aic", "aicc", "bic", "blocked_cv"}:
        raise ValueError("selection_metric must be aic, aicc, bic, or blocked_cv")
    delta_keep = _finite_number("delta_keep", delta_keep, 0)
    if max_workers is not None:
        max_workers = _positive_integer("max_workers", max_workers)
    if metric == "blocked_cv":
        cv_folds = _positive_integer("cv_folds", cv_folds, 2)
        cv_gap_points = _positive_integer("cv_gap_points", cv_gap_points, 0)
        cv_min_train_line = _positive_integer("cv_min_train_line", cv_min_train_line, 5)
        cv_minuit_max_calls = _positive_integer("cv_minuit_max_calls", cv_minuit_max_calls)
        cv_minuit_n_starts = _positive_integer("cv_minuit_n_starts", cv_minuit_n_starts)
    scan_models = str(scan_models).lower()
    if scan_models not in {"none", "best", "supported", "all"}:
        raise ValueError("scan_models must be none, best, supported, or all")
    if candidate_transfers is None:
        candidate_transfers = SUPPORTED_TRANSFERS
    candidates = []
    for name in candidate_transfers:
        canonical = normalize_transfer_name(name)
        if canonical not in candidates:
            candidates.append(canonical)
    if not candidates:
        raise ValueError("candidate_transfers is empty")

    # Joint-MLE screening. All workers receive the same inputs/configuration.
    base_kwargs = dict(run_mle_kwargs)
    base_kwargs.pop("transfer", None)
    base_kwargs.pop("run_scan", None)
    base_kwargs.pop("save", None)
    base_kwargs.pop("make_figures", None)
    # Preserve the established no-prior model-selection behavior. Smoothing
    # stays with the requested model; information criteria use a separate fit.
    if base_kwargs.get("use_prior"):
        warnings.warn(
            "automatic model selection uses use_prior=False, as before; "
            "use run_mle for a fit with priors",
            UserWarning,
            stacklevel=2,
        )
    base_kwargs["use_prior"] = False
    data = load_data(continuum, line)
    continuum_array = np.column_stack(
        [data["t_c"], data["c_flux"], data["c_err"]]
    )
    line_array = np.column_stack([data["t_l"], data["l_flux"], data["l_err"]])
    payloads = [
        {
            "transfer": transfer,
            "continuum": continuum_array,
            "line": line_array,
            "run_scan": False,
            "selection_metric": metric,
            "kwargs": base_kwargs,
            "cv_config": {
                "enabled": bool(metric == "blocked_cv" and len(candidates) > 1),
                "n_folds": cv_folds,
                "gap_points": cv_gap_points,
                "min_train_line": cv_min_train_line,
                "minuit_max_calls": cv_minuit_max_calls,
                "minuit_n_starts": cv_minuit_n_starts,
            },
        }
        for transfer in candidates
    ]

    stage1 = {}
    if parallel and len(payloads) > 1:
        workers = max_workers or min(len(payloads), max(os.cpu_count() or 1, 1))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_fit_transfer_worker, p): p["transfer"] for p in payloads}
            for future in as_completed(future_map):
                transfer = future_map[future]
                try:
                    _, result, criteria = future.result()
                    stage1[transfer] = {"result": result, "criteria": criteria, "error": None}
                except Exception as exc:
                    stage1[transfer] = {"result": None, "criteria": None, "error": repr(exc)}
    else:
        for payload in payloads:
            transfer = payload["transfer"]
            try:
                _, result, criteria = _fit_transfer_worker(payload)
                stage1[transfer] = {"result": result, "criteria": criteria, "error": None}
            except Exception as exc:
                stage1[transfer] = {"result": None, "criteria": None, "error": repr(exc)}

    if metric == "blocked_cv" and len(candidates) == 1:
        # No model choice is required for a single specified transfer. Keep the
        # interface valid without spending time on unnecessary CV refits.
        only_entry = stage1.get(candidates[0])
        if only_entry is not None and only_entry.get("criteria") is not None:
            only_entry["criteria"].update(
                blocked_cv_logscore=0.0,
                blocked_cv_mean_logscore=0.0,
                blocked_cv_n_test=0,
                blocked_cv_n_folds=0,
                blocked_cv_failed_folds=0,
                blocked_cv_folds=[],
                blocked_cv_error="skipped_single_candidate",
            )

    metric_key = "blocked_cv_logscore" if metric == "blocked_cv" else metric
    valid = [
        v for v in stage1.values()
        if v["criteria"] is not None and np.isfinite(v["criteria"].get(metric_key, np.nan))
    ]
    if not valid:
        details = {
            k: (v.get("error") or ((v.get("criteria") or {}).get("blocked_cv_error")))
            for k, v in stage1.items()
        }
        raise RuntimeError(f"all transfer-model fits/validation runs failed: {details}")

    if metric == "blocked_cv":
        best_value = max(v["criteria"][metric_key] for v in valid)
        for transfer, entry in stage1.items():
            if entry["criteria"] is not None and np.isfinite(entry["criteria"].get(metric_key, np.nan)):
                # Twice the predictive-logscore loss gives a deviance-like delta,
                # retaining the familiar exp(-delta/2) relative-weight mapping.
                entry["criteria"]["delta_metric"] = float(
                    2.0 * (best_value - entry["criteria"][metric_key])
                )
        ranked_entries = sorted(
            [
                (name, entry) for name, entry in stage1.items()
                if entry["criteria"] is not None
                and np.isfinite(entry["criteria"].get(metric_key, np.nan))
            ],
            key=lambda item: item[1]["criteria"][metric_key],
            reverse=True,
        )
    else:
        best_value = min(v["criteria"][metric_key] for v in valid)
        for transfer, entry in stage1.items():
            if entry["criteria"] is not None and np.isfinite(entry["criteria"][metric_key]):
                entry["criteria"]["delta_metric"] = float(
                    entry["criteria"][metric_key] - best_value
                )
        ranked_entries = sorted(
            [(name, entry) for name, entry in stage1.items()
             if entry["criteria"] is not None and np.isfinite(entry["criteria"][metric_key])],
            key=lambda item: item[1]["criteria"][metric_key],
        )
    best_transfer = ranked_entries[0][0]
    supported = [
        name for name, entry in ranked_entries
        if entry["criteria"]["delta_metric"] <= float(delta_keep)
    ]
    if scan_models == "none":
        to_scan = []
    elif scan_models == "best":
        to_scan = [best_transfer]
    elif scan_models == "supported":
        to_scan = supported
    else:
        to_scan = [name for name, _ in ranked_entries]
    # Lag scans reuse the already-computed stage-1 joint-MLE results, avoiding
    # a repeated joint optimization for the selected model(s).
    if to_scan:
        regular_scan_mode = str(base_kwargs.get("scan_mode", "profile-local"))
        scan_payloads = []
        for transfer in to_scan:
            mode = regular_scan_mode
            scan_payloads.append(
                {
                    "transfer": transfer,
                    "result": stage1[transfer]["result"],
                    "scan_mode": mode,
                }
            )

        if parallel and len(scan_payloads) > 1:
            workers = max_workers or min(len(scan_payloads), max(os.cpu_count() or 1, 1))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(_scan_existing_result_worker, p): p["transfer"]
                    for p in scan_payloads
                }
                for future in as_completed(future_map):
                    transfer = future_map[future]
                    try:
                        _, result = future.result()
                        stage1[transfer]["result"] = result
                        stage1[transfer]["scan_error"] = None
                        stage1[transfer]["scan_mode_used"] = result["config"]["scan_mode"]
                    except Exception as exc:
                        stage1[transfer]["scan_error"] = repr(exc)
        else:
            for payload in scan_payloads:
                transfer = payload["transfer"]
                try:
                    _, result = _scan_existing_result_worker(payload)
                    stage1[transfer]["result"] = result
                    stage1[transfer]["scan_error"] = None
                    stage1[transfer]["scan_mode_used"] = result["config"]["scan_mode"]
                except Exception as exc:
                    stage1[transfer]["scan_error"] = repr(exc)

        scan_errors = {name: stage1[name]["scan_error"] for name in to_scan
                       if stage1[name].get("scan_error")}
        if scan_errors:
            raise RuntimeError(f"requested model lag scans failed: {scan_errors}")

    # Akaike-style weights based on the selected information criterion.
    deltas = np.array([entry["criteria"]["delta_metric"] for _, entry in ranked_entries])
    raw_w = np.exp(-0.5 * deltas)
    raw_w /= np.sum(raw_w)
    ranking = []
    for rank, ((name, _), weight) in enumerate(zip(ranked_entries, raw_w), start=1):
        entry = stage1[name]
        row = _serializable_model_summary(entry["result"], entry["criteria"])
        row.update(
            rank=rank,
            delta_metric=float(entry["criteria"]["delta_metric"]),
            model_weight=float(weight),
            supported=bool(name in supported),
        )
        ranking.append(row)


    selection = {
        "selection_metric": metric,
        "delta_keep": float(delta_keep),
        "blocked_cv_config": {
            "n_folds": int(cv_folds),
            "gap_points": int(cv_gap_points),
            "min_train_line": int(cv_min_train_line),
            "minuit_max_calls": int(cv_minuit_max_calls),
            "minuit_n_starts": int(cv_minuit_n_starts),
        } if metric == "blocked_cv" else None,
        "best_transfer": best_transfer,
        "supported_transfers": supported,
        "ranking": ranking,
        "models": stage1,
    }
    if save:
        _save_model_selection_outputs(output_prefix, selection)
    if save or make_figures:
        base = Path(output_prefix)
        for transfer in to_scan:
            entry = stage1.get(transfer, {})
            result = entry.get("result")
            if not result or not result.get("run_scan"):
                continue
            model_dir = base.parent / f"{base.name}_models" / transfer
            model_dir.mkdir(parents=True, exist_ok=True)
            model_prefix = model_dir / transfer
            if save:
                save_outputs(
                    str(model_prefix),
                    result["lag_grid"],
                    result["loglike"],
                    result["relative_likelihood_density"],
                    result["p_hat"],
                    result["params_at_lag"],
                    result["config"]["scan_mode"],
                    result["summary"],
                    result["transform_info"],
                )
            if make_figures:
                make_plots(
                    str(model_prefix),
                    result["data"],
                    result["z_c"],
                    result["z_l"],
                    result["sigma_z_c"],
                    result["sigma_z_l"],
                    result["lag_grid"],
                    result["loglike"],
                    result["relative_likelihood_density"],
                    result["p_hat"],
                    save=True,
                    show=False,
                    score_kind=result["fit_score_kind"],
                )
    return selection

# =============================================================================
# FLEXIBLE THREE-GAUSSIAN TRANSFER MODEL
# =============================================================================
# This section preserves the original joint Gaussian-copula likelihood and the
# existing package/output structure.  It adds one low-dimensional flexible
# transfer function, ``flexible_gaussian3``, parameterized by an exact centroid
# lag, one spacing/width scale, and three positive normalized mixture weights.

def _softmax_three_from_two_logits(logit_1, logit_2):
    values = np.array([float(logit_1), float(logit_2), 0.0], dtype=float)
    values -= float(np.max(values))
    weights = np.exp(values)
    weights /= float(np.sum(weights))
    return weights


def _weights_to_two_logits(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("flexible transfer weights must contain three positive values")
    weights = weights / float(np.sum(weights))
    return np.array(
        [math.log(weights[0] / weights[2]), math.log(weights[1] / weights[2])],
        dtype=float,
    )


def _parse_flexible_initial_weights(value):
    if value is None:
        return np.array([0.25, 0.50, 0.25], dtype=float)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                "flexible_initial_weights must contain three comma-separated values"
            )
        arr = np.asarray([float(part) for part in parts], dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("flexible initial weights must contain three positive values")
    return arr / float(np.sum(arr))


def _flexible_response_metadata(
    lag, width, mixture_weights, sigma_ratio, enforce_nonnegative,
    flexible_center=None,
):
    """Resolve a continuous response, without silently selecting a root branch."""
    lag = _finite_number("lag", lag)
    if enforce_nonnegative and lag <= 0:
        raise ValueError("a positive-width non-negative transfer requires lag>0")
    if flexible_center is None:
        roots = _flexible.center_roots_for_lag(
            lag, width, mixture_weights, sigma_ratio=sigma_ratio,
            enforce_nonnegative=enforce_nonnegative,
        )
        if len(roots) != 1:
            raise _flexible.AmbiguousCentroidError(
                "this flexible response has multiple center branches; specify "
                "flexible_center explicitly (the middle Gaussian center)"
            )
        flexible_center = float(roots[0])
    metadata = _flexible.metadata_from_center(
        flexible_center, width, mixture_weights, sigma_ratio=sigma_ratio,
        enforce_nonnegative=enforce_nonnegative,
    )
    tolerance = 1e-8 * max(1.0, abs(lag), abs(width))
    if abs(metadata["centroid"] - lag) > tolerance:
        raise ValueError("flexible_center is inconsistent with the requested centroid lag")
    return metadata


def _flexible_gaussian3_quadrature(
    lag,
    width,
    mixture_weights,
    sigma_ratio=0.45,
    grid_size=81,
    enforce_nonnegative=False,
    flexible_center=None,
):
    """Quadrature of a continuous 3G response, with explicit ambiguous branches.

    Unlike the historical midpoint rule, centers are calibrated independently
    of the quadrature. Nodes may be nonuniform; they must not be passed to the
    uniform-grid autocorrelation shortcut.
    """
    metadata = _flexible_response_metadata(
        lag, width, mixture_weights, sigma_ratio, enforce_nonnegative,
        flexible_center=flexible_center,
    )
    q, weights = _flexible.quadrature_from_metadata(metadata, grid_size=grid_size)
    return np.asarray(q, float) - lag, np.asarray(weights, float), metadata


def make_transfer_quadrature(
    transfer,
    lag,
    width,
    grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    enforce_nonnegative=False,
    flexible_weights=None,
    flexible_sigma_ratio=0.45,
    flexible_center=None,
):
    transfer = normalize_transfer_name(transfer)
    if transfer == "flexible_gaussian3":
        if flexible_weights is None:
            flexible_weights = np.array([0.25, 0.50, 0.25], dtype=float)
        offsets, weights, _ = _flexible_gaussian3_quadrature(
            lag=lag,
            width=width,
            mixture_weights=flexible_weights,
            sigma_ratio=flexible_sigma_ratio,
            grid_size=grid_size,
            enforce_nonnegative=enforce_nonnegative,
            flexible_center=flexible_center,
        )
        return offsets, weights
    return _make_discrete_transfer_quadrature(
        transfer=transfer,
        lag=lag,
        width=width,
        grid_size=grid_size,
        long_tail_width_ratio=long_tail_width_ratio,
        long_tail_scale_ratio=long_tail_scale_ratio,
        double_exp_decay_ratio=double_exp_decay_ratio,
        enforce_nonnegative=enforce_nonnegative,
    )


def build_reverberation_covariance(
    t_c,
    t_l,
    lag,
    tau_drw,
    rho,
    transfer="tophat",
    transfer_width=None,
    transfer_grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    enforce_nonnegative_transfer=False,
    flexible_weights=None,
    flexible_sigma_ratio=0.45,
    flexible_center=None,
):
    lag = _finite_number("lag", lag)
    tau_drw = _finite_number("tau_drw", tau_drw)
    rho = _finite_number("rho", rho)
    if tau_drw <= 0.0 or not (0.0 <= rho < 1.0):
        return None
    transfer = normalize_transfer_name(transfer)
    if transfer_width is None:
        if transfer_has_width(transfer):
            raise ValueError(
                "transfer_width is required when constructing a finite-width covariance"
            )
        transfer_width = 1.0
    transfer_width = _finite_number("transfer_width", transfer_width)
    t_c = np.asarray(t_c, float)
    t_l = np.asarray(t_l, float)
    for name, times in (("t_c", t_c), ("t_l", t_l)):
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError(f"{name} must be a finite one-dimensional time array")
    Kcc = ou_corr(t_c, t_c, tau_drw)

    if transfer == "delta":
        if enforce_nonnegative_transfer and lag < 0.0:
            return None
        Kcl = float(rho) * ou_corr(t_c, t_l - float(lag), tau_drw)
        Kll = ou_corr(t_l, t_l, tau_drw)
    elif transfer == "tophat":
        if transfer_width <= 0.0:
            return None
        if enforce_nonnegative_transfer and lag - 0.5 * transfer_width < 0.0:
            return None
        Kcl = float(rho) * ou_tophat_cross_corr(
            t_c, t_l, lag=lag, tau=tau_drw, width=transfer_width
        )
        Kll = ou_tophat_auto_corr(t_l, t_l, tau=tau_drw, width=transfer_width)
    elif transfer == "flexible_gaussian3":
        if transfer_width <= 0 or (enforce_nonnegative_transfer and lag <= 0):
            return None
        if flexible_weights is None:
            flexible_weights = np.array([0.25, 0.50, 0.25])
        metadata = _flexible_response_metadata(
            lag, transfer_width, flexible_weights, flexible_sigma_ratio,
            enforce_nonnegative_transfer, flexible_center=flexible_center,
        )
        Kcl, Kll = _flexible.ou_covariance_from_metadata(
            t_c, t_l, tau_drw, rho, metadata, grid_size=transfer_grid_size,
        )
    else:
        try:
            offsets, weights = make_transfer_quadrature(
                transfer,
                lag=lag,
                width=transfer_width,
                grid_size=transfer_grid_size,
                long_tail_width_ratio=long_tail_width_ratio,
                long_tail_scale_ratio=long_tail_scale_ratio,
                double_exp_decay_ratio=double_exp_decay_ratio,
                enforce_nonnegative=enforce_nonnegative_transfer,
                flexible_weights=flexible_weights,
                flexible_sigma_ratio=flexible_sigma_ratio,
            )
        except ValueError:
            return None
        Kcl, Kll = _weighted_ou_convolution_covariance(
            t_c, t_l, lag, tau_drw, rho, offsets, weights
        )
    K = np.block([[Kcc, Kcl], [Kcl.T, Kll]])
    return 0.5 * (K + K.T)


def build_kobs(
    t_c,
    t_l,
    sigma_z_c,
    sigma_z_l,
    lag,
    tau_drw,
    rho,
    transfer="tophat",
    transfer_width=None,
    transfer_grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    enforce_nonnegative_transfer=False,
    jitter=1e-8,
    flexible_weights=None,
    flexible_sigma_ratio=0.45,
    flexible_center=None,
):
    K = build_reverberation_covariance(
        t_c=t_c,
        t_l=t_l,
        lag=lag,
        tau_drw=tau_drw,
        rho=rho,
        transfer=transfer,
        transfer_width=transfer_width,
        transfer_grid_size=transfer_grid_size,
        long_tail_width_ratio=long_tail_width_ratio,
        long_tail_scale_ratio=long_tail_scale_ratio,
        double_exp_decay_ratio=double_exp_decay_ratio,
        enforce_nonnegative_transfer=enforce_nonnegative_transfer,
        flexible_weights=flexible_weights,
        flexible_sigma_ratio=flexible_sigma_ratio,
        flexible_center=flexible_center,
    )
    if K is None:
        return None
    sigma_z = np.r_[np.asarray(sigma_z_c, float), np.asarray(sigma_z_l, float)]
    K = K + np.diag(sigma_z * sigma_z)
    if jitter is not None and jitter > 0.0:
        K = K + float(jitter) * np.eye(K.shape[0])
    return 0.5 * (K + K.T)


def loglike_gcp_rm(
    z,
    sigma_z,
    t_c,
    t_l,
    lag,
    tau_drw,
    rho,
    transfer="tophat",
    transfer_width=None,
    transfer_grid_size=61,
    long_tail_width_ratio=10.0,
    long_tail_scale_ratio=1.0 / 3.0,
    double_exp_decay_ratio=1.2 / 0.62,
    enforce_nonnegative_transfer=False,
    jitter=1e-8,
    cholesky_jitter=1e-8,
    flexible_weights=None,
    flexible_sigma_ratio=0.45,
    flexible_center=None,
):
    try:
        z = np.asarray(z, dtype=float)
        sigma_z = np.asarray(sigma_z, dtype=float)
        t_c = np.asarray(t_c, dtype=float)
        t_l = np.asarray(t_l, dtype=float)
        n_c, n_l = len(t_c), len(t_l)
        if len(z) != n_c + n_l or len(sigma_z) != n_c + n_l:
            return -np.inf
        if np.any(~np.isfinite(sigma_z)) or np.any(sigma_z < 0.0):
            return -np.inf
        transfer = normalize_transfer_name(transfer)
        if transfer_has_width(transfer) and (
            transfer_width is None or transfer_width <= 0.0
        ):
            return -np.inf
        K = build_kobs(
            t_c,
            t_l,
            sigma_z[:n_c],
            sigma_z[n_c:],
            lag=lag,
            tau_drw=tau_drw,
            rho=rho,
            transfer=transfer,
            transfer_width=transfer_width,
            transfer_grid_size=transfer_grid_size,
            long_tail_width_ratio=long_tail_width_ratio,
            long_tail_scale_ratio=long_tail_scale_ratio,
            double_exp_decay_ratio=double_exp_decay_ratio,
            enforce_nonnegative_transfer=enforce_nonnegative_transfer,
            jitter=jitter,
            flexible_weights=flexible_weights,
            flexible_sigma_ratio=flexible_sigma_ratio,
            flexible_center=flexible_center,
        )
        if K is None:
            return -np.inf
        R = covariance_to_correlation(K)
        return gaussian_copula_loglike_from_R(
            z, R, cholesky_jitter=cholesky_jitter
        )
    except (_flexible.NumericalConvergenceError, _flexible.AmbiguousCentroidError):
        raise
    except Exception:
        return -np.inf


def _is_flexible_cfg(cfg):
    return normalize_transfer_name(cfg["transfer"]) == "flexible_gaussian3"


def _uses_flexible_center_coordinates(cfg):
    """A causal mixture is identified by its center, not a multivalued inverse."""
    return _is_flexible_cfg(cfg) and cfg["enforce_nonnegative_transfer"]


def transfer_parameter_count(name, fit_transfer_width=True):
    name = normalize_transfer_name(name)
    if name == "flexible_gaussian3":
        return (1 if fit_transfer_width else 0) + 2
    return int(bool(fit_transfer_width and name in WIDTH_TRANSFERS))


def _joint_bounds(cfg):
    position_bounds = (cfg["lag_min"], cfg["lag_max"])
    if _uses_flexible_center_coordinates(cfg):
        from ._flexible import center_bounds

        if cfg["lag_max"] <= 0.0:
            raise ValueError("a causal positive-width flexible transfer requires lag_max>0")
        width_max = (cfg["transfer_width_bounds"][1]
                     if cfg["fit_transfer_width"] else cfg["transfer_width"])
        lower, upper = center_bounds(
            cfg["lag_min"], cfg["lag_max"], width_max,
            sigma_ratio=cfg["flexible_sigma_ratio"],
        )
        position_bounds = (-np.inf if lower is None else lower, upper)
    bounds = [
        position_bounds,
        (math.log(cfg["tau_drw_bounds"][0]), math.log(cfg["tau_drw_bounds"][1])),
        cfg["rho_logit_bounds"],
    ]
    if cfg["fit_transfer_width"]:
        bounds.append(
            (
                math.log(cfg["transfer_width_bounds"][0]),
                math.log(cfg["transfer_width_bounds"][1]),
            )
        )
    if _is_flexible_cfg(cfg):
        bounds.extend([cfg["flexible_weight_logit_bounds"]] * 2)
    return bounds


def _nuisance_bounds(cfg):
    return _joint_bounds(cfg)[1:]


def _parameter_names(cfg, include_lag=True):
    position_name = "transfer_center" if _uses_flexible_center_coordinates(cfg) else "lag"
    names = [position_name] if include_lag else []
    names.extend(["log_tau_drw", "rho_logit"])
    if cfg["fit_transfer_width"]:
        names.append("log_transfer_width")
    if _is_flexible_cfg(cfg):
        names.extend(["weight_logit_1", "weight_logit_2"])
    return names


def _flexible_params_from_center(p, center, cfg):
    """Attach continuous response metadata without solving a centroid inverse."""
    from ._flexible import metadata_from_center

    metadata = metadata_from_center(
        center, p["transfer_width"], p["transfer_weights"],
        sigma_ratio=cfg["flexible_sigma_ratio"],
        enforce_nonnegative=cfg["enforce_nonnegative_transfer"],
    )
    params = dict(p)
    params.update(
        lag=float(metadata["centroid"]),
        transfer_component_centers=np.asarray(metadata["component_centers"], float),
        transfer_component_sigma=float(metadata["component_sigma"]),
        transfer_common_center_shift=float(metadata["common_center_shift"]),
    )
    return params


def _flexible_params_at_lag(p, lag, cfg):
    """Return every continuous causal response having this physical centroid.

    The inverse map need not be one-to-one.  Callers with observations must
    compare these candidates using the same objective instead of selecting a
    root according to a numerical solver's initial bracket.
    """
    from ._flexible import center_roots_for_lag

    lag = float(lag)
    if cfg["enforce_nonnegative_transfer"] and lag <= 0.0:
        return []
    roots = center_roots_for_lag(
        lag, p["transfer_width"], p["transfer_weights"],
        sigma_ratio=cfg["flexible_sigma_ratio"],
        enforce_nonnegative=cfg["enforce_nonnegative_transfer"],
    )
    if len(roots) == 0:
        raise _flexible.NumericalConvergenceError(
            f"no continuous flexible center found for the valid centroid lag={lag}"
        )
    candidates = [_flexible_params_from_center(p, center, cfg) for center in roots]
    # The root residual is numerical, not a change to the requested profile lag.
    for candidate in candidates:
        candidate["lag"] = lag
    return candidates


def _flexible_metadata_for_params(lag, width, weights, cfg):
    """Compatibility helper that refuses to silently choose a mixture branch."""
    candidates = _flexible_params_at_lag(
        {"transfer_width": width, "transfer_weights": weights}, lag, cfg
    )
    if len(candidates) != 1:
        raise _flexible.AmbiguousCentroidError(
            f"flexible centroid has {len(candidates)} center solutions; "
            "supply an explicit center or compare all response branches"
        )
    params = candidates[0]
    return {
        "component_centers": params["transfer_component_centers"],
        "component_sigma": params["transfer_component_sigma"],
        "common_center_shift": params["transfer_common_center_shift"],
    }


def _x_to_params(x, cfg, fixed_lag=None, *, all_flexible_branches=False):
    x = np.asarray(x, dtype=float)
    offset = 0
    if fixed_lag is None:
        lag = float(x[0])
        offset = 1
    else:
        lag = float(fixed_lag)
    tau_drw = math.exp(float(x[offset]))
    rho = cfg["rho_max"] * expit_scalar(float(x[offset + 1]))
    offset += 2
    if cfg["fit_transfer_width"]:
        transfer_width = math.exp(float(x[offset]))
        offset += 1
    else:
        transfer_width = cfg["transfer_width"]
    params = {
        "lag": lag,
        "tau_drw": float(tau_drw),
        "rho": float(rho),
        "transfer": cfg["transfer"],
        "transfer_width": (
            float(transfer_width) if transfer_has_width(cfg["transfer"]) else None
        ),
        "jitter": cfg["jitter"],
    }
    if _is_flexible_cfg(cfg):
        params["transfer_weights"] = _softmax_three_from_two_logits(
            x[offset], x[offset + 1]
        )
        if fixed_lag is None and _uses_flexible_center_coordinates(cfg):
            return _flexible_params_from_center(params, float(x[0]), cfg)
        candidates = _flexible_params_at_lag(params, lag, cfg)
        if all_flexible_branches:
            return candidates
        if len(candidates) != 1:
            raise _flexible.AmbiguousCentroidError(
                f"flexible centroid has {len(candidates)} center solutions; "
                "request all_flexible_branches and compare their objectives"
            )
        return candidates[0]
    return params


def _params_to_joint_starts(p, cfg):
    """Recover saved centers, including old results, without changing branches."""
    nuisance = _params_to_nuisance_x(p, cfg)
    if not _uses_flexible_center_coordinates(cfg):
        return [np.r_[p["lag"], nuisance]]
    centers = p.get("transfer_component_centers")
    if centers is not None:
        centers = np.asarray(centers, float)
        if centers.shape == (3,) and np.all(np.isfinite(centers)):
            return [np.r_[centers[1], nuisance]]
    base = dict(p)
    if base.get("transfer_weights") is None:
        base["transfer_weights"] = cfg["flexible_initial_weights"]
    return [np.r_[candidate["transfer_component_centers"][1], nuisance]
            for candidate in _flexible_params_at_lag(base, p["lag"], cfg)]


def _params_to_nuisance_x(p, cfg):
    vals = [
        math.log(float(p["tau_drw"])),
        logit_scalar(float(p["rho"]) / cfg["rho_max"]),
    ]
    if cfg["fit_transfer_width"]:
        vals.append(math.log(float(p.get("transfer_width", cfg["transfer_width"]))))
    if _is_flexible_cfg(cfg):
        weights = p.get("transfer_weights")
        if weights is None:
            weights = cfg["flexible_initial_weights"]
        vals.extend(_weights_to_two_logits(weights).tolist())
    return np.asarray(vals, dtype=float)


def _data_loglike_for_params(p, z, sigma_z, t_c, t_l, cfg):
    return loglike_gcp_rm(
        z,
        sigma_z,
        t_c,
        t_l,
        p["lag"],
        p["tau_drw"],
        p["rho"],
        transfer=p["transfer"],
        transfer_width=p["transfer_width"],
        transfer_grid_size=cfg["transfer_grid_size"],
        long_tail_width_ratio=cfg["long_tail_width_ratio"],
        long_tail_scale_ratio=cfg["long_tail_scale_ratio"],
        double_exp_decay_ratio=cfg["double_exp_decay_ratio"],
        enforce_nonnegative_transfer=cfg["enforce_nonnegative_transfer"],
        jitter=p["jitter"],
        cholesky_jitter=cfg["cholesky_jitter"],
        flexible_weights=p.get("transfer_weights"),
        flexible_sigma_ratio=cfg["flexible_sigma_ratio"],
        flexible_center=(
            float(p["transfer_component_centers"][1])
            if _is_flexible_cfg(cfg) and p.get("transfer_component_centers") is not None
            else None
        ),
    )


def _objective_value(p, z, sigma_z, t_c, t_l, cfg):
    if _uses_flexible_center_coordinates(cfg) and not (
        cfg["lag_min"] <= p["lag"] <= cfg["lag_max"]
    ):
        return -np.inf
    val = _data_loglike_for_params(p, z, sigma_z, t_c, t_l, cfg)
    val += log_prior_params(
        p["tau_drw"], p["rho"], p["transfer_width"], cfg
    )
    if _is_flexible_cfg(cfg) and p.get("transfer_weights") is not None:
        a = np.asarray(p["transfer_weights"], dtype=float)
        val -= float(
            cfg["flexible_smoothness"]
            * ((a[1] - a[0]) ** 2 + (a[2] - a[1]) ** 2)
        )
    return float(val)


def _default_joint_x(cfg):
    duration = cfg["total_duration"]
    tau0 = np.clip(
        cfg["default_tau_drw_frac"] * duration, *cfg["tau_drw_bounds"]
    )
    rho0 = min(
        max(cfg["default_rho"], 1e-8),
        cfg["rho_max"] * (1.0 - 1e-8),
    )
    lag0 = 0.5 * (cfg["lag_min"] + cfg["lag_max"])
    if _uses_flexible_center_coordinates(cfg) and lag0 <= 0.0:
        lag0 = 0.5 * cfg["lag_max"]
    values = [
        lag0,
        math.log(float(tau0)),
        logit_scalar(rho0 / cfg["rho_max"]),
    ]
    if cfg["fit_transfer_width"]:
        width0 = np.clip(cfg["transfer_width"], *cfg["transfer_width_bounds"])
        values.append(math.log(float(width0)))
    if _is_flexible_cfg(cfg):
        values.extend(_weights_to_two_logits(cfg["flexible_initial_weights"]).tolist())
    if _uses_flexible_center_coordinates(cfg):
        candidates = _flexible_params_at_lag(
            {
                "transfer_width": float(width0) if cfg["fit_transfer_width"]
                                  else cfg["transfer_width"],
                "transfer_weights": cfg["flexible_initial_weights"],
            },
            lag0, cfg,
        )
        if not candidates:
            raise ValueError("lag bounds contain no positive causal flexible centroid")
        # This is just one default vector; get_optimal_x expands every root at
        # every seed lag below, so this initialization does not select a branch.
        values[0] = float(candidates[0]["transfer_component_centers"][1])
    return np.asarray(values, dtype=float)


def get_optimal_x(z, sigma_z, t_c, t_l, cfg):
    if cfg.get("total_duration") is None:
        cfg["total_duration"] = data_duration(t_c, t_l)
    bounds = _joint_bounds(cfg)
    names = _parameter_names(cfg, include_lag=True)

    def objective(x):
        p = _x_to_params(x, cfg)
        val = _objective_value(p, z, sigma_z, t_c, t_l, cfg)
        return -val if np.isfinite(val) else 1e100

    default_x = _default_joint_x(cfg)
    n_starts = int(cfg["minuit_n_starts"])
    n_seed_lags = max(9, min(41, 5 * n_starts))
    seed_bounds = (cfg["lag_min"], cfg["lag_max"])
    if _uses_feasible_tophat_coordinates(cfg):
        seed_bounds = _feasible_tophat_lag_bounds(cfg)
    seed_lags = np.linspace(*seed_bounds, n_seed_lags)
    if _uses_flexible_center_coordinates(cfg):
        lower, upper = max(cfg["lag_min"], 0.0), cfg["lag_max"]
        seed_lags = np.linspace(lower, upper, n_seed_lags + 1)[1:]
    weight_seeds = [
        np.asarray(cfg.get("flexible_initial_weights", [0.25, 0.50, 0.25]), float)
    ]
    if _is_flexible_cfg(cfg):
        weight_seeds.extend(
            [
                np.array([1.0 / 3.0] * 3),
                np.array([0.60, 0.30, 0.10]),
                np.array([0.10, 0.30, 0.60]),
                np.array([0.45, 0.10, 0.45]),
            ]
        )
    seed_points = []
    seed_failures = []
    for lag in seed_lags:
        for weights in weight_seeds:
            point = default_x.copy()
            point[0] = lag
            if _uses_feasible_tophat_coordinates(cfg):
                width_upper = min(cfg["transfer_width_bounds"][1], 2.0 * lag)
                width = np.clip(cfg["transfer_width"],
                                cfg["transfer_width_bounds"][0], width_upper)
                point[names.index("log_transfer_width")] = _log_width_inside_support(
                    width, width_upper
                )
            if _is_flexible_cfg(cfg):
                point[-2:] = _weights_to_two_logits(weights)
            if _uses_flexible_center_coordinates(cfg):
                width = (math.exp(point[names.index("log_transfer_width")])
                         if cfg["fit_transfer_width"] else cfg["transfer_width"])
                try:
                    candidates = _flexible_params_at_lag(
                        {"transfer_width": width, "transfer_weights": weights}, lag, cfg
                    )
                except _flexible.NumericalConvergenceError as exc:
                    # A root unresolved at machine precision is a numerical
                    # seed failure, not an exclusion from physical support.
                    seed_failures.append(exc)
                    continue
                for candidate in candidates:
                    center_point = point.copy()
                    center_point[0] = candidate["transfer_component_centers"][1]
                    try:
                        seed_points.append((objective(center_point), center_point))
                    except _flexible.NumericalConvergenceError as exc:
                        seed_failures.append(exc)
            else:
                seed_points.append((objective(point), point))
    seed_points.sort(key=lambda item: item[0])

    best_x = None
    best_loglike = -np.inf
    best_minuit = None
    failures = list(seed_failures)
    for _, start in seed_points[:n_starts]:
        try:
            fit = _run_minuit(
                objective, start, bounds, names, cfg["minuit_max_calls"], cfg
            )
            candidate_x = _minuit_values(fit, names)
        except Exception as exc:
            failures.append(exc)
            continue
        if not fit.valid:
            failures.append(RuntimeError(f"optimizer did not converge: {_optimizer_diagnostics(fit)}"))
            continue
        p = _x_to_params(candidate_x, cfg)
        value = _objective_value(p, z, sigma_z, t_c, t_l, cfg)
        if value > best_loglike:
            best_x, best_loglike, best_minuit = candidate_x, value, fit
    if best_x is None:
        message = (
            f"joint GCP-RM optimization failed to converge for {cfg['transfer']}; "
            "increase minuit_max_calls/minuit_n_starts or check data and bounds"
        )
        if failures:
            raise RuntimeError(f"{message}; {failures[0]}") from failures[0]
        raise RuntimeError(message)
    p_hat = _x_to_params(best_x, cfg)
    best_loglike = _objective_value(p_hat, z, sigma_z, t_c, t_l, cfg)
    return p_hat, float(best_loglike), best_minuit


class ProfileLagInfeasible(ValueError):
    """A fixed lag has no nuisance parameters satisfying transfer support."""


def _nonnegative_transfer_support_feasible(lag, width, cfg):
    """Return whether one transfer has non-negative physical-delay support."""
    if not cfg["enforce_nonnegative_transfer"]:
        return True
    lag = float(lag)
    transfer = normalize_transfer_name(cfg["transfer"])
    if transfer == "delta":
        return lag >= 0.0
    if transfer == "edge_on_ring":
        return lag > 0.0
    width = float(width)
    if transfer == "flexible_gaussian3":
        # Every positive width and positive centroid is attainable by the
        # continuous truncated family. Numerical root failure is not a support
        # restriction and must never tighten the scientific width bounds.
        return lag > 0.0 and width > 0.0
    if transfer == "tophat":
        return width > 0.0 and lag - 0.5 * width >= 0.0
    try:
        make_transfer_quadrature(
            transfer,
            lag=lag,
            width=width,
            grid_size=cfg["transfer_grid_size"],
            long_tail_width_ratio=cfg["long_tail_width_ratio"],
            long_tail_scale_ratio=cfg["long_tail_scale_ratio"],
            double_exp_decay_ratio=cfg["double_exp_decay_ratio"],
            enforce_nonnegative=True,
            flexible_weights=cfg.get("flexible_initial_weights"),
            flexible_sigma_ratio=cfg.get("flexible_sigma_ratio", 0.45),
        )
        return True
    except (TypeError, ValueError, FloatingPointError):
        return False


def _profile_nuisance_bounds_at_lag(lag, cfg):
    """Build nuisance bounds conditional on a fixed lag's physical support.

    Transfer width and lag are coupled when negative physical delays are
    forbidden.  The joint optimizer uses rectangular global bounds, but a
    profile fit at fixed lag can and should tighten the width bound before
    calling Minuit.  This avoids a flat 1e100 penalty region.
    """
    bounds = [tuple(pair) for pair in _nuisance_bounds(cfg)]
    if cfg["transfer"] == "edge_on_ring" and lag <= 0:
        raise ProfileLagInfeasible("edge_on_ring requires lag > 0")
    if not cfg["enforce_nonnegative_transfer"]:
        return bounds
    if _is_flexible_cfg(cfg):
        if lag <= 0.0:
            raise ProfileLagInfeasible("a causal positive-width flexible transfer requires lag>0")
        return bounds

    if not cfg["fit_transfer_width"]:
        if not _nonnegative_transfer_support_feasible(
            lag, cfg["transfer_width"], cfg
        ):
            raise ProfileLagInfeasible(
                f"lag={float(lag):.12g} is incompatible with fixed "
                f"transfer={cfg['transfer']}, width={cfg['transfer_width']}"
            )
        return bounds

    width_min, width_max = map(float, cfg["transfer_width_bounds"])
    if not _nonnegative_transfer_support_feasible(lag, width_min, cfg):
        raise ProfileLagInfeasible(
            f"lag={float(lag):.12g} has no feasible transfer width "
            f"at or above width_min={width_min:.12g}"
        )

    feasible_width_max = width_max
    if not _nonnegative_transfer_support_feasible(lag, width_max, cfg):
        # Supported transfer families have a monotone support constraint in
        # width. Locate the largest feasible value using the actual quadrature
        # implementation so every transfer parameterization stays consistent.
        low, high = width_min, width_max
        for _ in range(60):
            middle = 0.5 * (low + high)
            if _nonnegative_transfer_support_feasible(lag, middle, cfg):
                low = middle
            else:
                high = middle
        feasible_width_max = low

    # Stay just inside the support boundary; evaluating exactly on it can turn
    # invalid through floating-point roundoff (for a top-hat, lag-width/2).
    safe_width_max = min(width_max, feasible_width_max)
    if safe_width_max < width_max:
        safe_width_max = max(
            width_min,
            np.nextafter(safe_width_max * (1.0 - 1e-10), width_min),
        )
    if safe_width_max <= width_min * (1.0 + 1e-12):
        raise ProfileLagInfeasible(
            f"lag={float(lag):.12g} leaves no non-degenerate width interval"
        )

    names = _parameter_names(cfg, include_lag=False)
    width_index = names.index("log_transfer_width")
    bounds[width_index] = (math.log(width_min), math.log(safe_width_max))
    return bounds


def optimize_nuisance_local(lag, starts, z, sigma_z, t_c, t_l, cfg):
    if cfg.get("total_duration") is None:
        cfg["total_duration"] = data_duration(t_c, t_l)
    bounds = _profile_nuisance_bounds_at_lag(lag, cfg)
    names = _parameter_names(cfg, include_lag=False)

    def params_and_value(x):
        candidates = _x_to_params(
            x, cfg, fixed_lag=lag, all_flexible_branches=True
        )
        if isinstance(candidates, dict):
            candidates = [candidates]
        scored = [(_objective_value(p, z, sigma_z, t_c, t_l, cfg), p)
                  for p in candidates]
        if not scored:
            raise RuntimeError(
                f"continuous flexible centroid solver returned no response at lag={lag}"
            )
        finite = [item for item in scored if np.isfinite(item[0])]
        return max(finite, key=lambda item: item[0]) if finite else (-np.inf, scored[0][1])

    def objective(x):
        val, _ = params_and_value(x)
        return -val if np.isfinite(val) else 1e100

    best_x = None
    best_params = None
    best_f = -np.inf
    failures = []
    for st in starts:
        st = np.asarray(st, float)
        # Keep the initial finite-difference stencil away from active bounds.
        st = np.array(
            [
                np.clip(
                    st[i],
                    bounds[i][0] + 1e-8 * (bounds[i][1] - bounds[i][0]),
                    bounds[i][1] - 1e-8 * (bounds[i][1] - bounds[i][0]),
                )
                for i in range(len(bounds))
            ],
            dtype=float,
        )
        candidate_vectors = []
        try:
            fit = _run_minuit(
                objective, st, bounds, names, cfg["profile_max_calls"], cfg
            )
            if fit.valid:
                candidate_vectors.append(_minuit_values(fit, names))
        except Exception as exc:
            failures.append(exc)
        for cand_x in candidate_vectors:
            cand_f, p = params_and_value(cand_x)
            if np.isfinite(cand_f) and cand_f > best_f:
                best_x, best_f = cand_x, cand_f
                best_params = p
    if best_x is None:
        message = (
            f"fixed-lag nuisance optimization did not converge at "
            f"lag={float(lag):.12g}"
        )
        if failures:
            raise RuntimeError(f"{message}; {failures[0]}") from failures[0]
        raise RuntimeError(message)
    return best_params, float(best_f)


def _data_loglike_at_hat(result):
    p = result["p_hat"]
    cfg = result["config"]
    return _data_loglike_for_params(
        p, result["z"], result["sigma_z"],
        result["data"]["t_c"], result["data"]["t_l"], cfg,
    )


def _refit_information_criteria(result, scoring_cfg, known_loglike):
    """Score from default starts and the known fit without changing that fit."""
    z, sigma_z = result["z"], result["sigma_z"]
    t_c, t_l = result["data"]["t_c"], result["data"]["t_l"]
    candidates = []
    failures = []
    try:
        params, value, fit = get_optimal_x(z, sigma_z, t_c, t_l, scoring_cfg)
        candidates.append((float(value), params, fit))
    except RuntimeError as exc:
        failures.append(str(exc))

    def objective(x):
        params = _x_to_params(x, scoring_cfg)
        if _uses_flexible_center_coordinates(scoring_cfg) and not (
            scoring_cfg["lag_min"] <= params["lag"] <= scoring_cfg["lag_max"]
        ):
            return 1e100
        value = _data_loglike_for_params(params, z, sigma_z, t_c, t_l, scoring_cfg)
        return -value if np.isfinite(value) else 1e100

    for start in _params_to_joint_starts(result["p_hat"], scoring_cfg):
        try:
            names = _parameter_names(scoring_cfg, include_lag=True)
            fit = _run_minuit(objective, start, _joint_bounds(scoring_cfg), names,
                              scoring_cfg["minuit_max_calls"], scoring_cfg)
            if fit.valid:
                params = _x_to_params(_minuit_values(fit, names), scoring_cfg)
                value = _data_loglike_for_params(params, z, sigma_z, t_c, t_l, scoring_cfg)
                if np.isfinite(value) and objective(_minuit_values(fit, names)) < 1e100:
                    candidates.append((float(value), params, fit))
            else:
                failures.append(f"original-fit start did not converge: {_optimizer_diagnostics(fit)}")
        except Exception as exc:
            failures.append(str(exc))

    if not candidates:
        raise RuntimeError(f"information-criteria scoring did not converge: {failures}")
    value, params, fit = max(candidates, key=lambda item: item[0])
    tolerance = 1e-8 * max(1.0, abs(known_loglike))
    if value < known_loglike - tolerance:
        raise RuntimeError(
            "information-criteria scoring found a worse likelihood than the known "
            "original parameters; increase minuit_max_calls/minuit_n_starts"
        )
    return params, value, _optimizer_diagnostics(fit)


def model_information_criteria(result):
    """Evaluate classical criteria without changing the fitted model.

    A penalized or prior-based input needs a separate likelihood-only scoring
    fit. Its configuration and parameters are returned as metadata, while the
    supplied result (including its smoothing, priors and parameters) is retained.
    """
    if not (result.get("optimizer_diagnostics") or {}).get("converged", False):
        raise ValueError("information criteria require a converged fit")
    cfg = result["config"]
    refit = _fit_score_kind(cfg) != "data_loglike"
    scoring_params = result["p_hat"]
    scoring_diagnostics = result["optimizer_diagnostics"]
    data_loglike_at_fit = _finite_number("data log likelihood at fit", _data_loglike_at_hat(result))
    if refit:
        scoring_cfg = dict(cfg, use_prior=False, flexible_smoothness=0.0)
        scoring_params, ll, scoring_diagnostics = _refit_information_criteria(
            result, scoring_cfg, data_loglike_at_fit,
        )
    else:
        ll = data_loglike_at_fit
    ll = float(ll)
    transfer = normalize_transfer_name(cfg["transfer"])
    k = 3 + transfer_parameter_count(transfer, cfg["fit_transfer_width"])
    n = int(len(result["z"]))
    aic = 2.0 * k - 2.0 * ll
    aicc = (
        aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        if n > k + 1
        else float("inf")
    )
    bic = k * math.log(max(n, 1)) - 2.0 * ll
    return {
        "transfer": transfer,
        "likelihood_mode": "joint_gaussian_copula",
        "n_parameters": int(k),
        "n_observations": int(n),
        "max_data_loglike": ll,
        "data_loglike_at_fit": data_loglike_at_fit,
        "information_criteria_status": "unpenalized_refit" if refit else "fitted_likelihood",
        "information_criteria_refit": refit,
        "information_criteria_fit": {
            "initial_data_loglike": data_loglike_at_fit,
            "initialization": "original_fit_and_default_starts" if refit else "original_fit",
            "use_prior": False,
            "flexible_smoothness": 0.0 if transfer == "flexible_gaussian3" else cfg["flexible_smoothness"],
            "optimizer_diagnostics": dict(scoring_diagnostics),
            "parameters": {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in scoring_params.items()
            },
        },
        "aic": float(aic),
        "aicc": float(aicc),
        "bic": float(bic),
    }


def blocked_time_predictive_validation(
    continuum,
    line,
    transfer,
    run_mle_kwargs,
    n_folds=5,
    gap_points=0,
    min_train_line=10,
    cv_minuit_max_calls=None,
    cv_minuit_n_starts=None,
):
    full_data = load_data(continuum, line)
    continuum_array = np.column_stack(
        [full_data["t_c"], full_data["c_flux"], full_data["c_err"]]
    )
    line_array = np.column_stack(
        [full_data["t_l"], full_data["l_flux"], full_data["l_err"]]
    )
    folds = make_contiguous_line_folds(
        len(line_array),
        n_folds=n_folds,
        gap_points=gap_points,
        min_train_line=min_train_line,
    )
    base_kwargs = dict(run_mle_kwargs)
    for key in (
        "transfer",
        "run_scan",
        "save",
        "make_figures",
        "show_figures",
        "output_prefix",
        "verbose",
    ):
        base_kwargs.pop(key, None)
    base_kwargs["use_prior"] = False
    base_kwargs["compute_hesse_errors"] = False
    if cv_minuit_max_calls is not None:
        base_kwargs["minuit_max_calls"] = int(cv_minuit_max_calls)
    if cv_minuit_n_starts is not None:
        base_kwargs["minuit_n_starts"] = int(cv_minuit_n_starts)

    fold_rows = []
    total_score = 0.0
    total_test = 0
    for fold in folds:
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]
        train_line_array = line_array[train_idx]
        fit = run_mle(
            continuum_array,
            train_line_array,
            transfer=transfer,
            run_scan=False,
            save=False,
            make_figures=False,
            show_figures=False,
            verbose=False,
            **base_kwargs,
        )
        line_transform = kde_gaussian_transform_from_reference(
            train_line_array[:, 1],
            line_array[:, 1],
            line_array[:, 2],
            bandwidth=fit["transform_info"]["l"]["bandwidth"],
            sigma_z_floor=fit["config"]["sigma_z_floor"],
            sigma_z_max=fit["config"]["sigma_z_max"],
        )
        z_c = np.asarray(fit["z_c"], dtype=float)
        sigma_z_c = np.asarray(fit["sigma_z_c"], dtype=float)
        z_l_all = np.asarray(line_transform["z"], dtype=float)
        sigma_z_l_all = np.asarray(line_transform["sigma_z"], dtype=float)
        z_all = np.r_[z_c, z_l_all]
        p_hat = fit["p_hat"]
        cfg = fit["config"]
        K = build_kobs(
            full_data["t_c"],
            full_data["t_l"],
            sigma_z_c,
            sigma_z_l_all,
            lag=p_hat["lag"],
            tau_drw=p_hat["tau_drw"],
            rho=p_hat["rho"],
            transfer=p_hat["transfer"],
            transfer_width=p_hat["transfer_width"],
            transfer_grid_size=cfg["transfer_grid_size"],
            long_tail_width_ratio=cfg["long_tail_width_ratio"],
            long_tail_scale_ratio=cfg["long_tail_scale_ratio"],
            double_exp_decay_ratio=cfg["double_exp_decay_ratio"],
            enforce_nonnegative_transfer=cfg["enforce_nonnegative_transfer"],
            jitter=p_hat["jitter"],
            flexible_weights=p_hat.get("transfer_weights"),
            flexible_sigma_ratio=cfg["flexible_sigma_ratio"],
            flexible_center=(float(p_hat["transfer_component_centers"][1])
                             if p_hat.get("transfer_component_centers") is not None
                             else None),
        )
        if K is None:
            raise RuntimeError(f"invalid covariance in blocked CV fold {fold['fold']}")
        R = covariance_to_correlation(K)
        n_c = len(full_data["t_c"])
        train_global = np.r_[np.arange(n_c, dtype=int), n_c + train_idx]
        test_global = n_c + test_idx
        fold_score = conditional_gaussian_copula_logscore_from_R(
            z_all,
            R,
            train_global,
            test_global,
            cholesky_jitter=cfg["cholesky_jitter"],
        )
        if not np.isfinite(fold_score):
            raise RuntimeError(f"non-finite blocked CV score in fold {fold['fold']}")
        n_test = int(len(test_idx))
        total_score += float(fold_score)
        total_test += n_test
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "n_train_line": int(len(train_idx)),
                "n_test_line": n_test,
                "test_time_min": float(full_data["t_l"][test_idx[0]]),
                "test_time_max": float(full_data["t_l"][test_idx[-1]]),
                "conditional_copula_logscore": float(fold_score),
                "mean_logscore_per_test_point": float(fold_score / n_test),
                "fit_lag": float(p_hat["lag"]),
                "fit_tau_drw": float(p_hat["tau_drw"]),
                "fit_rho": float(p_hat["rho"]),
                "fitted_transfer_width": _reported_transfer_width(p_hat),
                "fitted_transfer_weights": (
                    None
                    if p_hat.get("transfer_weights") is None
                    else np.asarray(p_hat["transfer_weights"], float).tolist()
                ),
            }
        )
    return {
        "blocked_cv_logscore": float(total_score),
        "blocked_cv_mean_logscore": float(total_score / max(total_test, 1)),
        "blocked_cv_n_test": int(total_test),
        "blocked_cv_n_folds": int(len(fold_rows)),
        "blocked_cv_failed_folds": 0,
        "blocked_cv_folds": fold_rows,
    }


def _serializable_model_summary(result, criteria):
    p = result["p_hat"]
    weights = p.get("transfer_weights")
    centers = p.get("transfer_component_centers")
    uncertainties = result.get("parameter_uncertainties") or {}
    diagnostics = uncertainties.get("diagnostics") or {}
    out = dict(criteria)
    out.update(
        likelihood_mode="joint_gaussian_copula",
        joint_loglike=float(result["joint_loglike"]),
        joint_objective=float(result["joint_objective"]),
        fit_score_kind=result["fit_score_kind"],
        optimizer_converged=bool(result["optimizer_diagnostics"]["converged"]),
        optimizer_diagnostics=result["optimizer_diagnostics"],
        joint_mle_lag=float(p["lag"]),
        joint_mle_lag_error=physical_parameter_error(uncertainties, "lag"),
        joint_mle_tau_drw=float(p["tau_drw"]),
        joint_mle_tau_drw_error=physical_parameter_error(
            uncertainties, "tau_drw"
        ),
        joint_mle_rho=float(p["rho"]),
        joint_mle_rho_error=physical_parameter_error(uncertainties, "rho"),
        joint_mle_transfer_width=_reported_transfer_width(p),
        joint_mle_transfer_width_error=physical_parameter_error(
            uncertainties, "transfer_width"
        ),
        hesse_available=bool(uncertainties.get("available", False)),
        hesse_status=uncertainties.get("status"),
        hesse_valid=diagnostics.get("is_valid"),
        hesse_accurate_covariance=diagnostics.get("has_accurate_covar"),
        hesse_positive_definite=diagnostics.get("has_posdef_covar"),
        hesse_parameters_at_limit=diagnostics.get("has_parameters_at_limit"),
        hesse_edm=diagnostics.get("edm"),
        hesse_message=uncertainties.get("message"),
        parameter_uncertainties=uncertainties,
        run_scan=bool(result.get("run_scan", False)),
        scan_mode_used=(
            (result.get("config") or {}).get("scan_mode")
            if result.get("run_scan")
            else None
        ),
        scan_reused_joint_mle=bool(result.get("scan_reused_joint_mle", False)),
        lag_profile_mle=result["summary"].get("lag_mle"),
        lr68_regions=result["summary"].get("lr68_regions", []),
        lr95_regions=result["summary"].get("lr95_regions", []),
    )
    if weights is not None and centers is not None and len(centers) == 3:
        for index in range(3):
            suffix = index + 1
            out[f"flexible_weight_{suffix}"] = float(weights[index])
            out[f"flexible_weight_{suffix}_error"] = physical_parameter_error(
                uncertainties, f"flexible_weight_{suffix}"
            )
            out[f"flexible_center_{suffix}"] = float(centers[index])
            out[f"flexible_center_{suffix}_error"] = physical_parameter_error(
                uncertainties, f"flexible_center_{suffix}"
            )
        out["flexible_component_sigma"] = float(
            p["transfer_component_sigma"]
        )
        out["flexible_component_sigma_error"] = physical_parameter_error(
            uncertainties, "flexible_component_sigma"
        )
        out["flexible_common_center_shift"] = float(
            p["transfer_common_center_shift"]
        )
        out["flexible_common_center_shift_error"] = physical_parameter_error(
            uncertainties, "flexible_common_center_shift"
        )
    return out


def _save_model_selection_outputs(output_prefix, selection):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = selection["ranking"]
    fields = [
        "rank",
        "transfer",
        "likelihood_mode",
        "n_parameters",
        "n_observations",
        "max_data_loglike",
        "data_loglike_at_fit",
        "information_criteria_status",
        "information_criteria_refit",
        "joint_loglike",
        "joint_objective",
        "fit_score_kind",
        "optimizer_converged",
        "aic",
        "aicc",
        "bic",
        "blocked_cv_logscore",
        "blocked_cv_mean_logscore",
        "blocked_cv_n_test",
        "blocked_cv_n_folds",
        "blocked_cv_failed_folds",
        "delta_metric",
        "model_weight",
        "supported",
        "joint_mle_lag",
        "joint_mle_lag_error",
        "joint_mle_tau_drw",
        "joint_mle_tau_drw_error",
        "joint_mle_rho",
        "joint_mle_rho_error",
        "joint_mle_transfer_width",
        "joint_mle_transfer_width_error",
        "flexible_weight_1",
        "flexible_weight_1_error",
        "flexible_weight_2",
        "flexible_weight_2_error",
        "flexible_weight_3",
        "flexible_weight_3_error",
        "flexible_center_1",
        "flexible_center_1_error",
        "flexible_center_2",
        "flexible_center_2_error",
        "flexible_center_3",
        "flexible_center_3_error",
        "flexible_component_sigma",
        "flexible_component_sigma_error",
        "flexible_common_center_shift",
        "flexible_common_center_shift_error",
        "hesse_available",
        "hesse_status",
        "hesse_valid",
        "hesse_accurate_covariance",
        "hesse_positive_definite",
        "hesse_parameters_at_limit",
        "hesse_edm",
        "hesse_message",
        "run_scan",
        "scan_mode_used",
        "scan_reused_joint_mle",
        "lag_profile_mle",
    ]
    with (prefix.parent / f"{prefix.name}_model_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_payload = {
        "method": "joint_gaussian_copula_reverberation_mapping",
        "selection_metric": selection["selection_metric"],
        "delta_keep": selection["delta_keep"],
        "blocked_cv_config": selection.get("blocked_cv_config"),
        "best_transfer": selection["best_transfer"],
        "supported_transfers": selection["supported_transfers"],
        "ranking": rows,
    }
    with (prefix.parent / f"{prefix.name}_model_selection.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(json_payload, f, indent=2)


def save_outputs(
    prefix,
    lag_grid,
    loglike,
    rel_density,
    p_hat,
    params_at_lag,
    scan_mode,
    summary,
    transform_info,
):
    _save_base_outputs(
        prefix,
        lag_grid,
        loglike,
        rel_density,
        p_hat,
        params_at_lag,
        scan_mode,
        summary,
        transform_info,
    )
    if p_hat.get("transfer_weights") is not None:
        with open(f"{prefix}_mle_params.txt", "a", encoding="utf-8") as f:
            weights = np.asarray(p_hat["transfer_weights"], float)
            centers = np.asarray(p_hat["transfer_component_centers"], float)
            f.write(
                "flexible_transfer_weights "
                + " ".join(map(str, weights.tolist()))
                + "\n"
            )
            f.write(
                "flexible_component_centers "
                + " ".join(map(str, centers.tolist()))
                + "\n"
            )
            f.write(
                f"flexible_component_sigma {p_hat['transfer_component_sigma']}\n"
            )
            f.write(
                f"flexible_sigma_ratio "
                f"{p_hat['transfer_component_sigma'] / p_hat['transfer_width']}\n"
            )


def format_report(result):
    p = result["p_hat"]
    summary = result["summary"]
    uncertainties = result.get("parameter_uncertainties") or {}
    physical = uncertainties.get("physical_parameters") or {}
    lines = [
        "GCP-RM joint-likelihood report",
        f"scan_mode={result['config']['scan_mode']}",
        f"transfer={p['transfer']}",
        f"joint fit: lag_centroid={p['lag']:.4f}, tau_drw={p['tau_drw']:.4f}, rho={p['rho']:.4f}",
        f"joint log likelihood={result['joint_loglike']:.3f}",
        f"optimizer converged={result['optimizer_diagnostics']['converged']}",
    ]
    if result["fit_score_kind"] != "data_loglike":
        lines.append(f"penalized joint objective={result['joint_objective']:.3f}")
    if transfer_has_width(p["transfer"]):
        width_label = (
            "fitted" if result["config"]["fit_transfer_width"] else "fixed"
        )
        lines.insert(
            3,
            f"transfer_width={p['transfer_width']:.4f} ({width_label})",
        )
    if p.get("transfer_weights") is not None:
        w = np.asarray(p["transfer_weights"], float)
        c = np.asarray(p["transfer_component_centers"], float)
        lines.append("basis weights=" + ", ".join(f"{x:.4f}" for x in w))
        lines.append("basis centers=" + ", ".join(f"{x:.4f}" for x in c))
        lines.append(f"basis sigma={p['transfer_component_sigma']:.4f}")
    lines.append(f"HESSE status={uncertainties.get('status', 'unavailable')}")
    if uncertainties.get("message"):
        lines.append(f"HESSE note={uncertainties['message']}")
    for name, item in physical.items():
        value = item.get("value")
        error = item.get("error")
        if value is None:
            continue
        if error is None:
            lines.append(f"parameter {name}={value:.6g} (not computed or fixed)")
        else:
            lines.append(f"parameter {name}={value:.6g} +/- {error:.6g}")
    if summary.get("lag_mle") is not None:
        lines.append(f"profile/fixed scan optimum lag={float(summary['lag_mle']):.4f}")
        if summary["scan_score_kind"] == "data_loglike":
            lines.append(f"LR 68 regions={summary['lr68_regions']}")
            lines.append(f"LR 95 regions={summary['lr95_regions']}")
        else:
            lines.append("scan uses a penalized objective; ordinary LR regions are unavailable")
    return "\n".join(lines)
