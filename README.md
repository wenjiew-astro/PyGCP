# PyGCP

PyGCP is a Python implementation of Gaussian-copula-process reverberation
mapping for estimating the lag between a continuum light curve and an
emission-line light curve.

The distribution name is `pygcp-rm`; the Python import name is `pygcp`.
The underlying reverberation-mapping model is referred to as GCP-RM.

The model Gaussianizes the observed marginal flux distributions with KDE,
propagates measurement errors into score space, and fits an OU/DRW covariance
model convolved with a configurable transfer function.

## Features

- Joint maximum-likelihood estimation of lag, DRW timescale and correlation.
- Measurement-error-aware Gaussian-copula likelihood.
- Automatically fitted transfer widths, with an explicit fixed-width option.
- Transfer models: `delta`, `tophat`, `isosceles_triangle`,
  `forward_triangle`, `long_tail`, `double_exponential`, `edge_on_ring`, and
  the flexible three-Gaussian model `flexible_gaussian3`.
- Continuous three-Gaussian centroids and error-controlled OU covariance
  integration, independent of the old global midpoint grid.
- Optional lag profile, likelihood-ratio regions and HESSE uncertainties.
- Automatic transfer-model selection using blocked predictive validation,
  AIC, AICc or BIC.

## Installation

Install from PyPI:

```bash
python -m pip install pygcp-rm
```

Matplotlib is installed automatically. Plotting support is part of the standard
package and requires no extra installation.

Verify the import:

```python
import pygcp
print(pygcp.__version__)
```

From the project directory, install the local source instead:

```bash
python -m pip install .
```

## Test the installation

Run the built-in fitting test from a terminal (Command Prompt or PowerShell on
Windows):

```bash
python -m pygcp.test_pygcp
```

The example data are included in the installed package. This command works from
any directory and does not require downloading the source repository or a
separate `tests/` folder. It runs a top-hat fit and prints a report followed by
`PyGCP installation test passed.` The fit can take some time; by default it does
not save files, create plots, scan lags, or compute HESSE errors. This is an
installation smoke test, not a scientific accuracy or uncertainty validation.

Run shell commands at the terminal prompt, **not** at Python's `>>>` prompt.
If you are already inside Python, enter `exit()` first. On Windows, use a
writable user directory rather than `C:\Windows\SysWOW64` or another system
directory when saving output.

To save fitted results and plots explicitly:

```bash
python -m pygcp.test_pygcp --save --plot --output-prefix results/example
```

For the full set of example options, run `python -m pygcp.test_pygcp --help`.

## Input format

`continuum` and `line` may be paths or NumPy arrays. Each table must contain
at least five finite rows and either two or three columns:

```text
time  flux  [flux_error]
```

Missing error columns are interpreted as zero measurement error. Rows are
sorted by time automatically.
Zero measurement errors are also supported with `sigma_z_floor=0` or `None`.
`load_data` drops non-finite rows by default. With `drop_nan=False`, it raises
an error for non-finite time, flux, or error values. The public covariance
builder also rejects non-finite scalar parameters and time arrays.

## Single-model fit

Choose a transfer model with the `transfer` argument. All eight available
models are listed below; the default is `tophat`.

| `transfer`           | Response model                                |
| ---------------------- | --------------------------------------------- |
| `delta`              | Delta response (no smoothing)                 |
| `tophat`             | Top-hat response (default)                    |
| `isosceles_triangle` | Symmetric triangular response                 |
| `forward_triangle`   | Asymmetric triangular response                |
| `long_tail`          | Long-tailed response                          |
| `double_exponential` | Exponential rise and decay response           |
| `edge_on_ring`       | Edge-on ring response (requires positive lag) |
| `flexible_gaussian3` | Flexible three-Gaussian mixture response      |

```python
from pygcp import format_report, run_mle

result = run_mle(
    "continuum.txt",
    "line.txt",
    lag_min=-10.0,
    lag_max=10.0,
    lag_step=0.1,
    transfer="tophat",
    fit_transfer_width=True,
    compute_hesse_errors=False,
    run_scan=False,
    save=False,
    make_figures=False,
    show_figures=False,
    output_prefix="results/example",
)

print(result["p_hat"]["lag"])
print(format_report(result))
```

This example fits the joint model without plotting, writing output files,
computing HESSE errors, or scanning a lag profile. The latter two steps can
be expensive; disabling plotting alone does not disable the scan. These
explicit options only control this example, not the package's API defaults.
In particular, `run_mle` still defaults to `run_scan=True`.

For a runnable example with bundled data, use the installed test module:

```bash
python -m pygcp.test_pygcp
```

See [tests/README.md](https://github.com/wenjiew-astro/PyGCP/blob/main/tests/README.md)
for plotting and lag-scan examples.

### Saving results and choosing the output directory

`output_prefix` combines an output directory and a filename prefix; it is not
just a directory name. With `save=True` and
`output_prefix="results/example"`, a fit without a lag scan writes:

```text
results/example_mle_params.txt
results/example_mle_summary.txt
results/example_parameter_uncertainties.json
```

The JSON file records uncertainty diagnostics even when HESSE is disabled.
`run_scan=True` additionally writes `results/example_mle_scan.txt`.
`save=True` does not enable plotting; `make_figures=True` independently saves
figures under the same prefix, even with `save=False`.

Missing output directories are created automatically. Relative paths are
resolved against the **current working directory** (check with `Path.cwd()`),
not the package's installation directory or the input-data directory. Without
an explicit `output_prefix`, `run_mle` uses `"gcp_rm"`, so `save=True` writes
`gcp_rm_mle_params.txt`, `gcp_rm_mle_summary.txt`, and
`gcp_rm_parameter_uncertainties.json` directly in the current directory.
Use a different prefix for each run to avoid overwriting existing output files.

For an absolute output path in your user directory on Windows, Linux, or macOS:

```python
from pathlib import Path
from pygcp import run_mle

result = run_mle(
    "continuum.txt",
    "line.txt",
    save=True,
    output_prefix=str(Path.home() / "PyGCP-results" / "run1"),
    run_scan=False,
    compute_hesse_errors=False,
    make_figures=False,
    show_figures=False,
)
```

Replace the two input paths with your own light-curve files. This example saves
results in the `PyGCP-results` folder under your user directory, with filenames
starting with `run1`.

### Model settings and diagnostics

When `fit_transfer_width=True` (the default), no width needs to be supplied.
The optimizer uses the logarithmic midpoint of `transfer_width_bounds` only as
an initial value and estimates the final width from the data. To hold a width
fixed, set `fit_transfer_width=False` and provide `transfer_width` explicitly.
Models without a free width (`delta` and `edge_on_ring`) do not report a
placeholder width.

Fits require non-negative transfer-delay support by default
(`enforce_nonnegative_transfer=True`), including automatic model selection.
For `tophat`, this requires `lag - transfer_width / 2 >= 0`; a positive
`lag_min` alone does not ensure this. Set `enforce_nonnegative_transfer=False`
explicitly if you want to allow negative-delay responses. This constraint does
not change the transfer formulas or fix the fitted width.

For constrained joint `tophat` fits with a fitted width, the optimizer uses
`transfer_width_fraction` in `[0, 1]`, mapped to the same physical interval
`[width_min, min(width_max, 2 * lag)]`. Candidate starts are feasible and the
initial lag step no longer scales with the requested lag range. This avoids
the discontinuous penalty at the coupled support boundary without changing
the likelihood, priors, parameter count, or configured physical bounds.
`result["p_hat"]["transfer_width"]` remains the physical width; only the
internal optimizer coordinate changes. HESSE includes the full coordinate
Jacobian. Fits at an active boundary report a warning: their symmetric HESSE
errors are not validated confidence intervals. Range stability is a numerical
check, not proof of a global optimum or agreement with the true lag.

Matplotlib is installed automatically with PyGCP. Set `show_figures=False`
when running on a server where figures should only be saved and not opened
interactively.
`make_figures=True` saves plots independently of `save`, and creates missing
output directories. With `run_scan=False`, the light-curve and score plots
are still available; `save=True` still writes fitted parameters, a summary,
and uncertainty diagnostics, without a scan file.
No scan grid is constructed for `run_scan=False`, so a narrow fitting range
works even when `lag_step` is larger than that range. When scanning, the grid
includes both requested bounds and never extrapolates beyond them. The last
interval can be shorter than `lag_step`; a coarser step uses just the endpoints.
Displayed relative densities and reported regions use the requested interval
as their boundary, rather than extending a half-cell outside it.

To opt into plots, add `make_figures=True` and
`output_prefix="results/gcp_rm"` to the fit above. This saves
`results/gcp_rm_mle_lightcurves.png` and
`results/gcp_rm_mle_scores_with_errors.png`. Add `show_figures=True` only if
you want an interactive display. A lag-profile plot additionally requires
`run_scan=True`; ordinary, unpenalized scans also provide a likelihood-ratio
plot. Scans and HESSE calculations are optional uncertainty diagnostics,
not part of the quick-start example.

Fits return `optimizer_diagnostics` even when HESSE is disabled. If no joint
optimization converges, fitting raises `RuntimeError` instead of returning
unconverged parameters. Joint fits currently use `minuit_n_starts=10` by default.
For `tophat`, this generates 41 candidate lag starts and locally optimizes
the 10 highest-scoring initial points. Set `minuit_n_starts=30` explicitly
to optimize 30 starts. Neither setting guarantees a global optimum.
Increase `minuit_max_calls` or `minuit_n_starts` when
appropriate. Failed profile optimizations produce a warning and are recorded
in `params_at_lag` and the summary; physically forbidden lags are recorded
separately.

## Flexible transfer function

```python
result = run_mle(
    "continuum.txt",
    "line.txt",
    transfer="flexible_gaussian3",
    flexible_sigma_ratio=0.45,
    flexible_smoothness=0.25,
    flexible_weight_logit_bound=5.0,
    flexible_initial_weights=(0.25, 0.50, 0.25),
    run_scan=False,
    compute_hesse_errors=False,
    save=False,
    make_figures=False,
    show_figures=False,
    output_prefix="results/flexible",
)
```

These options are ordinary Python arguments and do not depend on environment
variables.
`flexible_smoothness` retains its default of `0.25`, penalizing differences
between adjacent mixture weights. Set it to zero for unpenalized maximum
likelihood. `joint_loglike` always reports
the data log likelihood, while `joint_objective` reports the optimized value
including any smoothing or priors. `fit_score_kind` identifies the objective.
Penalized scans are marked by `summary["scan_score_kind"]` and do not report
ordinary likelihood-ratio confidence regions. `model_information_criteria`
requires a converged input. For a fit with smoothing or priors, it performs
a separate unpenalized scoring fit without changing the input configuration
or fitted parameters. `information_criteria_refit` identifies that case, and
`information_criteria_fit` contains the scoring parameters and diagnostics.
The scoring fit tries both the original fitted parameters and the established
default starts, using the original MIGRAD settings. It rejects scores worse
than the known original-parameter likelihood beyond floating-point tolerance.
This check does not certify a global optimum; inspect convergence diagnostics
and use adequate optimization budgets.

### Three-Gaussian numerical implementation

The common standard deviation is `flexible_sigma_ratio * transfer_width`;
`transfer_width` is the spacing between adjacent Gaussian centers, not the
standard deviation or full width of the final response. The three positive
basis weights have two independent softmax logits. With a fitted width, the
response has four independent parameters (centroid, spacing, and two logits),
or six joint parameters after including the DRW timescale and correlation.
The fixed ratio `0.45` and smoothing coefficient `0.25` are not fitted parameters.

With `enforce_nonnegative_transfer=True`, the Gaussian mixture is truncated at
zero and normalized on the positive half-line. The joint optimizer now uses
the middle Gaussian center internally and derives `p_hat["lag"]` from the
**continuous** truncated response's mean. The requested lag bounds constrain
that derived mean. This retains the response family and physical bounds but
corrects the historical discretized lag-to-response mapping; historical
numerical results are not guaranteed to stay unchanged.

The centroid inverse can have multiple center solutions. Fixed-lag scans
compare the solutions with the same objective rather than choosing a root
according to a solver bracket. Low-level `make_transfer_quadrature` and
`build_reverberation_covariance` calls must supply `flexible_center` (the middle
Gaussian center) when the requested centroid is ambiguous. This is a branch
identifier, not an additional fitted parameter. For a fitted result, reuse
`result["p_hat"]["transfer_component_centers"][1]` together with the matching
lag, width and weights. Ambiguous calls raise an explicit error.

For this model, covariance calculations use analytic Gaussian-exponential
cross integrals and an error-controlled integration of the continuous
delay-difference density for the line autocovariance. `transfer_grid_size`
is an initial numerical resolution, not a fixed 61-point response or a
model parameter. Increasing it does not recalibrate centers or select a
different response branch. Returned standalone response quadrature nodes
may be nonuniform; the old equal-grid autocorrelation shortcut must not be
used with them. Numerical convergence errors are explicit rather than
silently reclassified as forbidden physical support.

These changes do not remove the weight penalty: the default flexible fit
still optimizes the data log likelihood minus
`0.25 * ((a[1]-a[0])**2 + (a[2]-a[1])**2)`, not unpenalized MLE.

## Automatic transfer-model selection

```python
from pygcp import run_transfer_model_selection

selection = run_transfer_model_selection(
    "continuum.txt",
    "line.txt",
    candidate_transfers=("delta", "tophat", "flexible_gaussian3"),
    selection_metric="blocked_cv",
    scan_models="none",
    parallel=True,
    save=False,
    make_figures=False,
    show_figures=False,
    compute_hesse_errors=False,
    lag_min=-10.0,
    lag_max=10.0,
    lag_step=0.1,
    output_prefix="results/model_selection",
)

print(selection["best_transfer"])
best = selection["models"][selection["best_transfer"]]["result"]
print(best["p_hat"]["lag"])
```

Model selection performs multiple fits and cross-validation, so it can take
substantially longer than a single-model fit even without plots or scans.
The example above creates neither plots nor lag profiles.

To request plots of the selected profiled model explicitly, use
`scan_models="best"`, `make_figures=True`, and
`output_prefix="results/gcp_rm_auto"`. Plots are then saved under
`results/gcp_rm_auto_models/<transfer>/`.
The model-output folder is derived from the complete prefix name:
`<output_prefix>_models/<transfer>/`. Different prefixes in the same parent
directory therefore keep their parameters and plots separate.
This also works with `save=False, make_figures=True`. Automatic selection
preserves each candidate's smoothing settings, including the default `0.25`
for `flexible_gaussian3`. `blocked_cv` fits and validates that model directly;
ancillary AIC/BIC fields are left empty for penalized candidates. When selecting
with AIC/AICc/BIC, an additional unpenalized fit supplies the information
criteria. The returned model and its lag scan still use the requested smoothing.
As in the original implementation, automatic selection uses `use_prior=False`;
explicit prior requests produce a warning. Convergence diagnostics survive
parallel execution. A requested scan failure raises an error instead of silently
returning an unscanned result.

## Public API

The supported entry points are exported directly by `pygcp`:

- `run_mle`
- `run_transfer_model_selection`
- `load_data`
- `build_config`
- `make_transfer_quadrature`
- `build_reverberation_covariance`
- `model_information_criteria`
- `format_report`

The full numerical implementation lives in `pygcp.core`.
The default output prefixes, `gcp_rm` and `gcp_rm_auto`, control output filenames
and are independent of the Python import name.

## Development

The installed package includes the fitting test `pygcp.test_pygcp` and its two
input tables. The source-only `tests/` directory contains a wrapper for this
example and the numerical regression suite; this directory itself is not
installed by a wheel. To run the regression suite, download a source checkout
or unpack the source distribution, then run from its project root:

```bash
python -m pip install .
python -m unittest discover -s tests -p 'test_*.py' -v
```

These tests use bundled code and small in-memory arrays; no external simulation
archive or historical distribution is required. See
[tests/README.md](https://github.com/wenjiew-astro/PyGCP/blob/main/tests/README.md)
for the example and test instructions. Optional development tools can
be installed with `python -m pip install '.[dev]'`. These are build, test, and
upload tools; they are not required for fitting or plotting.

Author: Wenjie Wang (<wenjiew001@gmail.com>).

## License

PyGCP is distributed under the MIT License. See
[LICENSE.txt](https://github.com/wenjiew-astro/PyGCP/blob/main/LICENSE.txt).
