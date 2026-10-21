# PyGCP Examples and Numerical Tests

This directory contains a runnable fitting example and numerical regression tests
for the three-Gaussian response. All required files are included in the source
distribution and source repository; no external experiment directories or
historical validation archives are needed.

- `example.py`: reads the data in this directory, runs one fit, and prints a report.
- `continuum` and `line`: example inputs with three columns: time, flux, and
  measurement error.
- `test_flexible_numerics.py`: regression tests for the three-Gaussian numerical
  implementation.
- `test_package.py`: public `pygcp` imports, typing marker, and `pygcp-rm`
  distribution metadata (the metadata check requires an installed package).

## Installation and Usage

Unpack the source distribution or download the source repository, then enter the
project root containing `pyproject.toml` and run:

```bash
python -m pip install .
python tests/example.py
```

The distribution is installed as `pygcp-rm`, but Python code imports `pygcp`
(for example, `from pygcp import run_mle`).

The default example uses the `tophat` model and fits the transfer-function width.
It does not generate plots, scan lags, compute HESSE errors, or save output files.
Model settings and the number of optimization starts use the package defaults;
the fit still takes some time to run.

Input data paths are resolved relative to the script itself, so you can also run
the script by its absolute path from another directory. Relative paths supplied
to `--output-prefix` are resolved from the current working directory.

## Plotting and Lag Scans

To save light-curve and Gaussian-score plots without running an additional lag
scan:

```bash
python tests/example.py --plot
```

Matplotlib is installed as a required dependency; no plotting extra is needed.
Plots are saved under `results/` in the current working directory without opening
a window. Plotting does not automatically enable a lag scan. To request a scan
and its diagnostic plots explicitly:

```bash
python tests/example.py --plot --scan
```

Scanning performs repeated conditional optimizations and increases runtime. The
quick example without a scan or HESSE does not provide the corresponding
uncertainty intervals; a single point estimate is not a complete uncertainty
analysis.

## Changing the Model and Search Range

For example, select the flexible three-Gaussian model and set the lag search range:

```bash
python tests/example.py --transfer flexible_gaussian3 --lag-min 0.1 --lag-max 10
python tests/example.py --help
```

## Running Numerical Regression Tests

From the project root, use Python's standard-library test runner; pytest is not
required:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover boundary parameters, continuous centroids, multiple centre
branches, independent integration comparisons, covariance positive
semidefiniteness, tighter integration tolerances, and preservation of the original
model settings and weight-smoothing penalty. They use small in-memory arrays and
fixed parameters; no additional simulation data need to be downloaded.

## Source Distributions and Wheels

The source distribution includes this directory. A normal wheel installation
installs `pygcp`, the license, and dependencies, but does not place `tests/` in
your working directory. If you have already installed the package with pip,
download the matching source version separately to run these examples and tests.

The example data demonstrate API usage; they are not a correctness or ground-truth
validation for every model.
