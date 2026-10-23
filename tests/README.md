# PyGCP Examples and Numerical Tests

This directory contains a wrapper for the installed fitting example and numerical regression tests for the three-Gaussian response. No external experiment directories or historical validation archives are needed.

- `example.py`: source-tree wrapper for `python -m pygcp.test_pygcp`.
- `continuum` and `line`: example inputs with three columns: time, flux, and measurement error.
- `test_flexible_numerics.py`: regression tests for the three-Gaussian numerical implementation.
- `test_package.py`: public `pygcp` imports, typing marker, and `pygcp-rm` distribution metadata (the metadata check requires an installed package).

## Installation and Usage

After installing `pygcp-rm`, run the built-in example from any directory:

```bash
python -m pygcp.test_pygcp
```

This command runs in your terminal, not inside Python's `>>>` prompt. Use `exit()` to leave the Python interpreter before entering it. The data are loaded from installed package resources, so no local `tests/` folder or input files are required. Use a writable user directory when saving files, not a Windows system directory such as `C:\Windows\SysWOW64`.

The distribution is installed as `pygcp-rm`, but Python code imports `pygcp` (for example, `from pygcp import run_mle`).

The default example uses the `tophat` model and fits the transfer-function width. It does not generate plots, scan lags, compute HESSE errors, or save output files. Model settings and the number of optimization starts use the package defaults; the fit still takes some time to run.

The bundled data match `continuum` and `line` in this source directory. Relative paths supplied to `--output-prefix` are resolved from the current working directory. Model fitting defaults are unchanged.

## Plotting and Lag Scans

To save light-curve and Gaussian-score plots without running an additional lag scan:

```bash
python -m pygcp.test_pygcp --plot
```

Matplotlib is installed as a required dependency; no plotting extra is needed. Plots are saved under `results/` in the current working directory without opening a window. Plotting does not automatically enable a lag scan. To request a scan and its diagnostic plots explicitly:

```bash
python -m pygcp.test_pygcp --plot --scan
```

Scanning performs repeated conditional optimizations and increases runtime. The quick example without a scan or HESSE does not provide the corresponding uncertainty intervals; a single point estimate is not a complete uncertainty analysis.

## Saving Results and Choosing the Output Directory

To save fitted parameters, a summary, uncertainty diagnostics, and plots:

```bash
python -m pygcp.test_pygcp --save --plot --output-prefix results/example
```

`results/` is created below the current working directory, and output filenames start with `example`. `--output-prefix` can also be an absolute path. For example, in Windows Command Prompt, to save inside your user directory:

```bat
python -m pygcp.test_pygcp --save --output-prefix "%USERPROFILE%\PyGCP-results\example"
```

The default command saves nothing. `--save` saves numerical results, `--plot` saves figures, and neither option implies the other. Use a new prefix for each run to avoid overwriting existing output files.

## Changing the Model and Search Range

For example, select the flexible three-Gaussian model and set the lag search range:

```bash
python -m pygcp.test_pygcp --transfer flexible_gaussian3 --lag-min 0.1 --lag-max 10
python -m pygcp.test_pygcp --help
```

## Running Numerical Regression Tests

The regression suite is source-only. Download the source repository or unpack the matching source distribution, enter the project root containing `pyproject.toml`, and install that source version first. Then use Python's standard-library test runner; pytest is not required:

```bash
python -m pip install .
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover boundary parameters, continuous centroids, multiple centre branches, independent integration comparisons, covariance positive semidefiniteness, tighter integration tolerances, and preservation of the original model settings and weight-smoothing penalty. They use small in-memory arrays and fixed parameters; no additional simulation data need to be downloaded.

## Source Distributions and Wheels

Both wheels and source distributions include `pygcp.test_pygcp` and its example data. The source distribution additionally includes this `tests/` directory. An ordinary wheel installation does not place a `tests/` folder in your working directory. The older `python tests/example.py` invocation is a source-tree shortcut only; the installed `python -m pygcp.test_pygcp` command is preferred.

The example data demonstrate API usage; they are not a correctness or ground-truth validation for every model.
