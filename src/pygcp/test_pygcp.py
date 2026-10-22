"""Test an installed PyGCP package with a fit to bundled example data.

Run ``python -m pygcp.test_pygcp`` in a terminal, not at Python's ``>>>`` prompt.
"""

import argparse
from importlib import resources
from pathlib import Path

import numpy as np

from . import SUPPORTED_TRANSFERS, __version__, format_report, run_mle


def _load_example_data():
    """Read package resources without depending on the working directory."""
    tables = []
    for filename in ("continuum.txt", "line.txt"):
        resource = resources.files("pygcp").joinpath("data").joinpath(filename)
        with resource.open("r", encoding="utf-8") as stream:
            tables.append(np.loadtxt(stream))
    return tuple(tables)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", choices=SUPPORTED_TRANSFERS, default="tophat")
    parser.add_argument("--lag-min", type=float, default=-10.0)
    parser.add_argument("--lag-max", type=float, default=10.0)
    parser.add_argument("--save", action="store_true", help="Save fitted results.")
    parser.add_argument(
        "--plot", action="store_true",
        help="Save plots (Matplotlib is installed by default).",
    )
    parser.add_argument(
        "--scan", action="store_true", help="Also compute the slower lag profile."
    )
    parser.add_argument(
        "--output-prefix", default="results/example",
        help="Output directory and filename prefix (default: results/example).",
    )
    args = parser.parse_args(argv)

    continuum, line = _load_example_data()
    print(f"Running PyGCP {__version__} installation test (this performs a fit)...",
          flush=True)
    result = run_mle(
        continuum,
        line,
        transfer=args.transfer,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        lag_step=0.1,
        run_scan=args.scan,
        compute_hesse_errors=False,
        save=args.save,
        make_figures=args.plot,
        show_figures=False,
        output_prefix=args.output_prefix,
    )
    print(format_report(result))
    if args.save or args.plot:
        print(f"Output prefix: {Path(args.output_prefix).resolve()}")
    print("PyGCP installation test passed.")
    return result


if __name__ == "__main__":
    main()
