"""Run one PyGCP fit using the small tables beside this file."""

import argparse
from pathlib import Path

from pygcp import SUPPORTED_TRANSFERS, format_report, run_mle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", choices=SUPPORTED_TRANSFERS, default="tophat")
    parser.add_argument("--lag-min", type=float, default=-10.0)
    parser.add_argument("--lag-max", type=float, default=10.0)
    parser.add_argument(
        "--plot", action="store_true",
        help="Save plots (plotting support is installed by default).",
    )
    parser.add_argument(
        "--scan", action="store_true", help="Also compute the slower lag profile."
    )
    parser.add_argument("--output-prefix", default="results/example")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    result = run_mle(
        data_dir / "continuum",
        data_dir / "line",
        transfer=args.transfer,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        lag_step=0.1,
        run_scan=args.scan,
        compute_hesse_errors=False,
        save=False,
        make_figures=args.plot,
        show_figures=False,
        output_prefix=args.output_prefix,
    )
    print(format_report(result))
    if args.plot:
        print(f"Plots saved with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
