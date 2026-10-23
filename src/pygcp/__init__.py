"""Public API for PyGCP, the GCP-RM reverberation-mapping package."""

from .core import (
    SUPPORTED_TRANSFERS,
    build_config,
    build_reverberation_covariance,
    format_report,
    load_data,
    make_transfer_quadrature,
    model_information_criteria,
    run_mle,
    run_transfer_model_selection,
)

__version__ = "0.1.0"

__all__ = [
    "SUPPORTED_TRANSFERS",
    "__version__",
    "build_config",
    "build_reverberation_covariance",
    "format_report",
    "load_data",
    "make_transfer_quadrature",
    "model_information_criteria",
    "run_mle",
    "run_transfer_model_selection",
]
