"""pyaugur - Python port of Augur: cell type prioritization in single-cell data."""

from .core import calculate_auc
from .differential import calculate_differential_prioritization
from .feature_selection import select_variance, select_random

__version__ = "0.1.0"
__all__ = [
    "calculate_auc",
    "calculate_differential_prioritization",
    "select_variance",
    "select_random",
]
