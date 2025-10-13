"""Modern Reaction Optimization Framework.

A sophisticated framework for optimizing chemical reactions using
Bayesian Optimization with modern Python practices and rich output.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "RxnOpt Team"
__email__ = "contact@rxnopt.ai"

# Core classes
from .rxnopt import ReactionOptimizer
from .initialize import Initializer  
from .optimize import Optimizer

# Utilities
from .utils.utils import (
    array_process,
    cartesian_product_3d,
    done_array_process,
    generate_onehot_desc,
    normalize_data,
    track_called,
)

__all__ = [
    "ReactionOptimizer",
    "Initializer", 
    "Optimizer",
    "array_process",
    "cartesian_product_3d", 
    "done_array_process",
    "generate_onehot_desc",
    "normalize_data",
    "track_called",
]
