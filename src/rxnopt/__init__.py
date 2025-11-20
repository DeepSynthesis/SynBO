"""Modern Reaction Optimization Framework.

A sophisticated framework for optimizing chemical reactions using
Bayesian Optimization with modern Python practices and rich output.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Zhenzhi Tan"
__email__ = "zhenzhi-tan@outlook.com"

# Core classes
from .rxnopt import ReactionOptimizer
from .initialize import Initializer
from .optimize import Optimizer

__all__ = [
    "ReactionOptimizer",
    "Initializer",
    "Optimizer",
]
