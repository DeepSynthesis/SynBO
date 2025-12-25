# ReactionOpt

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/reactionopt.svg)](https://badge.fury.io/py/reactionopt)

A multi-objective reaction optimization framework based on Bayesian Optimization using Botorch & Ax.

## 🎯 Overview

ReactionOpt is a Python package designed for optimizing chemical reactions using advanced machine learning techniques. It leverages Bayesian Optimization to efficiently explore reaction spaces and optimize multiple objectives simultaneously (e.g., yield and enantioselectivity).

### Key Features

- **Multi-objective optimization** for reaction conditions (yield & ee optimization)
- **Bayesian Optimization** powered by [Botorch](https://github.com/pytorch/botorch) & [Ax](https://github.com/facebook/Ax)
- **GPU acceleration** for large-scale optimization
- **Flexible descriptor handling** for various reaction parameters
- **Automated visualization** of optimization results
- **High-throughput experimentation** support

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install rxnopt
```

### Development Installation

```bash
git clone https://github.com/yourusername/reactionopt.git
cd reactionopt
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- PyTorch >= 1.9.0
- Botorch >= 0.6.0
- RDKit >= 2021.9.1
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)
