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
pip install reactionopt
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

- Python 3.8+
- PyTorch >= 1.9.0
- Botorch >= 0.6.0
- Ax-platform >= 0.2.0
- RDKit >= 2021.9.1
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

## 🔧 Quick Start

### Basic Usage

```python
from rxnopt import ReactionOptimizer
import pandas as pd

# Initialize the optimizer
optimizer = ReactionOptimizer(
    objectives=['yield', 'ee'],  # Multi-objective optimization
    n_initial_points=10,
    n_iterations=50
)

# Load your reaction data
data = pd.read_csv('your_reaction_data.csv')

# Run optimization
results = optimizer.optimize(
    data=data,
    parameter_space={
        'temperature': [20, 80],
        'catalyst_loading': [0.1, 10.0],
        'solvent': ['DCM', 'THF', 'Toluene']
    }
)

# Visualize results
optimizer.plot_optimization_history()
optimizer.plot_pareto_front()
```

### Advanced Configuration

```python
# Custom acquisition function and surrogate model
optimizer = ReactionOptimizer(
    objectives=['yield', 'ee'],
    acquisition_function='EHVI',  # Expected Hypervolume Improvement
    surrogate_model='GP',         # Gaussian Process
    n_initial_points=20,
    n_iterations=100,
    random_seed=42,
    device='cuda'  # GPU acceleration
)
```

## 📊 Examples

Check out the `examples/` directory for detailed examples:

- `examples/1430-ultra-HTE/`: Ultra-high throughput experimentation example
- `examples/basic_optimization.py`: Simple reaction optimization
- `examples/multi_objective.py`: Multi-objective optimization tutorial

## 🧪 Project Structure

```
reactionopt/
├── rxnopt/                    # Main package
│   ├── __init__.py
│   ├── rxnopt.py             # Main ReactionOptimizer class
│   ├── initialize.py         # Initialization strategies
│   ├── optimize.py           # Optimization algorithms
│   ├── bo_algorithm/         # Bayesian optimization algorithms
│   │   ├── acf_opt.py       # Acquisition function optimization
│   │   └── GP_opt.py        # Gaussian Process implementation
│   └── utils/               # Utility functions
│       ├── utils.py
│       ├── visualization.py
│       └── write_excel.py
├── tests/                   # Test suite
├── docs/                   # Documentation
├── examples/              # Example notebooks and scripts
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rxnopt

# Run specific test file
pytest tests/test_optimizer.py
```

## 📖 Documentation

For detailed documentation, visit: [https://reactionopt.readthedocs.io](https://reactionopt.readthedocs.io)

Or build locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

## 🗺️ Roadmap

### ✅ Completed
- [x] Basic pipeline from initialization to automated file I/O
- [x] Reaction space construction efficiency optimization  
- [x] GPU model migration

### 🔄 In Progress (Short-term)
- [ ] Additional surrogate models (RF, NN, GNN)
- [ ] More acquisition functions (EHVI, UCB, Utopia Point)
- [ ] Alternative algorithms (heuristic learning, simulated annealing, evolutionary algorithms)

### 🔮 Future Plans (Medium-term)
- [ ] High-throughput dataset testing (ee + yield)
- [ ] Automated visualization of optimization results
- [ ] Interpretability analysis using LIME
- [ ] Integration with experimental workflows

### 🚀 Long-term Vision
- [ ] Automated reaction space reduction
- [ ] Transfer learning for substrate changes
- [ ] Multi-objective weight calibration
- [ ] Error handling for yield/ee measurements
- [ ] Restart algorithms and stopping criteria

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/yourusername/reactionopt](https://github.com/yourusername/reactionopt)

## 🙏 Acknowledgments

- [Botorch](https://github.com/pytorch/botorch) for Bayesian optimization
- [Ax](https://github.com/facebook/Ax) for experiment management
- [RDKit](https://github.com/rdkit/rdkit) for molecular descriptors
- The chemical informatics and machine learning communities

## 📚 Citation

If you use ReactionOpt in your research, please cite:

```bibtex
@software{reactionopt2024,
  title={ReactionOpt: A Multi-objective Reaction Optimization Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/reactionopt}
}
```

---

**Happy Optimizing! 🧪⚗️**