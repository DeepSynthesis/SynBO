# SynBO: Synthetic Bayesian Optimization for Reaction Condition Screening

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SynBO** (Synthetic Bayesian Optimization) is an intelligent reaction optimization tool designed specifically for synthetic chemists. It uses Bayesian Optimization (BO) algorithms to help you find optimal reaction conditions with minimal experimental effort.

---

## Why Do Chemists Need SynBO?

Optimizing a new chemical reaction typically involves screening numerous combinations of reaction conditions:

- **Catalysts** (various organocatalysts or metal complexes)
- **Solvents** 
- **Bases/Additives** (acids/bases, ligands, electrolyte etc.)
- **Temperature**
- **Concentration**
- **reaction time**, etc.

The traditional approach is **OFAT** (One-Factor-At-A-Time). But with 5 catalysts × 5 solvents × 4 bases × 4 temperatures = **400 combinations**, this is clearly impractical.

**SynBO's Solution**: Like an experienced chemist, it "learns" from previous experiments and "predicts" which conditions are most likely to succeed next. Typically, you only need **50-80 experiments** to find optimal conditions.

---

## How Does Bayesian Optimization Work?

Imagine you are a mountain climber searching for the highest peak in the dark:

1. **Initialization**: Take a few random steps and record the altitude (corresponds to: randomly run a few experimeecord yield/selectivity)
2. **Build a Mental Map**: Based on where you've been, infer the shape of the entire mountain (corresponds to: algorithm learns reaction patterns)
3. **Intelligent Decision**: Go to places that might be higher (exploitation), but also explore unknown areas (exploration)
4. **Iterate**: Repeat steps 2-3 until you find the highest peak (corresponds to: finding optimal reaction conditions)

**Chemistry Analogy**:
- Just like when you optimize reactions in the lab, adjusting your strategy based on previous rounds
- If a particular catalyst performs well, you'll try similar catalysts (exploitation)
- But you'll also try some conditions that look different, in case you miss something better (exploration)

---

## 🚀 Quick Start

### Installation

```bash
<<<<<<< HEAD
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
=======
# Requires Python 3.12 or higher
pip install synbo
```

### Basic Example: Optimizing a Coupling Reaction

```python
from synbo import ReactionOptimizer
import pandas as pd

# 1. Create optimizer and specify objectives
optimizer = ReactionOptimizer(
    opt_metrics=['yield', 'ee'],  # Optimize both yield and enantioselectivity
    opt_type='auto',               # Auto-detect init or optimization phase
    random_seed=42
)

# 2. Define reaction space (all possible condition combinations)
condition_dict = {
    'catalyst': ['Pd(OAc)2', 'Pd(PPh3)4', 'Pd2(dba)3', 'Xantphos-Pd'],
    'solvent': ['THF', 'Dioxane', 'Toluene', 'DMF', 'MeCN'],
    'base': ['Cs2CO3', 'K2CO3', 'NaOEt', 'DBU', 'Et3N'],
    'temperature': [25, 50, 80, 100]
}
optimizer.load_rxn_space(condition_dict)

# 3. Load molecular descriptors (optional, for more accurate predictions)
# If not provided, system will automatically use OneHot encoding
optimizer.load_desc()

# 4. Run first batch of experiments (recommend 5-10, Latin Hypercube Sampling)
optimizer.run(batch_size=8)

# 5. Save recommended experimental conditions
optimizer.save_results(filetype='csv')  # Generates "recommended_batch_0.csv"

# ============================================
# After completing these experiments in lab, fill results into CSV
# ============================================

# 6. Load completed experimental results
results = pd.read_csv('experimental_results.csv')  # Must contain 'yield' and 'ee' columns
optimizer.load_prev_rxn(results)

# 7. Continue optimization, algorithm recommends next batch based on data
optimizer.run(batch_size=5)
optimizer.save_results(filetype='csv')  # Generates "recommended_batch_1.csv"

# Repeat steps 6-7 until satisfactory yield and selectivity are achieved
```

### Single-Objective Optimization (Yield Only)

```python
optimizer = ReactionOptimizer(
    opt_metrics='yield',  # Only optimize yield
    opt_metric_settings={
        'opt_direct': 'max',      # Maximize
        'opt_range': [0, 100],    # Yield range 0-100%
        'metric_weight': 1.0
    }
)
```

### Multi-Objective Optimization (Yield + Enantioselectivity)

```python
optimizer = ReactionOptimizer(
    opt_metrics=['yield', 'ee'],
    opt_metric_settings=[
        {'opt_direct': 'max', 'opt_range': [0, 100], 'metric_weight': 1.0},  # Yield
        {'opt_direct': 'max', 'opt_range': [0, 100], 'metric_weight': 2.0}   # ee, higher weight
    ]
)
```

---

## 🔬 Advanced Features

### 1. LLM-Powered Analysis of Failed Experiments

When certain condition combinations repeatedly fail, SynBO can call a Large Language Model (LLM) to analyze the causes and automatically exclude these "problematic reagents":

```python
# After round 3, let AI analyze which conditions to avoid
constraints = optimizer.get_constraints(method='llm')

# Apply constraints to next round of optimization
optimizer.run(batch_size=5, constraints=constraints)
```

**Application Scenarios**:
- Discover "DBU + high temperature" always leads to decomposition → Auto-exclude
- Discover "toluene solvent" works best with specific catalyst → Prioritize similar combinations

### 2. Track Optimization Progress (Hypervolume)

In multi-objective optimization, the Hypervolume metric helps you determine if you're approaching the optimum:

```python
# Calculate Hypervolume for current Pareto front
hv = optimizer.calculate_current_hv()
print(f"Current optimization progress: {hv['hv_normalized']*100:.1f}%")

# View progress across rounds
progress = optimizer.calculate_hv_by_batch()
```

**Chemistry Explanation**:
- Hypervolume measures the "performance space" covered by currently found optimal conditions
- When Hypervolume growth slows down, you're near optimal and can consider stopping experiments

### 3. Choose Different Optimization Strategies

```python
# Standard Bayesian Optimization (Recommended)
optimizer.run(optimize_method='default_BO')

# Particle Swarm (suitable for complex nonlinear relationships)
optimizer.run(optimize_method='particle_swarm')

# Evolutionary Algorithm (suitable for discrete space search)
optimizer.run(optimize_method='evolution')

# Random Search (baseline comparison)
optimizer.run(optimize_method='random_select')
```

---

## 📊 The Chemistry Behind the Algorithms

### Surrogate Models — "Predicting Reaction Outcomes"

| Model | Chemistry Intuition | Best For |
|-------|---------------------|----------|
| **GP (Gaussian Process)** | Assumes similar conditions give similar results | Fewer experiments (<50), clear reaction mechanisms |
| **Random Forest** | Voting via multiple decision trees | Many categorical variables (many catalyst/solvent types) |
| **BNN (Neural Network Ensemble)** | Deep learning for complex nonlinear relationships | Large-scale high-throughput screening (>100 experiments) |
| **Bayesian Linear** | Linear approximation, fast but simple | Preliminary screening, need quick results |

### Acquisition Functions — "Choosing the Next Experiment"

| Function | Chemistry Strategy | When to Use |
|----------|-------------------|-------------|
| **EHVI** (Default) | Balance yield and selectivity, find Pareto optimal frontier | Optimizing yield and ee simultaneously, both important |
| **UCB** | Conservative strategy, prioritize high-yield conditions with certainty | Limited time, cannot afford failures |
| **ParEGO** | Transform multi-objective into single-objective | More than 2 objectives (e.g., yield + ee + cost) |
| **NEI** | Account for experimental error | High variability in replicate experiments |

---

## 📁 Real-World Case Studies

### Case 1: Asymmetric Hydrogenation

**Background**: Screening chiral phosphoric acid catalysts for imine asymmetric hydrogenation

```python
condition_dict = {
    'catalyst': ['CPA-1', 'CPA-2', 'CPA-3', 'CPA-4', 'CPA-5', 'CPA-6'],
    'additive': ['MsOH', 'TfOH', 'TFA', 'None'],
    'solvent': ['DCE', 'PhCF3', 'Toluene', 'Et2O'],
    'temperature': [-20, 0, 25, 40],
    'H2_pressure': [1, 10, 20, 50]  # atm
}

# Optimization objective: High yield + High ee
optimizer = ReactionOptimizer(
    opt_metrics=['yield', 'ee'],
    opt_type='auto'
)
```

**Result**: Only 24 experiments needed (vs. 384 full combinations), found conditions with 94% yield and 98% ee.

### Case 2: Buchwald-Hartwig Amination

**Background**: Pd-catalyzed aromatic amination, screening ligand and base combinations

```python
# Use LLM to analyze failed ligand-base combinations
constraints = optimizer.get_constraints(method='llm')
# LLM identifies "XPhos + strong base" leads to catalyst deactivation
# Automatically excludes these combinations, saving experimental time
```

---

## 🔧 Project Structure

```
synbo/
├── synbo.py              # Main optimizer class
├── initialize.py         # Initial sampling strategies (Latin Hypercube, etc.)
├── optimize.py           # Optimization algorithm dispatcher
├── algorithm/
│   ├── bo_core.py        # Bayesian optimization core
│   ├── acq_function.py   # Acquisition functions (EHVI/UCB/ParEGO/NEI)
│   ├── sg_model.py       # Surrogate models (GP/RF/BNN)
│   ├── evolution.py      # Evolutionary algorithm
│   └── particle_swarm.py # Particle swarm algorithm
├── descriptor/           # Molecular descriptor processing (RDKit support)
├── analysis/             # LLM-powered analysis module
└── utils/                # Utility functions (visualization, I/O, etc.)
```

---

## 📚 Citation

If you use SynBO in your research, please cite:

```bibtex
@software{synbo2025,
  title={SynBO: Synthetic Bayesian Optimization for Chemical Reaction Optimization},
  author={Zhenzhi Tan},
  year={2025},
  url={https://github.com/yourusername/synbo}
}
```

---

## 🤝 Contributing

Issues and Pull Requests are welcome! For synthetic chemistry-related feature suggestions, please describe your reaction type and optimization needs in detail.

---

## 📧 Contact

- **Author**: Zhenzhi Tan
- **Email**: zhenzhi-tan@outlook.com

---

**Happy Synthesizing! 🧪⚗️**
>>>>>>> dev-beta
