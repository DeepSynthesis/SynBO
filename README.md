# SynBO: Synthetic Bayesian Optimization for Reaction Condition Screening

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SynBO** (Synthetic Bayesian Optimization) is an intelligent reaction optimization framework that uses Bayesian Optimization to find optimal reaction conditions with minimal experimental effort.

---

## 🔬 Why SynBO?

Optimizing a chemical reaction typically involves screening dozens to hundreds of condition combinations (catalysts, solvents, bases, temperature, etc.). Traditional OFAT (One-Factor-At-A-Time) quickly becomes impractical — e.g., **5 catalysts × 5 solvents × 4 bases × 4 temperatures = 400 combinations**.

SynBO learns from previous experiments and intelligently recommends the next batch of conditions to test. Typically, only **50–80 experiments** are needed to find optimal conditions.

---

## 📁 Example Project: Cobalt-Catalyzed Asymmetric Reaction

The [`examples/`](examples/) directory contains a complete, runnable example of a cobalt-catalyzed reaction optimization with **5 reagent types** and **2 objectives** (yield + ee):

```
examples/
├── optimization_settings.json          # Optimization goals & settings
├── rxn_space/                          # Reaction space definitions
│   ├── alkali.csv                      #   9 alkali/additive options
│   ├── cobalt_catalyst.csv             #   8 Co-catalyst candidates
│   ├── organo_catalyst.csv             #   9 organocatalyst candidates
│   ├── oxidant.csv                     #   9 oxidant options
│   └── solvent.csv                     # 10 solvent options
├── descriptors/                        # RDKit molecular descriptors
│   ├── alkali_RDKit.csv
│   ├── cobalt_catalyst_RDKit.csv
│   ├── organo_catalyst_RDKit.csv
│   ├── oxidant_RDKit.csv
│   └── solvent_RDKit.csv
└── results/                            # Example optimization outputs
    ├── batch-0_20260420.csv            # Initial sampling results
    ├── batch-0_20260420.xlsx
    ├── batch-1_20260420.csv            # 1st optimization round results
    └── batch-1_20260420.xlsx
```

**Reaction space size**: 9 × 8 × 9 × 9 × 10 = **58,320 possible combinations**

### Step-by-Step Workflow

#### 1. Define Your Reaction Space

Create CSV files for each reagent/condition type under `rxn_space/`. Each file must contain `SMILES` and `name` columns:

```csv
# rxn_space/solvent.csv
SMILES,name
ClCCl,DCM
CC#N,CH3CN
C1CCOC1,THF
...
```

#### 2. Generate Molecular Descriptors

```bash
python scripts/get_desc.py --input rxn_space/solvent.csv --smiles-col 'SMILES' --name-col 'name'
```

Repeat for each reagent type. Outputs go to `descriptors/{reagent}_RDKit.csv`.

#### 3. Define Optimization Goals

Create `optimization_settings.json`:

```json
{
    "reagent_types": ["alkali", "cobalt_catalyst", "organo_catalyst", "oxidant", "solvent"],
    "opt_metrics": ["yield", "ee"],
    "opt_direct_info": [
        {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
        {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0}
    ]
}
```

#### 4. Initialize — Generate First Batch

**CLI:**
```bash
python scripts/initialize.py --project-dir examples --batch-size 8 --sampling-method lhs
```

**Python API:**
```python
from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict

desc_dict, condition_dict = load_desc_dict(
    reagent_types=["alkali", "cobalt_catalyst", "organo_catalyst", "oxidant", "solvent"],
    desc_dir="examples/descriptors",
    name_suffix="_RDKit",
    index_col="name",
    return_condition_dict=True,
)

optimizer = ReactionOptimizer(
    opt_metrics=["yield", "ee"],
    opt_type="init",
    random_seed=42,
    save_dir="examples/results",
)
optimizer.load_rxn_space(condition_dict)
optimizer.load_desc(desc_dict)
optimizer.initialize(batch_size=8, sampling_method="lhs")
optimizer.save_results(filetype="excel")
```

#### 5. Run Experiments & Record Results

Run the recommended experiments in the lab. Fill in the `yield` and `ee` columns in the output file (replace `[exp_data]` with actual measurements).

#### 6. Optimize — Get the Next Batch

**CLI:**
```bash
python scripts/optimize.py --project-dir examples --batch-size 5
```

**Python API:**
```python
from synbo.utils import get_prev_rxn

prev_data = get_prev_rxn("examples/results", "batch-*.csv")

optimizer = ReactionOptimizer(
    opt_metrics=["yield", "ee"],
    opt_type="auto",
    random_seed=42,
    save_dir="examples/results",
)
optimizer.load_rxn_space(condition_dict)
optimizer.load_desc(desc_dict)
optimizer.load_prev_rxn(prev_data)
optimizer.optimize(batch_size=5)
optimizer.save_results(filetype="excel")
```

#### 7. Repeat Steps 5–6 Until Satisfactory Results

---

## 📊 Jupyter Notebook Demo

An interactive Jupyter notebook demonstrating the full optimization workflow with visualizations is available at **[examples/demo_optimization.ipynb](examples/demo_optimization.ipynb)**. It covers:

- Loading the example reaction space and descriptors
- Running initialization and optimization rounds
- Visualizing the Pareto front (yield vs ee trade-off)
- Tracking optimization progress with Hypervolume metrics
- Interpreting explore vs exploit recommendations

> **Run it**: `jupyter notebook examples/demo_optimization.ipynb`

---

## 🚀 Quick Start

### Installation

```bash
pip install synbo
```

### Minimal Python Example

```python
from synbo import ReactionOptimizer

optimizer = ReactionOptimizer(
    opt_metrics=['yield', 'ee'],
    opt_type='auto',
    random_seed=42
)

# Load reaction space
optimizer.load_rxn_space({
    'catalyst': ['Pd(OAc)2', 'Pd(PPh3)4', 'Pd2(dba)3'],
    'solvent': ['THF', 'Dioxane', 'Toluene', 'DMF', 'MeCN'],
    'base': ['Cs2CO3', 'K2CO3', 'NaOEt', 'DBU'],
    'temperature': [25, 50, 80, 100]
})

# Use OneHot encoding (auto-generated when no descriptors provided)
optimizer.load_desc()

# Initial sampling
optimizer.run(batch_size=8)
optimizer.save_results(filetype='csv')

# After experiments, load results and optimize
# optimizer.load_prev_rxn(pd.read_csv('results.csv'))
# optimizer.run(batch_size=5)
```

---

## 🧪 Python API Reference

### ReactionOptimizer

```python
optimizer = ReactionOptimizer(
    opt_metrics=["yield", "ee"],
    opt_metric_settings=[
        {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
        {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
    ],
    opt_type="auto",    # "init" | "opt" | "auto"
    random_seed=42,
    save_dir="./results",
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `load_rxn_space(condition_dict)` | Load reaction space (all possible reagent combinations) |
| `load_desc(desc_dict=None)` | Load molecular descriptors (OneHot encoding used if None) |
| `load_prev_rxn(df)` | Load previous experimental results for optimization |
| `initialize(batch_size, sampling_method)` | Generate initial batch (LHS/Sobol/K-Means/Random) |
| `optimize(batch_size, constraints)` | Run Bayesian optimization to recommend next batch |
| `save_results(filetype)` | Save recommendations to CSV/Excel/JSON |
| `calculate_current_hv()` | Calculate current Hypervolume (multi-objective progress) |
| `calculate_hv_by_batch()` | Track Hypervolume across optimization rounds |

---

## 📈 Understanding Optimization Results

### Predictions with Uncertainty

Output files include predicted values with uncertainties:

| batch | alkali | cobalt_catalyst | ... | pred yield | pred ee | yield | ee |
|-------|--------|-----------------|-----|------------|---------|-------|-----|
| 1 | DBU | [Co]-5 | ... | 62.35±3.12 | 85.20±2.87 | [exp_data] | [exp_data] |

- **`pred yield` / `pred ee`**: Model prediction ± uncertainty
- **`[exp_data]`**: Placeholder for your experimental results

### Explore vs Exploit

- **EXPLORE**: Testing new areas of the reaction space
- **EXPLOIT**: Refining near known good results

### Hypervolume Tracking

```python
hv = optimizer.calculate_current_hv()
print(f"Progress: {hv['hv_normalized']*100:.1f}%")
history = optimizer.calculate_hv_by_batch()
```

---

## ⚙️ Advanced Features

### Reaction Constraints

```python
constraints = {"alkali": ["DBU"], "solvent": ["DMSO"]}
optimizer.optimize(batch_size=5, constraints=constraints)
```

Or use `prohibited_reagent.json` for automatic loading.

### GPU Acceleration

SynBO auto-detects GPU. Force CPU: `optimizer.optimize(batch_size=5, device="cpu")`

### Excel Output with Molecular Structures

```python
optimizer.save_results(
    filetype="excel",
    figure_output=["cobalt_catalyst", "organo_catalyst"],
    figure_path="examples/figures",
)
```

---

## 🔧 CLI Quick Reference

```bash
synbo --version
synbo create-config -o my_config.json
synbo validate my_config.json
synbo init my_config.json -b 8 -m lhs -o results/
synbo optimize my_config.json results/batch-0.csv -b 5 -o results/
```

---

## 📦 Dependencies

Core: numpy, pandas, scikit-learn, torch, botorch | Chemistry: rdkit, epam.indigo | CLI: typer, rich | Viz: matplotlib, seaborn

See [`pyproject.toml`](pyproject.toml) for the complete list.

---

## 📚 Citation

```bibtex
@software{synbo2025,
  title={SynBO: Synthetic Bayesian Optimization for Chemical Reaction Optimization},
  author={Zhenzhi Tan},
  year={2025},
  url={https://github.com/yourusername/synbo}
}
```

---

## 📧 Contact

- **Author**: Zhenzhi Tan
- **Email**: zhenzhi-tan@outlook.com

---

**Happy Synthesizing! 🧪⚗️**
