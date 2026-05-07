# EDBO+ Benchmark Class Overview

## 1. Benchmark Class Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Benchmark Workflow                         │
└─────────────────────────────────────────────────────────────────────┘

1. Initialization Phase
   ├─ Load complete dataset (ground truth)
   ├─ Compute true Pareto front and hypervolume
   ├─ Create EDBO+ optimizer instance
   └─ Generate initial reaction scope file (without target values)

2. Optimization Loop (iterative)
   ┌─────────────────────────────────────────┐
   │  Each iteration performs:               │
   │                                   │
   │ (a) Initial Sampling (first iteration)   │
   │    - Use random/lhs/cvtsampling          │
   │    - Select batch_size initial samples   │
   │                                   │
   │ (b) Train Surrogate Model               │
   │    - Use Gaussian Process                │
   │    - Train independent model per target  │
   │                                   │
   │ (c) Optimize Acquisition Function        │
   │    - EHVI: Expected Hypervolume          │
   │    - MOUCB: Multi-Objective UCB          │
   │    - MOGreedy: Greedy strategy           │
   │    - Select batch_size new experiments   │
   │                                   │
   │ (d) Simulate Experiment (from ground truth) │
   │    - Assign true target values to samples│
   │    - Update training dataset             │
   │                                   │
   │ (e) Evaluate Performance                 │
   │    - Compute current Pareto front        │
   │    - Compute hypervolume completion      │
   │    - Record best target values           │
   │    - Compute prediction errors           │
   │                                   │
   │ (f) Save Results                         │
   │    - Record all metrics for this step    │
   │                                   │
   └─────────────────────────────────────────┘

3. Repeat step 2 until reaching the specified number of iterations (steps)
```

## 2. Input Parameters

### Constructor Parameters (at initialization)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| **df_ground** | `pd.DataFrame` | Complete dataset with all feature and target columns | 1728-row x 8-column DataFrame |
| **index_column** | `str` | Index column name for tracking experiments | `'new_index'` |
| **objective_names** | `list[str]` | List of target column names to optimize | `['yield', 'cost']` |
| **objective_modes** | `list[str]` | Optimization mode per objective ('max' or 'min') | `['max', 'min']` |
| **objective_thresholds** | `list[Optional[float]]` | Worst-case threshold per objective (for hypervolume) | `[None, None]` |
| **features_regression** | `list[str]` | Feature column names for regression model | `['base', 'ligand', 'solvent', ...]` |
| **filename** | `str` | Temporary reaction scope filename | `'benchmark.csv'` |
| **filename_results** | `str` | Results output filename | `'results_benchmark.csv'` |
| **acquisition_function** | `str` | Acquisition function type | `'EHVI'`, `'MOUCB'`, `'MOGreedy'` |

### Run Method Parameters (bench.run())

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| **steps** | `int` | Number of iterations (total batches to run) | `12` |
| **batch** | `int` | Number of experiments suggested per batch | `5` |
| **seed** | `int` | Random seed for reproducibility | `1` |
| **init_method** | `str` | Initial sampling method | `'cvtsampling'`, `'lhs'`, `'seed'` |
| **plot_ground** | `bool` | Whether to plot ground truth | `False` |
| **plot_predictions** | `bool` | Whether to plot predictions | `False` |
| **plot_train** | `bool` | Whether to plot training process | `False` |

## 3. Output Details

### 3.1 Console Output (real-time)

```
Initialization Phase:
├─ High trade-off ground truth: [[...]]
├─ Ground truth hypervolume: 0.991471
└─ Number of Pareto optimal points: 9

Per Iteration:
├─ Best yield found: 76.02
├─ Best cost found: 0.028582675
├─ Total number of experiments: 10
├─ Hypervolume train (%): 73.85
├─ Maximin distance to Pareto: 23.98
└─ Maximin distance to Tradeoff: 10.44

Prediction Errors:
├─ MAE_yield: 21.88
├─ RMSE_yield: 25.23
├─ R2_yield: -0.05
└─ (similar metrics for each target)
```

### 3.2 CSV File Output

#### (1) Reaction Scope File (`{filename}.csv`)
```csv
new_index,base,ligand,solvent,concentration,temperature,yield,cost,priority
966,KOPiv,P(fur)3,DMAc,0.1,105,73.59,0.0377,1.0
```
- Contains all reaction conditions
- `priority=1.0`: recommended experiments
- `priority=-1.0`: completed experiments
- `priority=0.0`: unselected experiments

#### (2) Results File (`results_{filename_results}.csv`)
```csv
step,n_experiments,hypervolume_ground,hypervolume_sampled,
hypervolume completed (%),yield_best,cost_best,
dmaximin_pareto,dmaximin_tradeoff,...
0,5,0.991471,0.708429,71.49,73.59,0.02858,34.80,12.87,...
1,10,0.991471,0.732014,73.85,76.02,0.02858,23.98,10.44,...
```
**Key columns:**
- `step`: current iteration number (0, 1, 2, ...)
- `n_experiments`: cumulative experiment count
- `hypervolume_ground`: true hypervolume value (fixed)
- `hypervolume_sampled`: hypervolume of current sample set
- `hypervolume completed (%)`: completion percentage = `sampled/ground * 100`
- `{obj}_best`: current best value for each objective
- `dmaximin_pareto`: Hausdorff distance to the true Pareto front
- `dmaximin_tradeoff`: distance to the true trade-off point
- `MAE_{obj}`, `RMSE_{obj}`, `R2_{obj}`: prediction error metrics

#### (3) Prediction File (`pred_{filename}.csv`)
```csv
new_index,base,ligand,...,yield,priority,
yield_predicted_mean,yield_predicted_std_dev,yield_expected_improvement,
cost_predicted_mean,cost_predicted_std_dev,cost_expected_improvement
0,KOPiv,P(fur)3,...,73.59,1.0,
75.2,5.3,0.8,0.025,0.01,0.15
```
- `yield_predicted_mean`: GP model predicted mean
- `yield_predicted_std_dev`: prediction standard deviation (uncertainty)
- `yield_expected_improvement`: expected improvement value

## 4. Key Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Data Flow Diagram                    │
└──────────────────────────────────────────────────────────────┘

Ground Truth (complete dataset)
    │
    ├── Compute Pareto front and Hypervolume
    │
    └── Create initial reaction scope (without target values)
              │
              ↓
         Optimization Loop
              │
    ┌─────────┴─────────┬─────────┐
    │                 │         │
  Training Data  Candidate Set  New Batch
    │                 │         │
    ↓                 ↓         ↓
  GP Model      Optimize ACQ   Selected Samples
    │                           │
    └───────────────┬───────────┘
                    │
                    ↓
               Simulate Experiment (get true values from ground truth)
                    │
                    ↓
               Update Training Data
                    │
                    ↓
               Evaluate and Record Results
                    │
                    └─────→ Next Iteration
```

## 5. Usage Example (based on run.py)

```python
from edbo.plus.benchmark.multiobjective_benchmark import Benchmark
import pandas as pd

# 1. Load data
df_exp = pd.read_csv('../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv')

# 2. Define features and objectives
features = ['base', 'ligand', 'solvent', 'concentration', 'temperature']
objectives = ['yield', 'cost']
modes = ['max', 'min']

# 3. Initialize Benchmark
bench = Benchmark(
    df_ground=df_exp,
    index_column='new_index',
    objective_names=objectives,
    objective_modes=modes,
    objective_thresholds=[None, None],
    features_regression=features,
    filename='benchmark.csv',
    filename_results='results.csv',
    acquisition_function='EHVI'
)

# 4. Run optimization
bench.run(
    steps=12,
    batch=5,
    seed=1,
    init_method='cvtsampling',
    plot_ground=False,
    plot_predictions=False,
    plot_train=False
)

# 5. Read results
results = pd.read_csv('results/results.csv')
print(f"Final hypervolume completion: {results.iloc[-1]['hypervolume completed (%)']:.2f}%")
```

## 6. Core Algorithms

### Hypervolume Calculation
- **Purpose**: Measure overall quality of Pareto front
- **Reference point**: Worst value per target (or user-specified threshold)
- **Monotonicity**: Assumes maximization; all objectives normalized to [0,1]

### EHVI (Expected Hypervolume Improvement)
- **Principle**: Predict expected hypervolume improvement for each candidate
- **Advantage**: Automatically balances exploration and exploitation
- **Use case**: Multi-objective optimization without experimental noise

### CVT Sampling (Centroidal Voronoi Tessellation)
- **Principle**: Partition space into Voronoi cells; uniform sampling
- **Advantage**: More uniform spatial coverage than random sampling
- **Use case**: Initial sample selection