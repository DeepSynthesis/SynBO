# synbo

Bayesian optimization for chemical reactions using the `synbo` package. This skill provides Python scripts to set up reaction spaces, build descriptors, run optimization, download recommended conditions, and upload results.

---

## Installation

### Step 1: Check and Install Miniconda

First, check if conda is already installed:

```bash
# Check if conda is available
which conda
# or
conda --version
```

**If conda is NOT installed**, install Miniconda using Tsinghua mirror (for China):

```bash
# Download Miniconda installer
cd /tmp
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh

# Run installer (silent mode, for macOS arm64)
bash miniconda.sh -b -p $HOME/miniconda3

# Initialize conda for your shell
$HOME/miniconda3/bin/conda init zsh

# Restart shell or run:
source ~/.zshrc

# Verify installation
conda --version
```

**For Linux (x86_64):**
```bash
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

**For Linux (arm64):**
```bash
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Step 2: Create synbo_env Environment

```bash
# Create new environment with Python 3.12
conda create -n synbo_env python=3.13 -y

# Activate the environment
conda activate synbo_env
```

### Step 3: Install Required Packages

**Important:** Install `qspoc` first, then `synbo` (dependency order matters):

```bash
# Make sure you're in synbo_env
conda activate synbo_env

# Install qspoc first (required dependency)
pip install qspoc

# Then install synbo
pip install synbo
```

### Step 4: Verify Installation

```bash
conda activate synbo_env
python -c "from synbo import ReactionOptimizer; print('synbo installed successfully!')"
```

---

## Environment Requirements

- **Conda environment:** `synbo_env` (must be activated before running)
- **Python:** 3.12+
- **Package location:** `/Users/neonart/synbo` (if using local development version)
- **CLI script:** `~/.openclaw/workspace/skills/synbo/scripts/synbo_cli.py`

### Running Code

**Always activate the environment before running:**

```bash
# Method 1: Activate then run
conda activate synbo_env
python your_script.py

# Method 2: Use conda run (no activation needed)
conda run -n synbo_env python your_script.py
```

---

## 1. Setting Up the Reaction Space

The reaction space defines all possible conditions for your optimization. Create a JSON configuration file:

```json
{
  "opt_metrics": ["yield", "ee"],
  "opt_metric_settings": [
    {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
    {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 2.0}
  ],
  "condition_dict": {
    "catalyst": ["Pd(OAc)2", "Pd(PPh3)4", "Pd(dppf)Cl2"],
    "solvent": ["THF", "Dioxane", "Toluene", "DMF"],
    "base": ["Cs2CO3", "K2CO3", "Na2CO3", "Et3N"],
    "temperature": ["25", "50", "80", "100"]
  },
  "desc_dict": {}
}
```

**Important:** All values in `condition_dict` must be strings (including temperature: use `"25"` not `25`).

### Python Script: Create Reaction Space

```python
import json
import sys
sys.path.insert(0, '/Users/neonart/synbo/src')

from synbo import ReactionOptimizer

# Define reaction space
condition_dict = {
    'catalyst': ['Pd(OAc)2', 'Pd(PPh3)4'],
    'solvent': ['THF', 'Dioxane'],
    'base': ['Cs2CO3', 'K2CO3'],
    'temperature': ['25', '50', '80']
}

# Create optimizer and load space
optimizer = ReactionOptimizer(
    opt_metrics=['yield', 'ee'],
    opt_type='auto',
    random_seed=42
)
optimizer.load_rxn_space(condition_dict)

print(f"Reaction space loaded: {optimizer.rxn_space_size} conditions")
```

---

## 2. Building Descriptors

Descriptors encode chemical conditions for the model. You can use built-in descriptors or provide custom ones.

### Option A: Automatic OneHot Encoding (Default)

If `desc_dict` is empty or not provided, synbo automatically uses OneHot encoding:

```python
# No descriptor setup needed - OneHot encoding is automatic
optimizer.initialize(batch_size=5)
```

### Option B: Custom Descriptors

Provide molecular descriptors for catalysts, ligands, or substrates:

```json
{
  "desc_dict": {
    "catalyst": {
      "type": "morgan",
      "radius": 2,
      "n_bits": 2048
    },
    "solvent": {
      "type": "rdkit",
      "descriptors": ["MolWt", "LogP", "TPSA"]
    }
  }
}
```

### Python Script: Build Custom Descriptors

```python
import sys
sys.path.insert(0, '/Users/neonart/synbo/src')

from synbo import ReactionOptimizer

optimizer = ReactionOptimizer(opt_metrics=['yield'], opt_type='auto')

# Load reaction space
condition_dict = {
    'catalyst': ['Pd(OAc)2', 'Pd(PPh3)4'],
    'solvent': ['THF', 'Dioxane']
}
optimizer.load_rxn_space(condition_dict)

# Build descriptors
desc_dict = {
    'catalyst': {'type': 'morgan', 'radius': 2, 'n_bits': 2048}
}
optimizer.build_descriptors(desc_dict)

print("Descriptors built successfully")
```

---

## 3. Running the Optimization

### Step 1: Initialization (First Batch)

Run initial experiments using Sobol sampling for uniform coverage:

```python
import sys
sys.path.insert(0, '/Users/neonart/synbo/src')

from synbo import ReactionOptimizer
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Create optimizer
optimizer = ReactionOptimizer(
    opt_metrics=config['opt_metrics'],
    opt_type='auto',
    random_seed=42
)

# Load reaction space
optimizer.load_rxn_space(config['condition_dict'])

# Build descriptors if provided
if config.get('desc_dict'):
    optimizer.build_descriptors(config['desc_dict'])

# Initialize with Sobol sampling
optimizer.initialize(batch_size=5, method='sobol')

# Save recommendations
optimizer.save_results(filetype='csv', output_dir='./results')
print("Initial batch saved to ./results/")
```

**Command-line equivalent:**
```bash
cd ~/.openclaw/workspace/skills/synbo/scripts

# Method 1: Activate environment first
conda activate synbo_env
python synbo_cli.py init \
  --config ../examples/config.json \
  --batch-size 5 \
  --method sobol \
  --output ../results

# Method 2: Use conda run (recommended for scripts)
conda run -n synbo_env python synbo_cli.py init \
  --config ../examples/config.json \
  --batch-size 5 \
  --method sobol \
  --output ../results
```

### Step 2: Optimization (Subsequent Batches)

After running experiments and collecting results, optimize to recommend the next batch:

```python
import sys
sys.path.insert(0, '/Users/neonart/synbo/src')

from synbo import ReactionOptimizer
import pandas as pd
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Create optimizer
optimizer = ReactionOptimizer(
    opt_metrics=config['opt_metrics'],
    opt_type='auto',
    random_seed=42
)

# Load reaction space and descriptors
optimizer.load_rxn_space(config['condition_dict'])
if config.get('desc_dict'):
    optimizer.build_descriptors(config['desc_dict'])

# Load historical data
# CSV must contain: all condition columns + target columns (yield, ee) + batch column
historical_data = pd.read_csv('experimental_results.csv')
optimizer.load_prev_rxn(historical_data)

# Run optimization
optimizer.optimize(batch_size=5)

# Save recommendations
optimizer.save_results(filetype='csv', output_dir='./results')
print("Optimized batch saved to ./results/")
```

**Command-line equivalent:**
```bash
cd ~/.openclaw/workspace/skills/synbo/scripts

# Method 1: Activate environment first
conda activate synbo_env
python synbo_cli.py optimize \
  --config ../examples/config.json \
  --data ../examples/experimental_results.csv \
  --batch-size 5 \
  --output ../results

# Method 2: Use conda run (recommended for scripts)
conda run -n synbo_env python synbo_cli.py optimize \
  --config ../examples/config.json \
  --data ../examples/experimental_results.csv \
  --batch-size 5 \
  --output ../results
```

---

## 4. Downloading Recommended Conditions

After running initialization or optimization, retrieve the recommended conditions:

```python
import pandas as pd

# Load the latest batch recommendations
recommendations = pd.read_csv('./results/batch-1_20260326.csv')

# Display recommendations
print("Recommended Conditions:")
print(recommendations[['index', 'type', 'catalyst', 'solvent', 'base', 'temperature', 'pred yield']])

# Export to Excel for lab notebook
recommendations.to_excel('./results/recommendations_batch1.xlsx', index=False)

# Or get as JSON for API integration
recommendations_json = recommendations.to_dict('records')
print(f"\n{len(recommendations_json)} conditions recommended")
```

### Output File Format

The output CSV contains:
- `batch`: Batch number (0 = initial, 1+ = optimized)
- `index`: Condition index within batch
- `type`: Strategy type (explore/exploit)
- Condition columns (catalyst, solvent, base, temperature, etc.)
- `pred yield`: Predicted value ± standard deviation
- `yield`: Empty column for you to fill in experimental results

---

## 5. Uploading Optimization Results

After running experiments, prepare and upload results for the next optimization round:

### Step 1: Prepare Results CSV

Fill in the experimental results in the output CSV:

```python
import pandas as pd

# Load recommendations
df = pd.read_csv('./results/batch-1_20260326.csv')

# Fill in your experimental results
# IMPORTANT: All condition values must be strings!
df['yield'] = [85.2, 72.5, 91.0, 68.3, 79.8]  # Your measured yields
df['ee'] = [92.0, 88.5, 95.2, 85.0, 90.1]     # Your measured ee (if applicable)

# Save for next optimization round
df.to_csv('./results/experimental_results.csv', index=False)
print("Results saved for next optimization round")
```

### Step 2: Upload Results for Next Round

```python
import sys
sys.path.insert(0, '/Users/neonart/synbo/src')

from synbo import ReactionOptimizer
import pandas as pd
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Create optimizer
optimizer = ReactionOptimizer(
    opt_metrics=config['opt_metrics'],
    opt_type='auto',
    random_seed=42
)

# Load reaction space
optimizer.load_rxn_space(config['condition_dict'])

# Load your experimental results
results = pd.read_csv('./results/experimental_results.csv')
optimizer.load_prev_rxn(results)

# Run next optimization
optimizer.optimize(batch_size=5)

# Save new recommendations
optimizer.save_results(filetype='csv', output_dir='./results')
print("New recommendations generated based on uploaded results")
```

### Results CSV Format Requirements

Your results CSV **must** include:
1. All condition columns from `condition_dict` (exact names, string values)
2. Target columns (yield, ee, etc.)
3. `batch` column (starting from 0, incrementing each round)

Example:
```csv
batch,index,type,catalyst,solvent,base,temperature,yield,ee
0,1,explore,Pd(OAc)2,THF,Cs2CO3,25,85.2,92.0
0,2,exploit,Pd(PPh3)4,Dioxane,K2CO3,50,72.5,88.5
```

---

## Complete Workflow Example

```python
"""
Complete Bayesian Optimization Workflow for Chemical Reactions
"""
import sys
sys.path.insert(0, '/Users/neonart/synbo/src')

from synbo import ReactionOptimizer
import pandas as pd
import json

# ============ CONFIGURATION ============
config = {
    "opt_metrics": ["yield"],
    "opt_metric_settings": [
        {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0}
    ],
    "condition_dict": {
        "catalyst": ["Pd(OAc)2", "Pd(PPh3)4", "Pd(dppf)Cl2"],
        "solvent": ["THF", "Dioxane", "Toluene"],
        "base": ["Cs2CO3", "K2CO3", "Et3N"],
        "temperature": ["25", "50", "80"]
    },
    "desc_dict": {}
}

# ============ ROUND 1: INITIALIZATION ============
print("=== Round 1: Initialization ===")
optimizer = ReactionOptimizer(
    opt_metrics=config['opt_metrics'],
    opt_type='auto',
    random_seed=42
)
optimizer.load_rxn_space(config['condition_dict'])
optimizer.initialize(batch_size=5, method='sobol')
optimizer.save_results(filetype='csv', output_dir='./results')
print("Initial conditions saved. Run experiments and fill in results.\n")

# ============ ROUND 2: OPTIMIZATION (after experiments) ============
# Simulate experimental results
results_df = pd.read_csv('./results/batch-0_*.csv')
results_df['yield'] = [75.2, 82.5, 68.0, 91.3, 77.8]  # Your actual results
results_df.to_csv('./results/results_round1.csv', index=False)

print("=== Round 2: Optimization ===")
optimizer2 = ReactionOptimizer(
    opt_metrics=config['opt_metrics'],
    opt_type='auto',
    random_seed=42
)
optimizer2.load_rxn_space(config['condition_dict'])
optimizer2.load_prev_rxn(pd.read_csv('./results/results_round1.csv'))
optimizer2.optimize(batch_size=5)
optimizer2.save_results(filetype='csv', output_dir='./results')
print("Optimized conditions saved. Continue experimentation loop.\n")
```

---

## Important Notes

1. **String Format**: All condition values must be strings (e.g., `"25"` not `25` for temperature)
2. **Column Matching**: Historical data columns must exactly match `condition_dict` keys
3. **Batch Numbering**: Start batch at 0, increment by 1 each round
4. **Data Completeness**: Include all condition columns + target columns + batch column
5. **Descriptor Consistency**: Use the same `desc_dict` across all rounds

---

## Resources

- **Package location:** `/Users/neonart/synbo`
- **Documentation:** `/Users/neonart/synbo/docs`
- **Examples:** `/Users/neonart/synbo/examples`
- **CLI script:** `~/.openclaw/workspace/skills/synbo/scripts/synbo_cli.py`
- **Conda environment:** `synbo_env` (always activate before running)

## Quick Reference: Environment Commands

```bash
# Check if conda is installed
which conda

# Activate synbo environment
conda activate synbo_env

# Run Python script in synbo environment
conda run -n synbo_env python your_script.py

# Verify synbo installation
conda run -n synbo_env python -c "from synbo import ReactionOptimizer; print('OK')"

# List all conda environments
conda env list

# Remove synbo_env (if needed)
conda env remove -n synbo_env
```
