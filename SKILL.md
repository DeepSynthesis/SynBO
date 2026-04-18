---
name: synbo
description: Bayesian optimization for chemical reactions using the synbo package. This skill provides Python scripts to set up reaction spaces, build descriptors, run optimization, download recommended conditions, and upload results.
---

# synbo

Bayesian optimization for chemical reactions using the `synbo` package. This skill provides Python scripts to set up reaction spaces, build descriptors, run optimization, download recommended conditions, and upload results.

---

## CRITICAL: BO Optimization Prerequisites

**Before executing ANY Bayesian Optimization (synbo) tasks, you MUST sequentially verify the following 5 prerequisites. Do NOT proceed with the optimization process until ALL criteria are met:**

**1. Working Directory (`project_wd`) & Project Name (`project_name`)**
* **Initial Check:** Read `config.json` located in the skill's directory. If both `project_wd` and `project_name` are found, display the project name to the user (e.g., "Found existing project: [Project Name]") and use them.
* **If NOT found:** Stop and prompt the user to input a **Working Directory** and a **Project Name**.
* **Validation & Saving (CRITICAL):** Upon receiving the user's input:
    1. **Verify Path:** Check if the provided working directory actually exists on the local file system. If it does NOT exist, inform the user "The path is invalid/does not exist" and prompt them to re-enter it.
    2. **Sanitize & Save:** If the path exists, sanitize the project name (replace spaces and special characters with underscores). Then, immediately write/update the `config.json` file with this format: `{"project_wd": "xxx", "project_name": "xxx"}`.
    3. Use this path and sanitized project name to define the `save_dir` for all subsequent outputs.

**2. Reaction Space**
* **Check:** Verify if standard reaction space data exists specifically within the `project_wd/rxn_space` directory.
* **If NOT found:** Stop and prompt the user: "Reaction space data is missing in the `project_wd/rxn_space` directory. Please provide the standard reaction space data." Do not proceed until provided.

**3. Condition Descriptors**
* **Check:** Verify if the corresponding Condition Descriptors exist specifically within the `project_wd/mol_desc` directory.
* **If NOT found:** Stop and prompt the user: "Condition Descriptors are missing in the `project_wd/mol_desc` directory. Please provide the standard Condition Descriptors, OR let me know if you would like me to automatically generate them for you."

**4. Optimization Metrics**
* **Check:** Verify if the optimization settings file (e.g., `optimization_settings.json`) exists directly within the `project_wd` directory.
* **If NOT found:** Stop and prompt the user: "Optimization metrics are not defined. Please specify the target metrics you want to optimize (e.g., yield, ee), along with their optimization direction (max/min), expected numerical ranges, and relative weights (default 1.0)."
* **Validation & Saving (CRITICAL):** Upon receiving the user's optimization goals, format the data and immediately save it as `optimization_settings.json` in the `project_wd`. The JSON file MUST strictly adhere to the following structure:
```json
{
    "optimization_settings": {
        "opt_metrics": ["yield", "ee"],
        "opt_direct_info": [
            {
                "opt_direct": "max",
                "opt_range": [0, 100],
                "metric_weight": 1.0
            },
            {
                "opt_direct": "max",
                "opt_range": [0, 100],
                "metric_weight": 1.0
            }
        ]
    }
}
```

**Execution Block:**
You are strictly forbidden from executing any initialization (`initialize`), optimization (`optimize`), or other synbo tasks until **Steps 1 through 4** are fully verified and resolved.

---

## Reaction Space

When the user is required to provide the reaction space data, they may submit it via one of two methods:
1. **Direct Input:** Providing the SMILES strings for the corresponding molecules directly in the chat.
2. **File Upload:** Providing tabular files containing the SMILES strings.

**Naming Conventions:** 
It is highly recommended that the user assigns a specific name to each molecule. If no names are provided, you must automatically assign names using a sequential format based on the reagent type: `{reagent_type}-1`, `{reagent_type}-2`, etc.

**Data Storage Rules:** 
Regardless of the user's submission method, all reaction space data must be formatted and saved strictly into the `project_wd/rxn_space` directory.
* Each reagent type must be saved as an individual file named `{reagent_type}.csv`.
* The CSV files must contain exactly two headers: `SMILES` (for the SMILES strings) and `name` (for the molecule names).

---

## Condition Descriptors

When the user is required to provide Condition Descriptors, they may fulfill this requirement via one of two methods:
1. **Automated Generation:** Instructing you to generate the RDKit descriptors automatically.
2. **File Upload:** Providing pre-calculated tabular files containing the necessary descriptors.

**Automated RDKit Descriptor Generation Protocol:**
If the user requests automated generation, you must strictly adhere to the following workflow:
1. **Prerequisite Check:** You must verify that the Reaction Space data already exists. Automated generation is strictly prohibited if the Reaction Space data is missing.
2. **User Confirmation:** Before executing any calculations, you MUST explicitly ask the user to confirm exactly which reagent categories require descriptor calculation.
3. **Execution:** Once the categories are confirmed, invoke `scripts/get_desc.py` to compute the descriptors for each specified reagent category sequentially.

**Handling Invalid SMILES (0.0 Placeholder Strategy):**
When a reaction space CSV contains entries without valid SMILES strings (e.g., `no_alkali`, `none`, `blank`, or any non-parseable text), the script automatically:
1. Detects invalid SMILES using RDKit's `MolFromSmiles` parser.
2. Skips them during descriptor calculation.
3. Adds placeholder rows to the output descriptor file with **all descriptor values set to 0.0**.
4. Reports which entries were treated as placeholders.

This strategy ensures that categorical "none" options (e.g., "no catalyst", "no additive") are represented as zero vectors in the descriptor space, allowing the optimizer to learn their effect independently.

**Example output:**
```
⚠️  Found 1 entries with invalid SMILES (will use 0.0 placeholders):
    - no_alkali
Generating RDKit descriptors for 9 SMILES strings...
✅ Saved to: .../alkali_RDKit.csv
✅ Added 1 placeholder entries (0.0-filled) to: .../alkali_RDKit.csv
```

**Handling Continuous Variables:**
If the user's submitted conditions include continuous variables (e.g., temperature, time, voltage), do NOT attempt to calculate molecular descriptors for them. Instead, these continuous variables must be formatted and saved in a CSV file using the following strict layout:
```csv
Name,value
[variable_name],[variable_value]
```

---

## Initialize

The `initialize.py` script is used for initial sampling without previous reaction data. This is the starting point for Bayesian optimization when no experimental results are available yet.

**Script Location:** `scripts/initialize.py`

**Prerequisites:**
- Working directory and project name configured in `config.json`
- Reaction space data in `project_wd/rxn_space` directory
- Condition descriptors in `project_wd/mol_desc` directory
- Optimization settings file (`optimization_settings.json`) in `project_wd` directory

**Usage:**
```bash
python scripts/initialize.py --desc-dir <path_to_descriptors> --output <output_directory>
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--desc-dir` | Required | Directory containing descriptor files |
| `--output` | `output/initialize` | Output directory for results |
| `--reagent-types` | `base ligand solvent concentration temperature` | List of reagent types |
| `--name-suffix` | `_dft _dft _dft None None` | Name suffixes for descriptor files |
| `--batch-size` | `5` | Number of initial samples to generate |
| `--desc-normalize` | `minmax` | Descriptor normalization method (`minmax`, `zscore`, `l2`) |
| `--sampling-method` | `kmeans` | Sampling strategy (`sobol`, `random`, `lhs`, `kmeans`) |
| `--refine-desc` | `filter_0.8` | Descriptor refinement method (`auto_select`, `filter_only`, `pass`, `filter_0.8`) |
| `--random-seed` | `42` | Random seed for reproducibility |
| `--quiet` | - | Suppress verbose output |

**Examples:**
```bash
# Initialize with default settings
python scripts/initialize.py --desc-dir dataset/descriptors --output output/initialize

# Initialize with custom batch size and sampling method
python scripts/initialize.py --desc-dir dataset/descriptors --output output/initialize \
    --batch-size 10 --sampling-method sobol

# Initialize with custom configuration
python scripts/initialize.py --desc-dir dataset/descriptors --output output/initialize \
    --reagent-types base ligand solvent --batch-size 5
```

**Workflow Steps:**
1. Load optimization settings from `config.json` and `optimization_settings.json`
2. Load descriptors from specified directory
3. Create `ReactionOptimizer` instance with `opt_type="init"`
4. Load reaction space
5. Load descriptors
6. Run initialization with sampling (initial design generation)
7. Save recommended conditions to output directory

**Output:**
- Excel file (`recommended_conditions.xlsx`) containing recommended experimental conditions
- Results saved to the specified output directory

**Next Steps:**
1. Download the Excel file with recommended experimental conditions
2. Run the experiments with the recommended conditions
3. Collect the experimental results
4. Use `optimize.py` to continue optimization with the new data

---

## Optimize

The `optimize.py` script runs Bayesian optimization with previous reaction data to recommend new experimental conditions. This is used after initial sampling or previous optimization rounds.

**Script Location:** `scripts/optimize.py`

**Prerequisites:**
- Working directory and project name configured in `config.json`
- Reaction space data in `project_wd/rxn_space` directory
- Condition descriptors in `project_wd/mol_desc` directory
- Optimization settings file (`optimization_settings.json`) in `project_wd` directory
- Previous reaction data in CSV format (input file)

**Usage:**
```bash
python scripts/optimize.py --desc-dir <path_to_descriptors> --input <previous_data.csv> --output <output_directory>
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--desc-dir` | Required | Directory containing descriptor files |
| `--input` | Required | Path to CSV file with previous reaction data |
| `--output` | `output/optimize` | Output directory for results |
| `--reagent-types` | `base ligand solvent concentration temperature` | List of reagent types |
| `--name-suffix` | `_dft _dft _dft None None` | Name suffixes for descriptor files |
| `--batch-size` | `1` | Number of new conditions to recommend |
| `--desc-normalize` | `zscore` | Descriptor normalization method (`minmax`, `zscore`, `l2`) |
| `--refine-desc` | `pass` | Descriptor refinement method (`auto_select`, `filter_only`, `pass`) |
| `--optimize-method` | `default_BO` | Optimization algorithm to use |
| `--surrogate-model` | `GP` | Surrogate model type |
| `--random-seed` | `42` | Random seed for reproducibility |
| `--quiet` | - | Suppress verbose output |

**Examples:**
```bash
# Optimize with default settings
python scripts/optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv --output output/optimize

# Optimize with custom batch size
python scripts/optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv \
    --output output/optimize --batch-size 5

# Optimize with custom configuration
python scripts/optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv \
    --output output/optimize --reagent-types base ligand solvent \
    --surrogate-model RF 
```

**Workflow Steps:**
1. Load optimization settings from `config.json` and `optimization_settings.json`
2. Load descriptors from specified directory
3. Create `ReactionOptimizer` instance with `opt_type="auto"`
4. Load reaction space
5. Load descriptors
6. Load previous reaction data from input CSV file
7. Run Bayesian optimization to recommend new conditions
8. Save recommended conditions to output directory

**Input Data Format:**
The input CSV file should contain previous reaction data with columns corresponding to:
- Reagent types (matching those specified in `--reagent-types`)
- Optimization metrics (matching those defined in `optimization_settings.json`)
- Optional: `batch` column indicating batch numbers

**Output:**
- Excel file (`recommended_conditions.xlsx`) containing recommended experimental conditions
- Summary including number of exploit vs explore recommendations
- Results saved to the specified output directory

**Summary Information:**
After optimization, the script displays:
- Number of recommended new conditions
- Count of exploit vs explore recommendations

**Next Steps:**
1. Download the Excel file with recommended experimental conditions
2. Run the experiments with the recommended conditions
3. Collect the experimental results
4. Update your reaction data file with new results
5. Run `optimize.py` again to continue optimization
