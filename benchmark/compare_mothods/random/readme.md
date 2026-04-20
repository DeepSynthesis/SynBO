# Random Baseline Benchmark

This directory contains random baseline benchmark scripts for comparison with EDBO+ optimization. These scripts follow the same pattern as the EDBO+ benchmark scripts.

## Scripts

### `random_benchmark_B-H_HTE.py`
Random sampling baseline for B-H_HTE dataset (follows EDBO+ pattern).

**Usage:**
```bash
cd benchmark
python compare_mothods/random/random_benchmark_B-H_HTE.py
```

### `random_benchmark_suzuki_HTE.py`
Random sampling baseline for Suzuki HTE dataset (follows EDBO+ pattern).

**Usage:**
```bash
cd benchmark
python compare_mothods/random/random_benchmark_suzuki_HTE.py
```

## Features

- Follows the same structure as EDBO+ benchmark scripts
- Uses start points from `start_point.json` for fair comparison with EDBO+
- 10 rounds x 10 iterations x 5 batch size = 50 experiments per round
- Results saved to `results/` directory
- Calculates hypervolume metrics
- Output format compatible with EDBO+ benchmark results

## Output Files

- `results/mean_Random_for_{dataset}.csv` - Mean performance across rounds
- `results/timing_{dataset}.txt` - Timing information and performance summary
