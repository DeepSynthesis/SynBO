# Random Baseline Benchmark

This directory contains random baseline benchmark scripts for comparison with SynBO optimization.

## Scripts

### `random_baseline_B-H_HTE.py`
Random sampling baseline for B-H_HTE dataset.

**Usage:**
```bash
cd benchmark
python compare_mothods/random/random_baseline_B-H_HTE.py
```

### `random_baseline_suzuki_HTE.py`
Random sampling baseline for Suzuki HTE dataset.

**Usage:**
```bash
cd benchmark
python compare_mothods/random/random_baseline_suzuki_HTE.py
```

## Features

- Uses the same start points as SynBO for fair comparison
- 10 rounds × 10 iterations × 5 batch size = 50 experiments per round
- Results saved to `results/random_baseline/` directory
- Calculates metrics: average optimal targets, AUC, hypervolume

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| NUM_ROUNDS | 10 | Number of rounds to run |
| NUM_ITERATIONS | 10 | Iterations per round |
| BATCH_SIZE | 5 | Batch size for each iteration |
| RECALC | True | Force recalculation (skip cache) |
