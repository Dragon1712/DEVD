<div align="center">

# Divide & Learn: Multi-Objective Combinatorial Optimization at Scale (ICML 2026)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11346-b31b1b.svg)](https://arxiv.org/abs/2602.11346)

PyTorch official implementation, with benchmarks on MOCO (TSP, Knapsack, CVRP) and HW-SF datasets.

![Divide & Learn Demo](https://github.com/esha-singh/assets/raw/main/dnl-example.gif)

</div>

We reformulate multi-objective combinatorial optimization as online bandit learning over decomposed decision spaces, achieving substantially better efficiency than surrogate-modelling MOBO at scale.


## 🛠️ Environment

Requires Python 3.9+. Inside a virtual environment, install the required packages using pip:

```bash
pip install torch numpy pyyaml scipy matplotlib pymoo
```


## 🚀 Quickstart

Run a benchmark out-of-the-box using one of the provided YAML configs:

```bash
python benchmarking.py --config configs/ucb.yaml >> log.txt 
python benchmarking.py --config configs/ts.yaml
```

To run 200 instances of the UCB variant on the default problem set:

```bash
bash scripts/run_ucb_200.sh
```
### Logging

`benchmarking.py` prints progress to stdout — redirect or tee it to capture a run log:

```bash
# Append to a log file (no console output)
python benchmarking.py --config configs/ucb.yaml >> log.txt

# Keep console output and write to the log simultaneously
python benchmarking.py --config configs/ucb.yaml 2>&1 | tee log.txt
```

Results in `results/<method_name>_<timestamp>/` are written regardless of stdout redirection.

## Running Benchmarks

All benchmarks run through `benchmarking.py`. The CLI accepts either a YAML config (recommended) or inline arguments — **CLI flags override YAML values**.

### Inline overrides

Override any field from a config without editing the file:

```bash
# Different num_runs / cuda device
python benchmarking.py --config configs/ucb.yaml --num-runs 5 --cuda-device 1

# Different problem set (inline JSON)
python benchmarking.py --config configs/ucb.yaml \
  --problems '{"BiObjectiveTSP": {"large": {"n_cities": 100}}}'

# Tweak individual hyperparameters
python benchmarking.py --config configs/ucb.yaml \
  --param learning_rate=0.3 --param nb_rounds=10
```

### Without a config (everything inline)

```bash
python benchmarking.py \
  --algorithm ucb \
  --method-name MyUCBRun \
  --num-runs 3 \
  --problems '{"BiObjectiveTSP": {"medium": {"n_cities": 50}}}' \
  --params '{"learning_rate": 0.5, "decomposition_size": 25, "overlap": 18, "max_iterations": 150}'
```

### Config schema (`configs/*.yaml`)

```yaml
method_name: UCBHedgeHybridOCOAcc       # display name (also used for results filename)
algorithm: ucb                          # one of: ucb, ts
num_runs: 10                            # repetitions per (problem, size)
cuda_device: 0                          # ignored if CUDA unavailable

problems:                               # nested: type -> size_label -> problem_params
  BiObjectiveTSP:
    medium: {n_cities: 50}
  # MultiObjectiveKnapsack:
  #   large: {n_items: 200, n_objectives: 2, capacity: 25.0}

params:                                 # algorithm hyperparameters
  learning_rate: 0.5
  decomposition_size: 25
  overlap: 18
  ...
```

Problem parameter keys per problem type:

| Problem                  | Required keys                                    |
|--------------------------|--------------------------------------------------|
| `BiObjectiveTSP`         | `n_cities`                                       |
| `TriObjectiveTSP`        | `n_cities`                                       |
| `MultiObjectiveKnapsack` | `n_items`, `n_objectives`, `capacity`            |
| `BiObjectiveCVRP`        | `n_customers` (optional `n_vehicles`)            |


## Output

Each invocation creates one folder `results/<method_name>_<timestamp>/` containing:

- `summary.yaml` — per-(problem, size) hypervolume, runtime, non-dominated count, and run metadata.
- `<method>_<problem>_<iso-timestamp>.json` — detailed per-run artefacts written by `MOCOEvaluator` (one JSON per problem evaluated). Includes `individual_runs` (one entry per seed) and aggregate `statistics` (mean / median / std / cv).


## Multiple Seeds / Instances

`num_runs` controls how many problem instances are sampled — `MOCOEvaluator` uses `seed = run + 1` (so `num_runs=200` means seeds 1..200), regenerating the problem instance each time.

Customize via env vars or positional arguments:

```bash
# Override config / seeds / device
NUM_RUNS=50 CUDA_DEVICE=1 bash scripts/run_ucb_200.sh

# Run on a different problem/size (positional JSON arg)
bash scripts/run_ucb_200.sh '{"BiObjectiveTSP": {"large": {"n_cities": 100}}}'
```

The script is a thin wrapper over `python benchmarking.py --config configs/ucb.yaml --num-runs 200`. The TS variant runs the same way: `python benchmarking.py --config configs/ts.yaml --num-runs 200`.


## Adding a New Method

1. Add the wrapper class to `src/divide_n_learn/`.
2. Register it in `ALGORITHM_REGISTRY` in `benchmarking.py`.
3. Add a config in `configs/<name>.yaml`.


## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{singh2026divide,
  title     = {Divide and Learn: Multi-Objective Combinatorial Optimization at Scale},
  author    = {Esha Singh and Dongxia Wu and Chien-Yi Yang and Tajana Rosing and Rose Yu and Yian Ma},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=TK82ECnJzD}
}
```
