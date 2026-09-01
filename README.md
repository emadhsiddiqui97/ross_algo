# ROSS: Randomized Online Spot Scheduler Simulator (Track A)

Trace-driven backtest simulator and evaluation framework for the **ROSS scheduling algorithm** (Bhuyan, Kodialam, Bhatia, Lakshman — *arXiv:2601.14612*).

## Project Structure

```
ross_algo/
├── ross/
│   ├── __init__.py
│   ├── core.py                 # Action, Phase, SchedulerState, BaseScheduler protocol
│   ├── scheduler_ross.py       # ROSS algorithm (greedy & uniform warmup variants)
│   ├── scheduler_baselines.py  # UniformProgress and Greedy baselines
│   ├── traces.py               # Synthetic Markov generator & CSV trace loaders
│   ├── simulator.py            # run_policy, hindsight_optimal_cost, compare_policies
│   ├── validate.py             # Theoretical guarantee invariant checks (Theorem 2)
│   └── forecast.py             # (Part B) Prediction-augmented ROSS
├── data/
│   └── traces/                 # Real cloud traces (SpotLake, SkyPilot)
├── experiments/
│   ├── sweep_ld_ratio.py       # Sweep across L/D ratio
│   ├── sweep_cost_ratio.py     # Sweep across cost ratio K
│   └── monte_carlo_validate.py # Monte Carlo parameter grid invariant verification
├── tests/
│   ├── test_core.py            # Core interface and state transition tests
│   └── test_scheduler.py       # Checkpoint tests (all-False spot deadline guarantee, edge cases)
└── reports/                    # Generated plots, figures, and evaluation tables
```

## Quick Start

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Run Experiments
```bash
python experiments/monte_carlo_validate.py
python experiments/sweep_ld_ratio.py
python experiments/sweep_cost_ratio.py
```
