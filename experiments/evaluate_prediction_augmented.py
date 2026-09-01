"""Sweep lambda_ for prediction-augmented ROSS on real traces (time-split)."""

import os

import numpy as np
import pandas as pd

from ross.forecast import PredictionAugmentedROSS
from ross.simulator import run_policy
from ross.traces import Trace, load_all_traces


def main(
    data_dir: str = "data",
    output_dir: str = "reports",
    L: float = 100.0,
    D: float = 125.0,
    K: float = 5.0,
    lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
    num_seeds: int = 5,
    max_windows_per_trace: int = 5,
    cutoff_frac: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    traces = load_all_traces(data_dir=data_dir)
    window_size = int(D)
    ratios = {lam: [] for lam in lambdas}

    print("Prediction-augmented ROSS lambda sweep (time-based split)")
    print(f"L={L}, D={D}, K={K}, traces={len(traces)}")
    print("-" * 64)

    for tr in traces:
        cutoff = int(len(tr) * cutoff_frac)
        if cutoff < 1 or (len(tr) - cutoff) < window_size:
            continue
        prefix = tr.availability[:cutoff]
        suffix = Trace(availability=tr.availability[cutoff:], name=f"{tr.name}_eval")
        stride = max(50, window_size // 2)
        windows = suffix.sliding_windows(
            window_size=window_size,
            stride=stride,
            max_windows=max_windows_per_trace,
        )
        for win in windows:
            for seed in range(num_seeds):
                for lam in lambdas:
                    sched = PredictionAugmentedROSS(
                        L=L,
                        D=D,
                        K=K,
                        lambda_=lam,
                        warmup_mode="greedy",
                        seed=seed,
                        history=prefix,
                    )
                    res = run_policy(sched, win, dt=1.0, switch_penalty_pct=0.0)
                    cr = res.total_cost / res.opt_cost if res.opt_cost > 0 else 1.0
                    ratios[lam].append(cr)

    rows = []
    print(f"{'lambda':>8}  {'n':>6}  {'mean CR':>10}  {'var CR':>10}  {'std CR':>10}")
    for lam in lambdas:
        vals = np.asarray(ratios[lam], dtype=float)
        mean = float(np.mean(vals)) if vals.size else float("nan")
        var = float(np.var(vals)) if vals.size else float("nan")
        std = float(np.std(vals)) if vals.size else float("nan")
        rows.append({"lambda": lam, "n": int(vals.size), "mean_cr": mean, "var_cr": var, "std_cr": std})
        print(f"{lam:8.2f}  {vals.size:6d}  {mean:10.4f}  {var:10.4f}  {std:10.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "prediction_augmented_lambda_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df


if __name__ == "__main__":
    main()
