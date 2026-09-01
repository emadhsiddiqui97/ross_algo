"""Evaluation script: sweep cost ratio K from 1 to 10."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ross.traces import synthetic_trace
from ross.simulator import compare_policies


def run_k_sweep(output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    k_values = np.linspace(1.0, 10.0, 10)
    L = 100.0
    D = 180.0  # L/D ~ 0.55
    num_seeds = 30
    
    records = []

    print(f"Running Cost Ratio K sweep (L={L}, D={D}, {num_seeds} seeds/point)...")

    trace = synthetic_trace(
        n_steps=int(D) + 100,
        avg_availability=0.6,
        mean_run_len=10.0,
        seed=42,
    )

    for K in k_values:
        seed_results = {"ROSS (greedy)": [], "ROSS (uniform)": [], "Uniform Progress": [], "Greedy": []}

        for seed in range(num_seeds):
            res = compare_policies(trace=trace, L=L, D=D, K=K, seed=seed)
            for pol_name, r in res.items():
                seed_results[pol_name].append(r.overhead_to_opt_pct)

        row = {"K": K}
        for pol_name in seed_results:
            row[f"{pol_name}_overhead_opt_mean"] = float(np.mean(seed_results[pol_name]))
            row[f"{pol_name}_overhead_opt_std"] = float(np.std(seed_results[pol_name]))
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "sweep_cost_ratio.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Generate Plot
    plt.figure(figsize=(8, 5))
    for pol_name in ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy"]:
        plt.plot(df["K"], df[f"{pol_name}_overhead_opt_mean"], marker="s", label=pol_name)

    plt.title(f"Overhead to OPT (%) vs Cost Ratio K (L={L}, D={D})")
    plt.xlabel("Cost Ratio K (On-Demand / Spot)")
    plt.ylabel("Overhead to OPT (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "sweep_cost_ratio.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    run_k_sweep()
