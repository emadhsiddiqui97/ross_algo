"""Evaluation script: sweep L/D ratio from loose (0.4) to tight (0.9)."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ross.traces import synthetic_trace
from ross.simulator import compare_policies


def run_ld_sweep(output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    ld_ratios = np.linspace(0.4, 0.9, 11)
    L = 100.0
    K = 5.0
    num_seeds = 30
    
    records = []

    print(f"Running L/D ratio sweep (L={L}, K={K}, {num_seeds} seeds/point)...")

    for ld in ld_ratios:
        D = float(int(round(L / ld)))
        trace = synthetic_trace(
            n_steps=int(D) + 100,
            avg_availability=0.6,
            mean_run_len=10.0,
            seed=int(ld * 1000),
        )

        seed_results = {"ROSS (greedy)": [], "ROSS (uniform)": [], "Uniform Progress": [], "Greedy": []}

        for seed in range(num_seeds):
            res = compare_policies(trace=trace, L=L, D=D, K=K, seed=seed)
            for pol_name, r in res.items():
                seed_results[pol_name].append(r.cost_savings_vs_on_demand_pct)

        row = {"L_over_D": ld, "D": D}
        for pol_name in seed_results:
            row[f"{pol_name}_savings_mean"] = float(np.mean(seed_results[pol_name]))
            row[f"{pol_name}_savings_std"] = float(np.std(seed_results[pol_name]))
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "sweep_ld_ratio.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Generate Plot
    plt.figure(figsize=(8, 5))
    for pol_name in ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy"]:
        plt.plot(df["L_over_D"], df[f"{pol_name}_savings_mean"], marker="o", label=pol_name)

    plt.title(f"Cost Savings vs On-Demand over L/D Ratio (K={K})")
    plt.xlabel("L / D (Slack tightness)")
    plt.ylabel("Cost Savings vs On-Demand (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "sweep_ld_ratio.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    run_ld_sweep()
