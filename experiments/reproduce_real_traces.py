"""Automated experiment harness to reproduce paper findings (Section V) on real cloud traces.

Evaluates ROSS (greedy and uniform), Uniform Progress, Greedy, and Offline OPT across all
real availability traces (data/availability/) and preemption traces (data/preemption/).

Generates summary CSV tables and high-resolution publication figures replicating Figures 2-5.
"""

import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ross.traces import Trace, load_all_traces
from ross.simulator import compare_policies, hindsight_optimal_cost


def run_deadline_tightness_sweep(
    traces: List[Trace],
    output_dir: str = "reports",
    L: float = 100.0,
    K: float = 5.0,
    num_seeds_per_window: int = 5,
    max_windows_per_trace: int = 15,
) -> pd.DataFrame:
    """Sweeps deadline tightness L/D in [0.4, 0.9] at fixed cost ratio K=5.0.
    
    Replicates Figure 2 (% Cost Savings) and Figure 3 (% Cost Difference to OPT).
    """
    os.makedirs(output_dir, exist_ok=True)
    ld_ratios = np.linspace(0.4, 0.9, 11)
    
    policy_names = ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy", "OPT"]
    records = []

    print(f"\n=================================================================")
    print(f"   EXPERIMENT 1: DEADLINE TIGHTNESS SWEEP (L/D in [0.4, 0.9])    ")
    print(f"=================================================================")
    print(f"Workload L={L:.1f}, Cost Ratio K={K:.1f}, Real Traces={len(traces)}")
    print("-" * 65)

    for ld in ld_ratios:
        D = float(int(round(L / ld)))
        window_size = int(D)
        
        savings_collector: Dict[str, List[float]] = {p: [] for p in policy_names}
        overhead_collector: Dict[str, List[float]] = {p: [] for p in policy_names}

        for tr in traces:
            stride = max(50, window_size // 2)
            windows = tr.sliding_windows(
                window_size=window_size,
                stride=stride,
                max_windows=max_windows_per_trace,
            )
            for win in windows:
                for seed in range(num_seeds_per_window):
                    results = compare_policies(
                        trace=win,
                        L=L,
                        D=D,
                        K=K,
                        seed=seed,
                        switch_penalty_pct=0.01,
                    )
                    # Baselines & ROSS
                    for pol in ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy"]:
                        res = results[pol]
                        savings_collector[pol].append(res.cost_savings_vs_on_demand_pct)
                        overhead_collector[pol].append(res.overhead_to_opt_pct)

                    # Offline OPT
                    opt_c = results["ROSS (greedy)"].opt_cost
                    od_c = results["ROSS (greedy)"].on_demand_only_cost
                    opt_savings = ((od_c - opt_c) / od_c) * 100.0 if od_c > 0 else 0.0
                    savings_collector["OPT"].append(opt_savings)
                    overhead_collector["OPT"].append(0.0)

        row = {"L_over_D": round(float(ld), 4), "D": D, "total_windows": len(savings_collector["ROSS (greedy)"])}
        for pol in policy_names:
            row[f"{pol}_savings_mean"] = float(np.mean(savings_collector[pol]))
            row[f"{pol}_savings_std"] = float(np.std(savings_collector[pol]))
            row[f"{pol}_overhead_opt_mean"] = float(np.mean(overhead_collector[pol]))
            row[f"{pol}_overhead_opt_std"] = float(np.std(overhead_collector[pol]))

        records.append(row)
        print(
            f"L/D={ld:.2f} (D={D:5.1f}) | "
            f"Savings: ROSS(u)={row['ROSS (uniform)_savings_mean']:5.1f}%, "
            f"ROSS(g)={row['ROSS (greedy)_savings_mean']:5.1f}%, "
            f"UP={row['Uniform Progress_savings_mean']:5.1f}%, "
            f"OPT={row['OPT_savings_mean']:5.1f}% | "
            f"Overhead to OPT: ROSS(u)={row['ROSS (uniform)_overhead_opt_mean']:5.1f}%, "
            f"UP={row['Uniform Progress_overhead_opt_mean']:5.1f}%"
        )

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "real_traces_ld_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Plot Figure 2: % Cost Savings vs L/D
    plt.figure(figsize=(9, 5.5))
    colors = {
        "ROSS (greedy)": "#1f77b4",
        "ROSS (uniform)": "#2ca02c",
        "Uniform Progress": "#ff7f0e",
        "Greedy": "#d62728",
        "OPT": "#9467bd",
    }
    markers = {
        "ROSS (greedy)": "o",
        "ROSS (uniform)": "s",
        "Uniform Progress": "^",
        "Greedy": "v",
        "OPT": "*",
    }
    for pol in policy_names:
        plt.plot(
            df["L_over_D"],
            df[f"{pol}_savings_mean"],
            label=pol,
            color=colors[pol],
            marker=markers[pol],
            linewidth=2,
            markersize=6,
        )

    plt.title(f"Figure 2: Cost Savings vs On-Demand over Deadline Tightness L/D (K={K:.1f})", fontsize=12, fontweight="bold")
    plt.xlabel("Deadline Tightness (L / D)", fontsize=11)
    plt.ylabel("Cost Savings vs On-Demand (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_real_traces_savings_vs_ld.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}")

    # Plot Figure 3: % Cost Difference to OPT vs L/D
    plt.figure(figsize=(9, 5.5))
    for pol in ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy"]:
        plt.plot(
            df["L_over_D"],
            df[f"{pol}_overhead_opt_mean"],
            label=pol,
            color=colors[pol],
            marker=markers[pol],
            linewidth=2,
            markersize=6,
        )

    plt.title(f"Figure 3: Cost Difference to OPT (%) over Deadline Tightness L/D (K={K:.1f})", fontsize=12, fontweight="bold")
    plt.xlabel("Deadline Tightness (L / D)", fontsize=11)
    plt.ylabel("Cost Difference to OPT (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_real_traces_overhead_vs_ld.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")

    return df


def run_cost_ratio_sweep(
    traces: List[Trace],
    output_dir: str = "reports",
    L: float = 100.0,
    num_seeds_per_window: int = 5,
    max_windows_per_trace: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweeps cost ratio K in [1.0, 10.0] under loose (L/D=0.5) and strict (L/D=0.8) deadline settings.
    
    Replicates Figure 4 (% Cost Savings vs K) and Figure 5 (% Cost Difference to OPT vs K).
    """
    os.makedirs(output_dir, exist_ok=True)
    k_values = np.linspace(1.0, 10.0, 10)
    policy_names = ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy", "OPT"]

    settings = [
        ("loose", 0.5, 200.0),
        ("strict", 0.8, 125.0),
    ]

    dfs = {}

    for setting_name, ld, D in settings:
        print(f"\n=================================================================")
        print(f"   EXPERIMENT 2: COST RATIO SWEEP ({setting_name.upper()} DEADLINE L/D={ld}, D={D})   ")
        print(f"=================================================================")
        print(f"Workload L={L:.1f}, K in [1.0, 10.0], Real Traces={len(traces)}")
        print("-" * 65)

        records = []
        window_size = int(D)

        for K in k_values:
            savings_collector: Dict[str, List[float]] = {p: [] for p in policy_names}
            overhead_collector: Dict[str, List[float]] = {p: [] for p in policy_names}

            for tr in traces:
                stride = max(50, window_size // 2)
                windows = tr.sliding_windows(
                    window_size=window_size,
                    stride=stride,
                    max_windows=max_windows_per_trace,
                )
                for win in windows:
                    for seed in range(num_seeds_per_window):
                        results = compare_policies(
                            trace=win,
                            L=L,
                            D=D,
                            K=K,
                            seed=seed,
                            switch_penalty_pct=0.01,
                        )
                        for pol in ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy"]:
                            res = results[pol]
                            savings_collector[pol].append(res.cost_savings_vs_on_demand_pct)
                            overhead_collector[pol].append(res.overhead_to_opt_pct)

                        opt_c = results["ROSS (greedy)"].opt_cost
                        od_c = results["ROSS (greedy)"].on_demand_only_cost
                        opt_savings = ((od_c - opt_c) / od_c) * 100.0 if od_c > 0 else 0.0
                        savings_collector["OPT"].append(opt_savings)
                        overhead_collector["OPT"].append(0.0)

            row = {"K": round(float(K), 2), "L_over_D": ld, "D": D, "total_windows": len(savings_collector["ROSS (greedy)"])}
            for pol in policy_names:
                row[f"{pol}_savings_mean"] = float(np.mean(savings_collector[pol]))
                row[f"{pol}_savings_std"] = float(np.std(savings_collector[pol]))
                row[f"{pol}_overhead_opt_mean"] = float(np.mean(overhead_collector[pol]))
                row[f"{pol}_overhead_opt_std"] = float(np.std(overhead_collector[pol]))

            records.append(row)
            print(
                f"K={K:4.1f} | "
                f"Savings: ROSS(u)={row['ROSS (uniform)_savings_mean']:5.1f}%, "
                f"ROSS(g)={row['ROSS (greedy)_savings_mean']:5.1f}%, "
                f"UP={row['Uniform Progress_savings_mean']:5.1f}%, "
                f"OPT={row['OPT_savings_mean']:5.1f}% | "
                f"Overhead to OPT: ROSS(u)={row['ROSS (uniform)_overhead_opt_mean']:5.1f}%, "
                f"UP={row['Uniform Progress_overhead_opt_mean']:5.1f}%"
            )

        df_setting = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, f"real_traces_k_sweep_{setting_name}.csv")
        df_setting.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        dfs[setting_name] = df_setting

    colors = {
        "ROSS (greedy)": "#1f77b4",
        "ROSS (uniform)": "#2ca02c",
        "Uniform Progress": "#ff7f0e",
        "Greedy": "#d62728",
        "OPT": "#9467bd",
    }
    markers = {
        "ROSS (greedy)": "o",
        "ROSS (uniform)": "s",
        "Uniform Progress": "^",
        "Greedy": "v",
        "OPT": "*",
    }

    # Plot Figure 4: % Cost Savings vs K (2 subplots: Loose & Strict)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for pol in policy_names:
        ax1.plot(
            dfs["loose"]["K"],
            dfs["loose"][f"{pol}_savings_mean"],
            label=pol,
            color=colors[pol],
            marker=markers[pol],
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            dfs["strict"]["K"],
            dfs["strict"][f"{pol}_savings_mean"],
            label=pol,
            color=colors[pol],
            marker=markers[pol],
            linewidth=2,
            markersize=6,
        )

    ax1.set_title("Loose Deadline (L/D = 0.5, D = 200)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cost Ratio K (On-Demand / Spot)", fontsize=11)
    ax1.set_ylabel("Cost Savings vs On-Demand (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)

    ax2.set_title("Strict Deadline (L/D = 0.8, D = 125)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cost Ratio K (On-Demand / Spot)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle("Figure 4: Cost Savings vs On-Demand over Cost Ratio K", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_real_traces_savings_vs_k.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {fig4_path}")

    # Plot Figure 5: % Cost Difference to OPT vs K (2 subplots: Loose & Strict)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for pol in ["ROSS (greedy)", "ROSS (uniform)", "Uniform Progress", "Greedy"]:
        ax1.plot(
            dfs["loose"]["K"],
            dfs["loose"][f"{pol}_overhead_opt_mean"],
            label=pol,
            color=colors[pol],
            marker=markers[pol],
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            dfs["strict"]["K"],
            dfs["strict"][f"{pol}_overhead_opt_mean"],
            label=pol,
            color=colors[pol],
            marker=markers[pol],
            linewidth=2,
            markersize=6,
        )

    ax1.set_title("Loose Deadline (L/D = 0.5, D = 200)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cost Ratio K (On-Demand / Spot)", fontsize=11)
    ax1.set_ylabel("Cost Difference to OPT (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)

    ax2.set_title("Strict Deadline (L/D = 0.8, D = 125)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cost Ratio K (On-Demand / Spot)", fontsize=11)
    ax2.set_ylabel("Cost Difference to OPT (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle("Figure 5: Cost Difference to OPT (%) over Cost Ratio K", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_real_traces_overhead_vs_k.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {fig5_path}")

    return dfs["loose"], dfs["strict"]


def main():
    print("Loading all real cloud trace datasets from data/...")
    traces = load_all_traces(data_dir="data")
    print(f"Successfully loaded {len(traces)} real trace files.")

    # 1. Deadline tightness sweep (Figures 2 & 3)
    run_deadline_tightness_sweep(traces=traces, output_dir="reports")

    # 2. Cost ratio sweep (Figures 4 & 5)
    run_cost_ratio_sweep(traces=traces, output_dir="reports")

    print("\n=================================================================")
    print("      ALL REAL TRACE REPRODUCTION EXPERIMENTS COMPLETED!         ")
    print("=================================================================")


if __name__ == "__main__":
    main()
