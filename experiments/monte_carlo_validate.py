"""Monte Carlo validation script to verify theoretical invariants across a parameter grid."""

import sys
import numpy as np
from ross.traces import synthetic_trace
from ross.validate import validate_ross_invariants, theoretical_competitive_ratio


def run_monte_carlo_grid():
    ld_ratios = [0.3, 0.5, 0.7, 0.9]
    k_values = [2.0, 5.0, 10.0]
    num_seeds = 50
    L = 100.0

    print("=================================================================")
    print("           ROSS THEORETICAL INVARIANT VALIDATION GRID            ")
    print("=================================================================")
    print(f"Workload L: {L} | Seeds per config: {num_seeds} | Tolerance eps: 0.05")
    print("-" * 65)

    all_passed = True
    total_configs = len(ld_ratios) * len(k_values)
    config_idx = 0

    for ld in ld_ratios:
        D = float(int(round(L / ld)))
        trace = synthetic_trace(
            n_steps=int(D) + 100,
            avg_availability=0.6,
            mean_run_len=8.0,
            seed=42,
        )

        for K in k_values:
            config_idx += 1
            th_cr = theoretical_competitive_ratio(D=D, L=L, K=K)
            
            # Test greedy warmup
            rep_greedy = validate_ross_invariants(
                trace=trace,
                L=L,
                D=D,
                K=K,
                num_seeds=num_seeds,
                warmup_mode="greedy",
                epsilon=0.05,
            )

            # Test uniform warmup
            rep_uniform = validate_ross_invariants(
                trace=trace,
                L=L,
                D=D,
                K=K,
                num_seeds=num_seeds,
                warmup_mode="uniform",
                epsilon=0.05,
            )

            passed = (rep_greedy.pass_rate_pct == 100.0) and (rep_uniform.pass_rate_pct == 100.0)
            if not passed:
                all_passed = False

            status = "PASS" if passed else "FAIL"
            print(
                f"[{config_idx:02d}/{total_configs}] L/D={ld:.1f} (D={D:.1f}), K={K:4.1f} | "
                f"Th_CR={th_cr:5.2f} | "
                f"Realized (Gr={rep_greedy.max_realized_cr:5.2f}, Un={rep_uniform.max_realized_cr:5.2f}) | "
                f"Status: {status}"
            )

            if not passed:
                if rep_greedy.violations:
                    print(f"   Greedy violations: {rep_greedy.violations[:3]}")
                if rep_uniform.violations:
                    print(f"   Uniform violations: {rep_uniform.violations[:3]}")

    print("-" * 65)
    if all_passed:
        print("ALL CONFIGURATIONS PASSED 100% INVARIANT CHECKS!")
        sys.exit(0)
    else:
        print("SOME INVARIANT CHECKS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_monte_carlo_grid()
