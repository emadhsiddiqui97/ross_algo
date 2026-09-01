"""Theoretical validation harness and invariant checks for ROSS."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ross.simulator import SimulationResult, run_policy
from ross.scheduler_ross import ROSSScheduler
from ross.traces import Trace


def theoretical_competitive_ratio(D: float, L: float, K: float) -> float:
    """Computes the theoretical upper bound on the competitive ratio (Theorem 2).
    
    CR_ROSS(D, L, K) = sqrt(K)                          if D >= (1+2*sqrt(K))/(1+sqrt(K)) * L
                     = 1 + (K-1)*(2 - D/L)              otherwise
    """
    sqrt_k = math.sqrt(K)
    threshold = ((1.0 + 2.0 * sqrt_k) / (1.0 + sqrt_k)) * L
    if D >= threshold - 1e-9:
        return sqrt_k
    else:
        return 1.0 + (K - 1.0) * (2.0 - (D / L))


@dataclass
class ValidationReport:
    """Summary report of invariant checks over multiple runs/seeds."""
    total_runs: int
    deadline_passed: int
    cost_ratio_passed: int
    all_invariants_passed: int
    pass_rate_pct: float
    max_realized_cr: float
    mean_realized_cr: float
    theoretical_cr: float
    violations: List[str]


def validate_ross_invariants(
    trace: Trace,
    L: float,
    D: float,
    K: float,
    num_seeds: int = 50,
    warmup_mode: str = "greedy",
    epsilon: float = 0.05,
    switch_penalty_pct: float = 0.0,
) -> ValidationReport:
    """Runs ROSS across multiple random seeds and validates theoretical invariants.
    
    Args:
        trace: Spot availability trace.
        L: Workload compute requirement.
        D: Deadline time.
        K: Cost ratio.
        num_seeds: Number of Monte Carlo seeds to test.
        warmup_mode: 'greedy' or 'uniform'.
        epsilon: Discretization/float slack tolerance for cost ratio.
        switch_penalty_pct: Switch penalty (default 0.0 for pure theoretical bound check).
        
    Returns:
        ValidationReport instance.
    """
    th_cr = theoretical_competitive_ratio(D=D, L=L, K=K)
    deadline_passed = 0
    cr_passed = 0
    all_passed = 0
    realized_crs = []
    violations = []

    for seed in range(num_seeds):
        scheduler = ROSSScheduler(L=L, D=D, K=K, warmup_mode=warmup_mode, seed=seed)
        res = run_policy(
            scheduler=scheduler,
            trace=trace,
            dt=1.0,
            switch_penalty_pct=switch_penalty_pct,
        )

        # 1. Deadline invariant check
        dl_ok = res.completed and (res.finish_time <= D + 1e-9)
        if dl_ok:
            deadline_passed += 1
        else:
            violations.append(
                f"[Seed {seed}] Deadline missed: completed={res.completed}, finish_time={res.finish_time}, D={D}"
            )

        # 2. Cost ratio invariant check
        realized_cr = res.total_cost / res.opt_cost if res.opt_cost > 0 else 1.0
        realized_crs.append(realized_cr)
        
        cr_ok = realized_cr <= (th_cr + epsilon)
        if cr_ok:
            cr_passed += 1
        else:
            violations.append(
                f"[Seed {seed}] CR violation: realized={realized_cr:.4f} > theoretical={th_cr:.4f} + eps({epsilon})"
            )

        if dl_ok and cr_ok:
            all_passed += 1

    pass_rate = (all_passed / num_seeds) * 100.0 if num_seeds > 0 else 0.0

    return ValidationReport(
        total_runs=num_seeds,
        deadline_passed=deadline_passed,
        cost_ratio_passed=cr_passed,
        all_invariants_passed=all_passed,
        pass_rate_pct=pass_rate,
        max_realized_cr=float(np.max(realized_crs)) if realized_crs else 0.0,
        mean_realized_cr=float(np.mean(realized_crs)) if realized_crs else 0.0,
        theoretical_cr=th_cr,
        violations=violations,
    )
