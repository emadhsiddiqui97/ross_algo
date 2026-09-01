"""Simulator for trace-driven evaluation of scheduling policies."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np

from ross.core import Action, BaseScheduler
from ross.traces import Trace
from ross.scheduler_ross import ROSSScheduler
from ross.scheduler_baselines import UniformProgressScheduler, GreedyScheduler


@dataclass
class SimulationResult:
    """Detailed results from a single simulation run."""
    policy_name: str
    completed: bool
    finish_time: float
    total_cost: float
    on_demand_time: float
    spot_time: float
    idle_time: float
    n_switches: int
    switch_overhead_cost: float
    cost_savings_vs_on_demand_pct: float
    overhead_to_opt_pct: float
    opt_cost: float
    on_demand_only_cost: float


def hindsight_optimal_cost(D: float, L: float, K: float, trace: Trace) -> float:
    """Computes the offline hindsight optimal cost (OPT) over the horizon [0, D].
    
    OPT uses available spot ticks (up to L) within deadline D, and fills the remainder with On-Demand.
    """
    d_int = int(min(math.ceil(D), len(trace)))
    spot_available_count = int(np.sum(trace.availability[:d_int]))
    spot_used = min(float(spot_available_count), float(L))
    on_demand_needed = max(0.0, float(L) - spot_used)
    
    # Spot cost is 1.0 per unit, On-Demand cost is K per unit
    return spot_used * 1.0 + on_demand_needed * float(K)


def run_policy(
    scheduler: BaseScheduler,
    trace: Trace,
    dt: float = 1.0,
    switch_penalty_pct: float = 0.01,
) -> SimulationResult:
    """Runs a scheduling policy tick-by-tick across an availability trace.
    
    Args:
        scheduler: Instantiated scheduler implementing BaseScheduler.
        trace: Spot availability trace.
        dt: Tick step duration.
        switch_penalty_pct: Overhead penalty per state switch (fraction of K per switch).
        
    Returns:
        SimulationResult instance.
    """
    scheduler.reset()
    L = scheduler.L
    D = scheduler.D
    K = scheduler.K

    max_steps = int(math.ceil(D / dt))
    if max_steps > len(trace):
        raise ValueError(
            f"Trace length ({len(trace)}) is shorter than simulation horizon D/dt ({max_steps})"
        )

    for step in range(max_steps):
        if scheduler.is_done():
            break

        spot_avail = trace[step]
        action = scheduler.decide(spot_available=spot_avail)
        # Advance by full tick or fraction up to D
        time_rem = D - scheduler.state.t
        dt_step = min(dt, max(0.0, time_rem)) if time_rem > 0 else dt
        scheduler.advance(dt=dt_step, action=action)

    completed = scheduler.is_done()
    finish_time = scheduler.state.t if completed else scheduler.D
    
    # Calculate costs and state-switching latency overhead (~1% of compute length L as per Section V-A)
    base_cost = scheduler.state.cost
    switch_penalty_unit = switch_penalty_pct * L
    switch_overhead = scheduler.state.switches * switch_penalty_unit
    total_cost = base_cost + switch_overhead

    on_demand_only_cost = L * K
    opt_cost = hindsight_optimal_cost(D=D, L=L, K=K, trace=trace)

    cost_savings = ((on_demand_only_cost - total_cost) / on_demand_only_cost) * 100.0 if on_demand_only_cost > 0 else 0.0
    overhead_to_opt = ((total_cost - opt_cost) / opt_cost) * 100.0 if opt_cost > 0 else 0.0

    policy_name = getattr(scheduler, "name", scheduler.__class__.__name__)
    if isinstance(scheduler, ROSSScheduler):
        policy_name = f"ROSS ({scheduler.warmup_mode})"

    return SimulationResult(
        policy_name=policy_name,
        completed=completed,
        finish_time=finish_time,
        total_cost=total_cost,
        on_demand_time=scheduler.state.on_demand_time,
        spot_time=scheduler.state.spot_time,
        idle_time=scheduler.state.idle_time,
        n_switches=scheduler.state.switches,
        switch_overhead_cost=switch_overhead,
        cost_savings_vs_on_demand_pct=cost_savings,
        overhead_to_opt_pct=overhead_to_opt,
        opt_cost=opt_cost,
        on_demand_only_cost=on_demand_only_cost,
    )


def compare_policies(
    trace: Trace,
    L: float,
    D: float,
    K: float,
    seed: Optional[int] = None,
    switch_penalty_pct: float = 0.01,
) -> Dict[str, SimulationResult]:
    """Runs ROSS (greedy and uniform), UniformProgress, and Greedy baselines on the same trace."""
    policies = {
        "ROSS (greedy)": ROSSScheduler(L=L, D=D, K=K, warmup_mode="greedy", seed=seed),
        "ROSS (uniform)": ROSSScheduler(L=L, D=D, K=K, warmup_mode="uniform", seed=seed),
        "Uniform Progress": UniformProgressScheduler(L=L, D=D, K=K),
        "Greedy": GreedyScheduler(L=L, D=D, K=K),
    }

    results = {}
    for name, policy in policies.items():
        results[name] = run_policy(
            scheduler=policy,
            trace=trace,
            dt=1.0,
            switch_penalty_pct=switch_penalty_pct,
        )
    return results
