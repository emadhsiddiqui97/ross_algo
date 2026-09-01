"""Tests for core interfaces and trace utilities."""

import pytest
import numpy as np

from ross.core import Action, Phase, SchedulerState
from ross.traces import synthetic_trace, Trace
from ross.simulator import hindsight_optimal_cost


def test_action_and_phase_enums():
    assert Action.SPOT.name == "SPOT"
    assert Action.ON_DEMAND.name == "ON_DEMAND"
    assert Action.IDLE.name == "IDLE"

    assert Phase.WARMUP.name == "WARMUP"
    assert Phase.INJECTION.name == "INJECTION"
    assert Phase.CATCHUP.name == "CATCHUP"
    assert Phase.FORCED.name == "FORCED"
    assert Phase.DONE.name == "DONE"


def test_scheduler_state():
    state = SchedulerState(t=0.0, phi=0.0)
    assert state.t == 0.0
    assert state.phi == 0.0
    assert state.phase == Phase.WARMUP
    assert state.cost == 0.0
    assert state.switches == 0


def test_synthetic_trace_generation():
    trace = synthetic_trace(n_steps=1000, avg_availability=0.7, mean_run_len=10.0, seed=42)
    assert len(trace) == 1000
    assert isinstance(trace.availability, np.ndarray)
    assert trace.availability.dtype == bool
    # Average availability should be close to 0.7
    assert 0.60 <= trace.spot_availability_rate <= 0.80


def test_hindsight_optimal_cost():
    # 100 steps, all available, L=50, D=100, K=5.0
    all_avail = Trace(availability=np.ones(100, dtype=bool))
    opt = hindsight_optimal_cost(D=100, L=50, K=5.0, trace=all_avail)
    # All 50 units should come from spot at cost 1.0 each -> 50.0
    assert opt == 50.0

    # 100 steps, none available, L=50, D=100, K=5.0
    no_avail = Trace(availability=np.zeros(100, dtype=bool))
    opt_no = hindsight_optimal_cost(D=100, L=50, K=5.0, trace=no_avail)
    # All 50 units must come from On-Demand at cost K=5.0 each -> 250.0
    assert opt_no == 250.0

    # 100 steps, 30 spot available, L=50, D=100, K=5.0
    partial_avail = Trace(availability=np.array([True]*30 + [False]*70))
    opt_part = hindsight_optimal_cost(D=100, L=50, K=5.0, trace=partial_avail)
    # 30 * 1.0 + 20 * 5.0 = 130.0
    assert opt_part == 130.0
