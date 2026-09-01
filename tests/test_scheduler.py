"""Comprehensive test suite for ROSS and baseline scheduling algorithms."""

import math
import pytest
import numpy as np

from ross.core import Action, Phase
from ross.traces import Trace, synthetic_trace
from ross.scheduler_ross import ROSSScheduler
from ross.scheduler_baselines import UniformProgressScheduler, GreedyScheduler
from ross.simulator import run_policy, compare_policies
from ross.validate import theoretical_competitive_ratio, validate_ross_invariants


class TestROSSCheckpoint:
    """Step 2 Checkpoint: No-spot deadline completion guarantee across parameters."""

    @pytest.mark.parametrize("L, D, K", [
        (10, 10, 2.0),
        (20, 40, 2.0),
        (30, 100, 5.0),
        (50, 80, 10.0),
        (100, 120, 20.0),
        (100, 200, 3.0),
    ])
    @pytest.mark.parametrize("warmup_mode", ["greedy", "uniform"])
    def test_no_spot_completion_before_deadline(self, L: float, D: float, K: float, warmup_mode: str):
        """When spot is never available, job must complete on or before deadline D."""
        trace = Trace(availability=np.zeros(int(D) + 50, dtype=bool), name="all_false")
        scheduler = ROSSScheduler(L=L, D=D, K=K, warmup_mode=warmup_mode, seed=42)

        res = run_policy(scheduler=scheduler, trace=trace, dt=1.0)
        assert res.completed, f"Failed to complete job for L={L}, D={D}, K={K}, mode={warmup_mode}"
        assert res.finish_time <= D + 1e-9, f"Missed deadline: finish_time={res.finish_time} > D={D}"
        assert res.spot_time == 0.0
        assert abs(res.on_demand_time - L) < 1e-9


class TestFormulaeAndThresholds:
    """Step 8: Mathematical formulas and transition verifications."""

    def test_warmup_threshold_calculation(self):
        # For K = 4: sqrt(K) = 2 -> (1 + 2*2)/(1 + 2) = 5/3 = 1.6667
        s = ROSSScheduler(L=10, D=20, K=4.0)
        assert abs(s.warmup_threshold - (5.0 / 3.0)) < 1e-9

        # For K = 1: sqrt(K) = 1 -> (1 + 2*1)/(1 + 1) = 3/2 = 1.5
        s1 = ROSSScheduler(L=10, D=20, K=1.0)
        assert abs(s1.warmup_threshold - 1.5) < 1e-9

        # For K = 9: sqrt(K) = 3 -> (1 + 2*3)/(1 + 3) = 7/4 = 1.75
        s9 = ROSSScheduler(L=10, D=20, K=9.0)
        assert abs(s9.warmup_threshold - 1.75) < 1e-9

    def test_delta_formula_at_injection(self):
        # K = 4 -> sqrt(K) = 2. L = 30, D = 40.
        # At start, D/L = 40/30 = 1.333 < 1.6667 threshold, so injection setup triggers immediately at t=0
        s = ROSSScheduler(L=30, D=40, K=4.0, seed=123)
        assert s.state.phase == Phase.INJECTION
        assert s.xi1 == 0.0
        # delta = (L - phi) / (1 + sqrt(K)) = 30 / (1 + 2) = 10.0
        assert abs(s.delta - 10.0) < 1e-9
        assert s.sigma is not None
        assert 0.0 <= s.sigma <= (30.0 - 10.0)
        assert s.injection_window_end == 30.0


class TestEdgeCases:
    """Step 8: Boundary edge cases."""

    def test_zero_slack_L_equals_D(self):
        """When L == D, there is zero slack, scheduler must force on-demand immediately."""
        s = ROSSScheduler(L=20, D=20, K=5.0)
        # Slack is 0 -> immediately FORCED phase
        assert s.state.phase == Phase.FORCED
        assert s.decide(spot_available=True) == Action.ON_DEMAND
        assert s.decide(spot_available=False) == Action.ON_DEMAND

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            ROSSScheduler(L=0, D=10, K=2.0)
        with pytest.raises(ValueError):
            ROSSScheduler(L=20, D=10, K=2.0)  # D < L
        with pytest.raises(ValueError):
            ROSSScheduler(L=10, D=20, K=0.5)  # K < 1.0


class TestBaselines:
    """Baseline schedulers behavior."""

    def test_greedy_never_rents_on_demand_when_spot_is_available(self):
        greedy = GreedyScheduler(L=50, D=100, K=5.0)
        for _ in range(30):
            action = greedy.decide(spot_available=True)
            assert action == Action.SPOT
            greedy.advance(1.0, action)

    def test_uniform_progress_linear_pacing(self):
        up = UniformProgressScheduler(L=50, D=100, K=5.0)
        # At t=0, behind pace (target=0, phi=0), rents
        assert up.decide(spot_available=True) == Action.SPOT
        assert up.decide(spot_available=False) == Action.ON_DEMAND


class TestTheoreticalInvariants:
    """Validation harness sanity test."""

    def test_theoretical_competitive_ratio_formula(self):
        # Case 1: D/L >= threshold
        # K = 4 -> threshold = 5/3 = 1.6667. L=10, D=20 -> D/L = 2.0 >= 1.6667 -> CR = sqrt(4) = 2.0
        assert abs(theoretical_competitive_ratio(D=20, L=10, K=4.0) - 2.0) < 1e-9

        # Case 2: D/L < threshold
        # K = 4. L=10, D=12 -> D/L = 1.2 < 1.6667
        # CR = 1 + (4 - 1)*(2 - 1.2) = 1 + 3*(0.8) = 3.4
        assert abs(theoretical_competitive_ratio(D=12, L=10, K=4.0) - 3.4) < 1e-9

    def test_monte_carlo_invariants_pass_on_synthetic_trace(self):
        trace = synthetic_trace(n_steps=300, avg_availability=0.6, mean_run_len=5.0, seed=99)
        report = validate_ross_invariants(
            trace=trace,
            L=50,
            D=100,
            K=4.0,
            num_seeds=30,
            warmup_mode="uniform",
            epsilon=0.05,
        )
        assert report.pass_rate_pct == 100.0, f"Invariant failures: {report.violations}"
