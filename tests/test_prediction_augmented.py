"""Tests for prediction-augmented ROSS (additive Track B feature)."""

import pytest
import numpy as np

from ross.forecast import PredictionAugmentedROSS, forecast_unavailability_rates
from ross.scheduler_ross import ROSSScheduler
from ross.simulator import run_policy
from ross.traces import Trace, synthetic_trace
from ross.validate import theoretical_competitive_ratio


def _action_sequence(scheduler, trace: Trace):
    scheduler.reset()
    actions = []
    max_steps = int(np.ceil(scheduler.D))
    for step in range(max_steps):
        if scheduler.is_done():
            break
        action = scheduler.decide(spot_available=trace[step])
        actions.append(action)
        time_rem = scheduler.D - scheduler.state.t
        dt_step = min(1.0, max(0.0, time_rem)) if time_rem > 0 else 1.0
        scheduler.advance(dt=dt_step, action=action)
    return actions, scheduler.state.cost, scheduler.state.t


class TestFallbackEquivalence:
    """lambda_=0 must match vanilla ROSS exactly (same seed, same trace)."""

    def test_lambda_zero_matches_vanilla_actions_cost_and_finish(self):
        L, D, K, seed = 40.0, 80.0, 4.0, 42
        trace = synthetic_trace(n_steps=int(D) + 20, avg_availability=0.55, mean_run_len=6.0, seed=7)
        vanilla = ROSSScheduler(L=L, D=D, K=K, warmup_mode="greedy", seed=seed)
        aug = PredictionAugmentedROSS(L=L, D=D, K=K, lambda_=0.0, warmup_mode="greedy", seed=seed)

        a_v, cost_v, t_v = _action_sequence(vanilla, trace)
        a_a, cost_a, t_a = _action_sequence(aug, trace)

        assert a_v == a_a
        assert abs(cost_v - cost_a) < 1e-12
        assert abs(t_v - t_a) < 1e-12


_COMBOS = [
    (20.0, 40.0, 2.0),
    (40.0, 50.0, 5.0),
    (50.0, 80.0, 10.0),
    (100.0, 200.0, 3.0),
]


class TestInvariantsAllLambda:
    """Deadline and Theorem-2 cost-ratio bounds hold for lambda in {0, 0.5, 1}."""

    @pytest.mark.parametrize("lambda_", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("L, D, K", _COMBOS)
    def test_deadline_and_cost_ratio_bound(self, lambda_: float, L: float, D: float, K: float):
        trace = synthetic_trace(n_steps=int(D) + 50, avg_availability=0.6, mean_run_len=8.0, seed=11)
        sched = PredictionAugmentedROSS(L=L, D=D, K=K, lambda_=lambda_, warmup_mode="greedy", seed=3)
        res = run_policy(scheduler=sched, trace=trace, dt=1.0, switch_penalty_pct=0.0)

        assert res.completed
        assert res.finish_time <= D + 1e-9

        th_cr = theoretical_competitive_ratio(D=D, L=L, K=K)
        realized_cr = res.total_cost / res.opt_cost if res.opt_cost > 0 else 1.0
        assert realized_cr <= th_cr + 0.05


class TestForecastSanity:
    def test_all_false_history_predicts_high_unavailability(self):
        history = [False] * 40
        rates = forecast_unavailability_rates(history, n_starts=8, delta=5.0, window_len=20.0)
        assert rates.shape == (8,)
        assert np.all(rates >= 0.99)

    def test_all_true_history_predicts_low_unavailability(self):
        history = [True] * 40
        rates = forecast_unavailability_rates(history, n_starts=8, delta=5.0, window_len=20.0)
        assert np.all(rates <= 0.01)
