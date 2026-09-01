"""Automated reproduction tests for real cloud traces and Section V paper findings.

Validates:
1. Trace loader discovery and multi-format parsing (JSON/CSV, qualitative labels, continuous arrays).
2. Zero Deadline Violations: ROSS completes phi = L on or before D across all real traces.
3. Loose Deadline Savings: ROSS achieves higher cost savings than Uniform Progress when
   L/D <= (1+sqrt(K))/(1+2*sqrt(K)) (K-dependent crossover from Theorem 2).
4. Competitive Ratio Bound: Realized cost ratio relative to OPT stays bounded within
   the two-case Theorem 2 bound (sqrt(K) loose, 1+(K-1)*(2-D/L) tight).
5. Switching latency penalty overhead verification (scales with K).
"""

import math
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from ross.core import Action, Phase
from ross.traces import (
    Trace,
    discover_trace_files,
    load_all_availability_traces,
    load_all_preemption_traces,
    load_all_traces,
    load_availability_trace,
    load_preemption_trace,
    load_trace,
    parse_raw_availability,
)
from ross.scheduler_ross import ROSSScheduler
from ross.scheduler_baselines import UniformProgressScheduler, GreedyScheduler
from ross.simulator import run_policy, compare_policies, hindsight_optimal_cost
from ross.validate import theoretical_competitive_ratio, loose_deadline_ld_threshold


class TestDatasetDiscoveryAndParsing:
    """Tests for dataset auto-discovery, multi-format parsing, and sliding window slicing."""

    def test_directory_auto_discovery(self):
        """Verifies that discover_trace_files finds all availability and preemption traces."""
        discovered = discover_trace_files("data")
        assert "availability" in discovered
        assert "preemption" in discovered
        assert len(discovered["availability"]) == 20, f"Expected 20 availability traces, found {len(discovered['availability'])}"
        assert len(discovered["preemption"]) == 14, f"Expected 14 preemption traces, found {len(discovered['preemption'])}"

    def test_load_all_traces_helper(self):
        """Verifies bulk loader functions."""
        avail_traces = load_all_availability_traces("data", max_steps=100)
        preempt_traces = load_all_preemption_traces("data", max_steps=100)
        all_traces = load_all_traces("data", max_steps=100)

        assert len(avail_traces) == 20
        assert len(preempt_traces) == 14
        assert len(all_traces) == 34
        for t in all_traces:
            assert isinstance(t, Trace)
            assert len(t) == 100
            assert t.availability.dtype == bool

    def test_qualitative_and_numeric_parsing(self):
        """Tests parsing qualitative labels (high/low), numbers (>0), booleans, and dicts."""
        raw_values = [
            "HIGH", "high", "High", "low", "medium", "unavailable",
            True, False, 1, 0, 4, 16,
            {"status": "high"}, {"status": "low"}, {"available": True}
        ]
        parsed = parse_raw_availability(raw_values)
        expected = [
            True, True, True, False, False, False,
            True, False, True, False, True, True,
            True, False, True
        ]
        assert parsed.tolist() == expected

    def test_csv_trace_parsing(self):
        """Tests loading availability and preemption traces from CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Spotlake style CSV with qualitative labels
            csv_path1 = os.path.join(tmpdir, "spotlake_sample.csv")
            df1 = pd.DataFrame({"availability": ["high", "low", "high", "high", "medium"]})
            df1.to_csv(csv_path1, index=False)

            t1 = load_availability_trace(csv_path1)
            assert len(t1) == 5
            assert t1.availability.tolist() == [True, False, True, True, False]

            # 2. Preemption style CSV with binary flags
            csv_path2 = os.path.join(tmpdir, "preemption_sample.csv")
            df2 = pd.DataFrame({"spot_available": [1, 1, 0, 1, 0]})
            df2.to_csv(csv_path2, index=False)

            t2 = load_preemption_trace(csv_path2)
            assert len(t2) == 5
            assert t2.availability.tolist() == [True, True, False, True, False]

    def test_sliding_window_slicing(self):
        """Tests .slice() and .sliding_windows() on real traces."""
        discovered = discover_trace_files("data")
        sample_path = discovered["availability"][0]
        full_trace = load_availability_trace(sample_path)

        # Slice test
        sliced = full_trace.slice(start=100, length=250)
        assert len(sliced) == 250
        assert np.array_equal(sliced.availability, full_trace.availability[100:350])

        # Sliding windows test
        windows = full_trace.sliding_windows(window_size=200, stride=500, max_windows=4)
        assert len(windows) == 4
        for i, win in enumerate(windows):
            assert len(win) == 200
            expected_start = i * 500
            assert np.array_equal(win.availability, full_trace.availability[expected_start : expected_start + 200])


class TestZeroDeadlineViolations:
    """Verifies that ROSS completes job compute requirement (phi = L) on or before deadline D."""

    @pytest.mark.parametrize("file_idx", [0, 4, 8, 12, 16, 19])
    @pytest.mark.parametrize("ld_ratio", [0.4, 0.6, 0.85])
    @pytest.mark.parametrize("warmup_mode", ["greedy", "uniform"])
    def test_zero_deadline_violations_availability_traces(self, file_idx: int, ld_ratio: float, warmup_mode: str):
        discovered = discover_trace_files("data")["availability"]
        path = discovered[file_idx]
        L = 50.0
        D = float(int(round(L / ld_ratio)))
        K = 5.0

        trace = load_availability_trace(path, start_step=0, length=int(D) + 50)
        if len(trace) < int(D):
            pytest.skip(f"Trace {path} shorter than horizon {D}")

        scheduler = ROSSScheduler(L=L, D=D, K=K, warmup_mode=warmup_mode, seed=42)
        res = run_policy(scheduler=scheduler, trace=trace, dt=1.0, switch_penalty_pct=0.01)

        assert res.completed, f"Job failed to complete on trace {path} (L={L}, D={D}, mode={warmup_mode})"
        assert res.finish_time <= D + 1e-9, f"Missed deadline: finish_time={res.finish_time} > D={D} on trace {path}"

    @pytest.mark.parametrize("file_idx", [0, 3, 6, 9, 13])
    @pytest.mark.parametrize("ld_ratio", [0.5, 0.75, 0.9])
    @pytest.mark.parametrize("warmup_mode", ["greedy", "uniform"])
    def test_zero_deadline_violations_preemption_traces(self, file_idx: int, ld_ratio: float, warmup_mode: str):
        discovered = discover_trace_files("data")["preemption"]
        path = discovered[file_idx]
        L = 50.0
        D = float(int(round(L / ld_ratio)))
        K = 4.0

        trace = load_preemption_trace(path, start_step=0, length=int(D) + 50)
        if len(trace) < int(D):
            pytest.skip(f"Trace {path} shorter than horizon {D}")

        scheduler = ROSSScheduler(L=L, D=D, K=K, warmup_mode=warmup_mode, seed=123)
        res = run_policy(scheduler=scheduler, trace=trace, dt=1.0, switch_penalty_pct=0.01)

        assert res.completed, f"Job failed to complete on trace {path} (L={L}, D={D}, mode={warmup_mode})"
        assert res.finish_time <= D + 1e-9, f"Missed deadline: finish_time={res.finish_time} > D={D} on trace {path}"


class TestLooseDeadlineSavings:
    """Verifies that ROSS achieves higher cost savings than Uniform Progress in the loose-deadline
    regime, where L/D <= (1+sqrt(K))/(1+2*sqrt(K)) (K-dependent crossover from Theorem 2)."""

    @pytest.mark.parametrize("K", [2.0, 5.0, 10.0])
    def test_ross_savings_over_uniform_progress_per_k(self, K: float):
        """Under loose deadlines (L/D well below the K-dependent crossover), ROSS exploits spot better."""
        crossover_ld = loose_deadline_ld_threshold(K)
        # Pick L/D safely inside the loose regime (80% of the crossover)
        ld_ratio = crossover_ld * 0.80
        L = 100.0
        D = float(int(round(L / ld_ratio)))

        discovered = discover_trace_files("data")
        target_files = [
            f for f in discovered["availability"] + discovered["preemption"]
            if any(k in f for k in ["us-east-1c", "us-east-1d", "us-east-1f", "us-west-2a", "us-west-2b"])
        ]

        total_windows = 0
        ross_savings_sum = 0.0
        up_savings_sum = 0.0
        ross_outperformed_or_tied = 0

        for path in target_files:
            tr = load_trace(path)
            windows = tr.sliding_windows(window_size=int(D), stride=300, max_windows=5)
            for win in windows:
                res = compare_policies(trace=win, L=L, D=D, K=K, seed=42, switch_penalty_pct=0.01)
                ross_u_sav = res["ROSS (uniform)"].cost_savings_vs_on_demand_pct
                up_sav = res["Uniform Progress"].cost_savings_vs_on_demand_pct

                ross_savings_sum += ross_u_sav
                up_savings_sum += up_sav
                total_windows += 1

                if ross_u_sav >= up_sav - 1e-4:
                    ross_outperformed_or_tied += 1

        assert total_windows > 0
        avg_ross_savings = ross_savings_sum / total_windows
        avg_up_savings = up_savings_sum / total_windows

        assert avg_ross_savings >= avg_up_savings - 0.5, (
            f"K={K}: Expected ROSS savings ({avg_ross_savings:.2f}%) >= UP savings ({avg_up_savings:.2f}%)"
        )
        assert (ross_outperformed_or_tied / total_windows) >= 0.70, (
            f"K={K}: ROSS should match or beat UP in >= 70% of loose windows, "
            f"got {ross_outperformed_or_tied}/{total_windows}"
        )

    @pytest.mark.parametrize("K", [2.0, 5.0, 10.0])
    def test_crossover_boundary_above_and_below(self, K: float):
        """Test that the loose-deadline advantage holds just below the K-dependent crossover
        and does not necessarily hold just above it."""
        crossover_ld = loose_deadline_ld_threshold(K)

        # Just below crossover (loose side): L/D = crossover - 0.05
        ld_below = crossover_ld - 0.05
        L = 100.0
        D_below = float(int(round(L / ld_below)))

        # Just above crossover (tight side): L/D = crossover + 0.05
        ld_above = min(crossover_ld + 0.05, 0.99)
        D_above = float(int(round(L / ld_above)))

        discovered = discover_trace_files("data")
        sample_files = [discovered["availability"][1], discovered["preemption"][1]]

        for path in sample_files:
            tr = load_trace(path)

            # Below crossover (loose): ROSS should generally match or beat Uniform Progress
            windows_below = tr.sliding_windows(window_size=int(D_below), stride=400, max_windows=4)
            for win in windows_below:
                res = compare_policies(trace=win, L=L, D=D_below, K=K, seed=42, switch_penalty_pct=0.01)
                ross_sav = res["ROSS (uniform)"].cost_savings_vs_on_demand_pct
                up_sav = res["Uniform Progress"].cost_savings_vs_on_demand_pct
                # In the loose regime, ROSS should perform at least competitively
                assert ross_sav >= up_sav - 2.0, (
                    f"K={K}, L/D={ld_below:.3f} (loose side): "
                    f"ROSS savings {ross_sav:.2f}% fell far below UP savings {up_sav:.2f}%"
                )

            # Above crossover (tight): just verify both still complete on time (no savings assertion)
            windows_above = tr.sliding_windows(window_size=int(D_above), stride=400, max_windows=4)
            for win in windows_above:
                res = compare_policies(trace=win, L=L, D=D_above, K=K, seed=42, switch_penalty_pct=0.01)
                assert res["ROSS (uniform)"].completed, (
                    f"K={K}, L/D={ld_above:.3f} (tight side): ROSS failed to complete"
                )


class TestCompetitiveRatioBound:
    """Verifies that realized cost ratio relative to OPT stays bounded within Theorem 2 bound:
    
    CR_ROSS(D, L, K) = sqrt(K)                      if D/L >= (1 + 2*sqrt(K)) / (1 + sqrt(K))
                     = 1 + (K - 1) * (2 - D/L)      otherwise
    """

    @pytest.mark.parametrize("K", [2.0, 5.0, 9.0])
    @pytest.mark.parametrize("ld_ratio", [0.5, 0.8])  # loose (D/L=2.0) and tight (D/L=1.25) regimes
    def test_realized_competitive_ratio_on_real_traces(self, K: float, ld_ratio: float):
        discovered = discover_trace_files("data")
        sample_files = [discovered["availability"][1], discovered["preemption"][1]]

        L = 50.0
        D = float(int(round(L / ld_ratio)))
        th_cr = theoretical_competitive_ratio(D=D, L=L, K=K)
        epsilon = 0.15  # Discretization and sliding window tolerance

        for path in sample_files:
            tr = load_trace(path)
            windows = tr.sliding_windows(window_size=int(D), stride=400, max_windows=4)
            for win in windows:
                for seed in [0, 42, 100]:
                    scheduler = ROSSScheduler(L=L, D=D, K=K, warmup_mode="uniform", seed=seed)
                    res = run_policy(scheduler=scheduler, trace=win, dt=1.0, switch_penalty_pct=0.0)
                    realized_cr = res.total_cost / res.opt_cost if res.opt_cost > 0 else 1.0

                    assert realized_cr <= th_cr + epsilon, (
                        f"CR violation on {win.name} (L={L}, D={D}, K={K}, L/D={ld_ratio}): "
                        f"realized={realized_cr:.3f} > theoretical={th_cr:.3f} + eps({epsilon})"
                    )


class TestStateSwitchingPenalty:
    """Verifies state-switching latency penalty calculation and switch tracking in simulator.py."""

    def test_idle_transitions_not_counted_as_switches(self):
        """Transitions involving IDLE must NOT count as switches; only SPOT<->ON_DEMAND transitions do."""
        # 1. SPOT -> IDLE -> SPOT sequence: switches must stay 0
        trace_idle = Trace(availability=np.array([True, False, True] + [True] * 7, dtype=bool), name="spot_idle_spot")
        greedy = GreedyScheduler(L=2.0, D=10.0, K=5.0)
        # Step 0: spot -> SPOT (phi=1)
        # Step 1: no spot -> IDLE (phi=1)
        # Step 2: spot -> SPOT (phi=2, done)
        res_idle = run_policy(scheduler=greedy, trace=trace_idle, dt=1.0)
        assert res_idle.completed
        assert res_idle.spot_time == 2.0
        assert res_idle.idle_time == 1.0
        assert res_idle.n_switches == 0, f"Expected 0 switches for SPOT->IDLE->SPOT, got {res_idle.n_switches}"

        # 2. SPOT -> ON_DEMAND sequence: switches must be exactly 1
        trace_od = Trace(availability=np.array([True, False, False, False], dtype=bool), name="spot_to_od")
        # With L=2, D=3: step 0 takes SPOT, step 1 has no spot in greedy warmup -> takes ON_DEMAND (1 switch)
        s = ROSSScheduler(L=2.0, D=3.0, K=5.0, warmup_mode="greedy")
        res_od = run_policy(scheduler=s, trace=trace_od, dt=1.0)
        assert res_od.completed
        assert res_od.spot_time == 1.0
        assert res_od.on_demand_time == 1.0
        assert res_od.n_switches == 1, f"Expected 1 switch for SPOT->ON_DEMAND, got {res_od.n_switches}"

    def test_switching_penalty_applied_correctly(self):
        from ross.simulator import compute_switch_penalty
        # Alternating trace under ROSS (greedy warmup): takes SPOT when available, ON_DEMAND when unavailable
        trace = Trace(availability=np.array([True, False] * 20, dtype=bool), name="alternating")

        L = 20.0
        D = 40.0
        K = 5.0
        switch_penalty_pct = 0.01

        scheduler0 = ROSSScheduler(L=L, D=D, K=K, warmup_mode="greedy", seed=42)
        res0 = run_policy(scheduler=scheduler0, trace=trace, switch_penalty_pct=0.0)

        scheduler1 = ROSSScheduler(L=L, D=D, K=K, warmup_mode="greedy", seed=42)
        res1 = run_policy(scheduler=scheduler1, trace=trace, switch_penalty_pct=switch_penalty_pct)

        expected_switch_unit = compute_switch_penalty(K=K, switch_penalty_pct=switch_penalty_pct)
        expected_switch_overhead = res1.n_switches * expected_switch_unit

        assert res1.n_switches > 0
        assert abs(res1.switch_overhead_cost - expected_switch_overhead) < 1e-9
        assert abs(res1.total_cost - (res0.total_cost + expected_switch_overhead)) < 1e-9
