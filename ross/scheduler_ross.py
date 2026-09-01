"""Implementation of the ROSS scheduling algorithm (Bhuyan et al., arXiv:2601.14612)."""

import math
from typing import Optional, Literal
import numpy as np

from ross.core import Action, Phase, BaseScheduler, SchedulerState


class ROSSScheduler(BaseScheduler):
    """Randomized Online Spot Scheduler (ROSS).
    
    Supports both 'greedy' and 'uniform' warmup strategies as defined in Algorithm 1.
    """

    def __init__(
        self,
        L: float,
        D: float,
        K: float,
        warmup_mode: Literal["greedy", "uniform"] = "greedy",
        seed: Optional[int] = None,
    ):
        """
        Args:
            L: Total workload compute requirement.
            D: Deadline time.
            K: Cost ratio (on-demand cost / spot cost).
            warmup_mode: 'greedy' (spot else on-demand) or 'uniform' (rent only when behind pace).
            seed: Optional random seed for reproducible injection window sampling.
        """
        super().__init__(L=L, D=D, K=K)
        if warmup_mode not in ("greedy", "uniform"):
            raise ValueError(f"warmup_mode must be 'greedy' or 'uniform', got {warmup_mode}")

        self.warmup_mode = warmup_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Injection phase parameters
        self.xi1: Optional[float] = None
        self.delta: Optional[float] = None
        self.sigma: Optional[float] = None
        self.injection_window_end: Optional[float] = None

        self._check_initial_phase()

    @property
    def warmup_threshold(self) -> float:
        """The critical ratio (1 + 2*sqrt(K)) / (1 + sqrt(K))."""
        sqrt_k = math.sqrt(self.K)
        return (1.0 + 2.0 * sqrt_k) / (1.0 + sqrt_k)

    def _check_initial_phase(self) -> None:
        """Determines initial phase at t=0."""
        slack = (self.D - self.state.t) - (self.L - self.state.phi)
        if slack <= 1e-9:
            self.state.phase = Phase.FORCED
            return

        remaining_work = self.L - self.state.phi
        if remaining_work <= 1e-9:
            self.state.phase = Phase.DONE
            return

        ratio = (self.D - self.state.t) / remaining_work
        if ratio > self.warmup_threshold:
            self.state.phase = Phase.WARMUP
        else:
            self._trigger_injection_setup()

    def _trigger_injection_setup(self) -> None:
        """Sets up the randomized injection window (Sub-Routine 1)."""
        self.state.phase = Phase.INJECTION
        self.xi1 = self.state.t
        remaining_work = max(0.0, self.L - self.state.phi)
        sqrt_k = math.sqrt(self.K)
        self.delta = remaining_work / (1.0 + sqrt_k)
        
        window_len = remaining_work
        self.injection_window_end = self.xi1 + window_len

        # Sample uniform start offset inside [xi1, xi1 + window_len - delta]
        max_offset = max(0.0, window_len - self.delta)
        offset = float(self.rng.uniform(0.0, max_offset)) if max_offset > 1e-9 else 0.0
        self.sigma = self.xi1 + offset

    def decide(self, spot_available: bool) -> Action:
        """Pure decision function returning the next action without mutating state."""
        if self.is_done():
            return Action.IDLE

        t = self.state.t
        phi = self.state.phi
        slack = (self.D - t) - (self.L - phi)

        # Invariant safety: Forced phase when slack is exhausted
        if slack <= 1e-9:
            return Action.ON_DEMAND

        remaining_work = self.L - phi
        if remaining_work <= 1e-9:
            return Action.IDLE

        # Warmup Phase
        if self.state.phase == Phase.WARMUP:
            ratio = (self.D - t) / remaining_work
            if ratio > self.warmup_threshold:
                if self.warmup_mode == "greedy":
                    return Action.SPOT if spot_available else Action.ON_DEMAND
                else:  # 'uniform'
                    if spot_available:
                        return Action.SPOT
                    elif phi < (self.L / self.D) * t - 1e-9:
                        return Action.ON_DEMAND
                    else:
                        return Action.IDLE

        # Injection Phase
        if self.state.phase == Phase.INJECTION or (self.state.phase == Phase.WARMUP and (self.D - t) / remaining_work <= self.warmup_threshold):
            # If not yet set up, preview injection logic
            sigma = self.sigma if self.sigma is not None else t
            delta = self.delta if self.delta is not None else (remaining_work / (1.0 + math.sqrt(self.K)))
            window_end = self.injection_window_end if self.injection_window_end is not None else (t + remaining_work)

            if t < window_end:
                if sigma <= t < (sigma + delta):
                    return Action.ON_DEMAND
                else:
                    return Action.SPOT if spot_available else Action.IDLE

        # Catchup Phase
        return Action.SPOT if spot_available else Action.IDLE

    def advance(self, dt: float, action: Action) -> None:
        """Advances state and handles phase transitions."""
        super().advance(dt, action)

        if self.is_done():
            self.state.phase = Phase.DONE
            return

        t = self.state.t
        phi = self.state.phi
        slack = (self.D - t) - (self.L - phi)

        # Check for forced phase
        if slack <= 1e-9:
            self.state.phase = Phase.FORCED
            return

        remaining_work = self.L - phi
        if self.state.phase == Phase.WARMUP:
            ratio = (self.D - t) / remaining_work
            if ratio <= self.warmup_threshold:
                self._trigger_injection_setup()

        elif self.state.phase == Phase.INJECTION:
            if self.injection_window_end is not None and t >= self.injection_window_end - 1e-9:
                self.state.phase = Phase.CATCHUP

    def reset(self) -> None:
        """Resets the scheduler state and RNG."""
        self.state = SchedulerState()
        self.rng = np.random.default_rng(self.seed)
        self.xi1 = None
        self.delta = None
        self.sigma = None
        self.injection_window_end = None
        self._check_initial_phase()
