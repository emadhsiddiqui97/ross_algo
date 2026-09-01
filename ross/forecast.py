"""Prediction-augmented ROSS scheduler and forecast signals (Part B)."""

import math
from typing import Optional, Sequence
import numpy as np

from ross.core import Action, Phase
from ross.scheduler_ross import ROSSScheduler


class SpotForecaster:
    """Lightweight predictor of near-term spot availability."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: list[bool] = []

    def update(self, spot_available: bool) -> None:
        self.history.append(spot_available)

    def predict_unavailability_density(self, horizon: int) -> np.ndarray:
        """Predicts probability density of spot unavailability over the future horizon.
        
        Returns a normalized 1D probability distribution over the horizon bins.
        """
        if len(self.history) < 2:
            return np.ones(horizon) / max(1, horizon)
        
        recent = self.history[-self.window_size:]
        unavail_rate = 1.0 - float(np.mean(recent))
        
        # Default smooth distribution weighted by recent unavailability
        weights = np.ones(horizon) * max(0.01, unavail_rate)
        total = np.sum(weights)
        return weights / total if total > 0 else np.ones(horizon) / max(1, horizon)


class PredictionAugmentedROSS(ROSSScheduler):
    """Prediction-augmented ROSS scheduling algorithm (Part B).
    
    Blends uniform random injection window sampling with forecast-weighted sampling.
    When lambda_ = 0.0, behavior is strictly identical to vanilla ROSS.
    """

    def __init__(
        self,
        L: float,
        D: float,
        K: float,
        lambda_: float = 0.0,
        warmup_mode: str = "greedy",
        seed: Optional[int] = None,
    ):
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError(f"lambda_ must be in [0.0, 1.0], got {lambda_}")
        
        self.lambda_ = float(lambda_)
        self.forecaster = SpotForecaster()
        super().__init__(L=L, D=D, K=K, warmup_mode=warmup_mode, seed=seed)

    def _trigger_injection_setup(self) -> None:
        """Sets up the randomized injection window with forecast weighting."""
        self.state.phase = Phase.INJECTION
        self.xi1 = self.state.t
        remaining_work = max(0.0, self.L - self.state.phi)
        sqrt_k = math.sqrt(self.K)
        self.delta = remaining_work / (1.0 + sqrt_k)

        window_len = remaining_work
        self.injection_window_end = self.xi1 + window_len
        max_offset = max(0.0, window_len - self.delta)

        if max_offset <= 1e-9:
            self.sigma = self.xi1
            return

        if self.lambda_ <= 1e-9:
            # Strictly vanilla ROSS: Uniform random sampling
            offset = float(self.rng.uniform(0.0, max_offset))
        else:
            # Discretize possible start offsets into bins
            num_bins = max(10, int(max_offset))
            bin_offsets = np.linspace(0.0, max_offset, num_bins)
            
            # Uniform prior
            p_uniform = np.ones(num_bins) / num_bins
            
            # Forecast distribution
            p_forecast = self.forecaster.predict_unavailability_density(num_bins)
            
            # Mixture sampling
            p_mixture = (1.0 - self.lambda_) * p_uniform + self.lambda_ * p_forecast
            p_mixture = p_mixture / np.sum(p_mixture)
            
            chosen_bin = self.rng.choice(num_bins, p=p_mixture)
            offset = float(bin_offsets[chosen_bin])

        self.sigma = self.xi1 + offset

    def advance(self, dt: float, action: Action) -> None:
        super().advance(dt, action)
