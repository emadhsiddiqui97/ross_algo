"""Prediction-augmented ROSS: rolling unavailability forecast + injection-offset bias."""

import math
from typing import Optional, Sequence

import numpy as np

from ross.scheduler_ross import ROSSScheduler


def forecast_unavailability_rates(
    history: Sequence[bool],
    n_starts: int,
    delta: float,
    window_len: float,
    lookback: Optional[int] = None,
) -> np.ndarray:
    """Empirical unavailability rate for each candidate injection start.

    Uses the last ``lookback`` ticks (default: ``window_len``) as a template for
    the upcoming injection interval. Candidate ``i`` is scored by the fraction
    of unavailable ticks in the aligned sub-window of length ``delta``.

    Returns an array of shape ``(n_starts,)`` with values in ``[0, 1]``.
    """
    n_starts = max(1, int(n_starts))
    w = max(1, int(round(window_len)))
    d = max(1, int(round(delta)))
    d = min(d, w)
    lb = int(lookback) if lookback is not None else w
    lb = max(d, lb)

    hist = np.asarray(list(history), dtype=bool)
    if hist.size == 0:
        return np.full(n_starts, 0.5)

    template = hist[-lb:]
    if template.size < w:
        template = np.concatenate(
            [np.ones(w - template.size, dtype=bool), template]
        )
    else:
        template = template[-w:]

    unavail = ~template
    max_start = max(0, w - d)
    rates = np.empty(n_starts, dtype=float)
    for i in range(n_starts):
        start = int(round(i * max_start / max(n_starts - 1, 1)))
        rates[i] = float(np.mean(unavail[start : start + d]))
    return rates


class PredictionAugmentedROSS(ROSSScheduler):
    """ROSS with mixture sampling of the injection start offset.

    ``lambda_ = 0`` is identical to vanilla ROSS (same RNG draws).
    ``lambda_ = 1`` samples start offsets weighted by forecast unavailability.
    Intermediate values mix the two with a coin flip of probability ``lambda_``.
    """

    def __init__(
        self,
        L: float,
        D: float,
        K: float,
        lambda_: float = 0.0,
        warmup_mode: str = "greedy",
        seed: Optional[int] = None,
        history: Optional[Sequence[bool]] = None,
        lookback: Optional[int] = None,
    ):
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError(f"lambda_ must be in [0.0, 1.0], got {lambda_}")
        self.lambda_ = float(lambda_)
        self.lookback = lookback
        self._fit_history = list(history) if history is not None else []
        self.history: list[bool] = list(self._fit_history)
        self._obs: Optional[bool] = None
        super().__init__(L=L, D=D, K=K, warmup_mode=warmup_mode, seed=seed)

    def decide(self, spot_available: bool):
        self._obs = bool(spot_available)
        return super().decide(spot_available)

    def advance(self, dt: float, action) -> None:
        if self._obs is not None:
            self.history.append(self._obs)
            self._obs = None
        super().advance(dt, action)

    def reset(self) -> None:
        self.history = list(self._fit_history)
        self._obs = None
        super().reset()

    def _trigger_injection_setup(self) -> None:
        super()._trigger_injection_setup()
        if self.lambda_ == 0.0:
            return

        window_len = (
            (self.injection_window_end - self.xi1)
            if self.injection_window_end is not None and self.xi1 is not None
            else max(0.0, self.L - self.state.phi)
        )
        max_offset = max(0.0, window_len - (self.delta or 0.0))
        if max_offset <= 1e-9:
            return

        if self.lambda_ < 1.0 and float(self.rng.random()) >= self.lambda_:
            self.sigma = self.xi1 + float(self.rng.uniform(0.0, max_offset))
            return

        n_starts = max(2, int(math.floor(max_offset)) + 1)
        rates = forecast_unavailability_rates(
            self.history,
            n_starts=n_starts,
            delta=self.delta or 1.0,
            window_len=window_len,
            lookback=self.lookback,
        )
        weights = rates.copy()
        if float(np.sum(weights)) <= 1e-12:
            weights = np.ones(n_starts)
        weights = weights / np.sum(weights)
        offsets = np.linspace(0.0, max_offset, n_starts)
        idx = int(self.rng.choice(n_starts, p=weights))
        self.sigma = self.xi1 + float(offsets[idx])


class SpotForecaster:
    """Thin wrapper around ``forecast_unavailability_rates`` for package exports."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: list[bool] = []

    def update(self, spot_available: bool) -> None:
        self.history.append(bool(spot_available))

    def fit(self, history: Sequence[bool]) -> None:
        self.history = list(history)

    def predict_unavailability_density(self, horizon: int) -> np.ndarray:
        rates = forecast_unavailability_rates(
            self.history,
            n_starts=max(1, horizon),
            delta=1.0,
            window_len=float(self.window_size),
            lookback=self.window_size,
        )
        total = float(np.sum(rates))
        if total <= 1e-12:
            return np.ones(len(rates)) / len(rates)
        return rates / total
