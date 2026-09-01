"""Baseline schedulers for comparison against ROSS."""

from ross.core import Action, Phase, BaseScheduler


class UniformProgressScheduler(BaseScheduler):
    """Uniform Progress baseline scheduler.
    
    Rents (spot if available, else on-demand) whenever behind the linear progress pace (L/D)*t.
    Forces on-demand unconditionally once slack reaches zero.
    """

    def decide(self, spot_available: bool) -> Action:
        if self.is_done():
            return Action.IDLE

        t = self.state.t
        phi = self.state.phi
        slack = (self.D - t) - (self.L - phi)

        # Forced completion safety check
        if slack <= 1e-9:
            return Action.ON_DEMAND

        linear_target = (self.L / self.D) * t
        if phi < linear_target - 1e-9 or t <= 1e-9:
            # Behind pace -> Rent compute (prefer spot, fallback to on-demand)
            return Action.SPOT if spot_available else Action.ON_DEMAND
        else:
            # Ahead of pace -> Only grab spot if available, otherwise idle
            return Action.SPOT if spot_available else Action.IDLE

    def advance(self, dt: float, action: Action) -> None:
        super().advance(dt, action)
        if self.is_done():
            self.state.phase = Phase.DONE
            return

        slack = (self.D - self.state.t) - (self.L - self.state.phi)
        if slack <= 1e-9:
            self.state.phase = Phase.FORCED
        else:
            self.state.phase = Phase.WARMUP

    def reset(self) -> None:
        super().reset()
        self.state.phase = Phase.WARMUP


class GreedyScheduler(BaseScheduler):
    """Greedy baseline scheduler.
    
    Takes spot instances whenever available.
    Otherwise idles until slack is exhausted, at which point it forces on-demand execution.
    """

    def decide(self, spot_available: bool) -> Action:
        if self.is_done():
            return Action.IDLE

        t = self.state.t
        phi = self.state.phi
        slack = (self.D - t) - (self.L - phi)

        # Forced completion safety check
        if slack <= 1e-9:
            return Action.ON_DEMAND

        # Prefer spot whenever available, otherwise idle
        return Action.SPOT if spot_available else Action.IDLE

    def advance(self, dt: float, action: Action) -> None:
        super().advance(dt, action)
        if self.is_done():
            self.state.phase = Phase.DONE
            return

        slack = (self.D - self.state.t) - (self.L - self.state.phi)
        if slack <= 1e-9:
            self.state.phase = Phase.FORCED
        else:
            self.state.phase = Phase.CATCHUP

    def reset(self) -> None:
        super().reset()
        self.state.phase = Phase.CATCHUP
