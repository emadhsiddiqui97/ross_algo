"""Core data structures and interfaces for the ROSS scheduler framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any


class Action(Enum):
    """Scheduling actions available at each discrete time step."""
    IDLE = 0
    SPOT = 1
    ON_DEMAND = 2

    def __str__(self) -> str:
        return self.name


class Phase(Enum):
    """Lifecycle phases for job execution under ROSS and baseline policies."""
    WARMUP = auto()
    INJECTION = auto()
    CATCHUP = auto()
    FORCED = auto()
    DONE = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class SchedulerState:
    """Tracks state variables of a scheduling run."""
    t: float = 0.0                     # Current elapsed time
    phi: float = 0.0                   # Cumulative compute completed
    phase: Phase = Phase.WARMUP        # Current algorithm phase
    cost: float = 0.0                  # Total cost accumulated
    switches: int = 0                  # Number of state switches
    last_action: Optional[Action] = None # Previous action taken
    on_demand_time: float = 0.0        # Time spent renting On-Demand
    spot_time: float = 0.0             # Time spent renting Spot
    idle_time: float = 0.0             # Time spent idling
    history: List[Dict[str, Any]] = field(default_factory=list) # Optional detailed trace history

    def copy(self) -> "SchedulerState":
        return SchedulerState(
            t=self.t,
            phi=self.phi,
            phase=self.phase,
            cost=self.cost,
            switches=self.switches,
            last_action=self.last_action,
            on_demand_time=self.on_demand_time,
            spot_time=self.spot_time,
            idle_time=self.idle_time,
        )


class BaseScheduler(ABC):
    """Abstract interface for all scheduling policies (ROSS, UniformProgress, Greedy)."""

    def __init__(self, L: float, D: float, K: float):
        """
        Args:
            L: Required computation workload (time units of compute required).
            D: Deadline (time units from t=0).
            K: Cost ratio (cost_on_demand / cost_spot, where cost_spot normalized to 1.0).
        """
        if L <= 0:
            raise ValueError(f"Workload L must be positive, got {L}")
        if D < L:
            raise ValueError(f"Deadline D ({D}) cannot be strictly less than workload L ({L})")
        if K < 1.0:
            raise ValueError(f"Cost ratio K must be >= 1.0, got {K}")

        self.L = float(L)
        self.D = float(D)
        self.K = float(K)
        self.state = SchedulerState()

    @abstractmethod
    def decide(self, spot_available: bool) -> Action:
        """Pure decision function: returns the action to take given current spot availability.
        Must NOT mutate scheduler state.
        """
        pass

    def advance(self, dt: float, action: Action) -> None:
        """Applies the decided action over duration dt and mutates state.
        
        Args:
            dt: Time duration of the step (typically 1.0 or tick length).
            action: Action executed during this step.
        """
        if self.is_done():
            self.state.phase = Phase.DONE
            return

        # Track switches (only after an initial action has been taken)
        if self.state.last_action is not None and self.state.last_action != action:
            self.state.switches += 1

        # Advance compute and time
        self.state.t += dt
        if action == Action.SPOT:
            self.state.phi = min(self.L, self.state.phi + dt)
            self.state.spot_time += dt
            self.state.cost += 1.0 * dt
        elif action == Action.ON_DEMAND:
            self.state.phi = min(self.L, self.state.phi + dt)
            self.state.on_demand_time += dt
            self.state.cost += self.K * dt
        elif action == Action.IDLE:
            self.state.idle_time += dt

        self.state.last_action = action

        if self.state.phi >= self.L - 1e-9:
            self.state.phase = Phase.DONE

    def is_done(self) -> bool:
        """Returns True if the required compute workload L has been completed."""
        return self.state.phi >= self.L - 1e-9

    def reset(self) -> None:
        """Resets the scheduler state to initial conditions."""
        self.state = SchedulerState()
