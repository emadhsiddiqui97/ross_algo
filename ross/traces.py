"""Trace loaders, multi-format parsers, and synthetic trace generators for spot availability."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import glob
import json
import os
import numpy as np
import pandas as pd


@dataclass
class Trace:
    """Represents a discrete time series of spot availability flags."""
    availability: np.ndarray  # 1D array of booleans where True means Spot is available
    name: str = "trace"

    def __post_init__(self):
        if not isinstance(self.availability, np.ndarray):
            self.availability = np.asarray(self.availability, dtype=bool)
        elif self.availability.dtype != bool:
            self.availability = self.availability.astype(bool)

    def __len__(self) -> int:
        return len(self.availability)

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            return self.availability[idx]
        return bool(self.availability[idx])

    @property
    def spot_availability_rate(self) -> float:
        return float(np.mean(self.availability)) if len(self.availability) > 0 else 0.0

    def slice(
        self,
        start: int = 0,
        length: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "Trace":
        """Returns a contiguous sub-slice of this trace.
        
        Args:
            start: Starting step index (0-indexed).
            length: Number of steps in the slice. If None, slices to end.
            name: Optional new trace identifier.
            
        Returns:
            New Trace instance containing the sliced window.
        """
        end = start + length if length is not None else len(self.availability)
        sliced_avail = self.availability[start:end]
        slice_name = name or f"{self.name}_s{start}_l{len(sliced_avail)}"
        return Trace(availability=sliced_avail, name=slice_name)

    def sliding_windows(
        self,
        window_size: int,
        stride: Optional[int] = None,
        max_windows: Optional[int] = None,
    ) -> List["Trace"]:
        """Generates sliding execution windows of length window_size across the trace.
        
        Args:
            window_size: Length of each sliding window (e.g. horizon D).
            stride: Step stride between consecutive window starts (defaults to window_size).
            max_windows: Optional cap on the maximum number of windows to generate.
            
        Returns:
            List of sliced Trace instances.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if len(self.availability) < window_size:
            return []

        st = stride if stride is not None and stride > 0 else window_size
        windows: List[Trace] = []

        for start in range(0, len(self.availability) - window_size + 1, st):
            sub_trace = self.slice(start=start, length=window_size, name=f"{self.name}_w{start}")
            windows.append(sub_trace)
            if max_windows is not None and len(windows) >= max_windows:
                break

        return windows


def parse_raw_availability(raw_data: Sequence[Any]) -> np.ndarray:
    """Parses raw qualitative strings, numeric values, or booleans into a boolean availability array.
    
    Mapping rules:
    - Booleans: True -> True, False -> False
    - Numbers: values > 0 (or >= 1) -> True, 0 -> False
    - Strings: 'high', 'true', '1', 'available', 'running', 'up', 'yes' -> True;
               'low', 'medium', 'false', '0', 'preempted', 'down', 'no', 'none' -> False
    """
    if len(raw_data) == 0:
        return np.zeros(0, dtype=bool)

    parsed = np.zeros(len(raw_data), dtype=bool)
    true_literals = {"1", "true", "high", "available", "running", "up", "yes"}

    for i, val in enumerate(raw_data):
        if isinstance(val, (bool, np.bool_)):
            parsed[i] = bool(val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            parsed[i] = bool(val > 0)
        elif isinstance(val, str):
            clean_str = val.strip().lower()
            parsed[i] = clean_str in true_literals
        elif isinstance(val, dict):
            # Extract common status / availability keys if dict
            status_val = val.get("status") or val.get("availability") or val.get("state") or val.get("available")
            if status_val is not None:
                if isinstance(status_val, (bool, np.bool_)):
                    parsed[i] = bool(status_val)
                elif isinstance(status_val, (int, float)):
                    parsed[i] = bool(status_val > 0)
                else:
                    parsed[i] = str(status_val).strip().lower() in true_literals
            else:
                parsed[i] = False
        else:
            parsed[i] = False

    return parsed


def load_availability_trace(
    path: Union[str, Path],
    start_step: int = 0,
    length: Optional[int] = None,
    max_steps: Optional[int] = None,
    name: Optional[str] = None,
) -> Trace:
    """Loads an availability trace from a JSON or CSV file.
    
    Supports SkyPilot 10-minute ping logs, Spotlake qualitative traces ('high' -> 1, others -> 0),
    and multi-node cluster availability traces.
    
    Args:
        path: Path to the JSON or CSV file.
        start_step: Offset step index to slice from.
        length: Window length to slice. If specified, overrides max_steps.
        max_steps: Optional cap on the number of steps loaded from start_step.
        name: Optional custom trace identifier.
        
    Returns:
        Trace instance with parsed boolean availability.
    """
    path_str = str(path)
    trace_name = name or os.path.basename(path_str).split(".")[0]

    if path_str.endswith(".json"):
        with open(path_str, "r", encoding="utf-8") as fp:
            content = json.load(fp)

        if isinstance(content, dict):
            raw = content.get("data") or content.get("availability") or content.get("values") or []
        elif isinstance(content, list):
            raw = content
        else:
            raw = []

        avail_bools = parse_raw_availability(raw)

    elif path_str.endswith(".csv"):
        df = pd.read_csv(path_str)
        # Identify availability column
        col_candidates = [
            "spot_available", "availability", "available", "status",
            "state", "State", "is_available", "spot_avail"
        ]
        target_col = None
        for col in col_candidates:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[0]

        avail_bools = parse_raw_availability(df[target_col].tolist())

    else:
        raise ValueError(f"Unsupported file format for availability trace: {path_str}")

    # Apply slicing / windowing
    n_total = len(avail_bools)
    start = max(0, min(start_step, n_total))
    if length is not None:
        avail_bools = avail_bools[start : start + length]
    elif max_steps is not None:
        avail_bools = avail_bools[start : start + max_steps]
    else:
        avail_bools = avail_bools[start:]

    return Trace(availability=avail_bools, name=trace_name)


def load_preemption_trace(
    path: Union[str, Path],
    start_step: int = 0,
    length: Optional[int] = None,
    max_steps: Optional[int] = None,
    name: Optional[str] = None,
) -> Trace:
    """Loads a preemption lifetime trace from a JSON or CSV file.
    
    Parses lifetime logs into continuous binary time-series arrays
    (1 = available / running, 0 = preempted / unavailable).
    
    Args:
        path: Path to the JSON or CSV file.
        start_step: Offset step index to slice from.
        length: Window length to slice. If specified, overrides max_steps.
        max_steps: Optional cap on the number of steps loaded from start_step.
        name: Optional custom trace identifier.
        
    Returns:
        Trace instance with parsed boolean availability.
    """
    path_str = str(path)
    trace_name = name or os.path.basename(path_str).split(".")[0]

    if path_str.endswith(".json"):
        with open(path_str, "r", encoding="utf-8") as fp:
            content = json.load(fp)

        if isinstance(content, dict):
            raw = content.get("data") or content.get("availability") or content.get("values") or []
        elif isinstance(content, list):
            raw = content
        else:
            raw = []

        avail_bools = parse_raw_availability(raw)

    elif path_str.endswith(".csv"):
        df = pd.read_csv(path_str)
        col_candidates = [
            "spot_available", "availability", "available", "status",
            "state", "State", "is_available", "preempted", "running"
        ]
        target_col = None
        for col in col_candidates:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[0]

        if target_col.lower() == "preempted":
            # 1 means preempted (unavailable) -> invert
            raw = df[target_col].tolist()
            preempted = parse_raw_availability(raw)
            avail_bools = ~preempted
        else:
            avail_bools = parse_raw_availability(df[target_col].tolist())

    else:
        raise ValueError(f"Unsupported file format for preemption trace: {path_str}")

    # Apply slicing / windowing
    n_total = len(avail_bools)
    start = max(0, min(start_step, n_total))
    if length is not None:
        avail_bools = avail_bools[start : start + length]
    elif max_steps is not None:
        avail_bools = avail_bools[start : start + max_steps]
    else:
        avail_bools = avail_bools[start:]

    return Trace(availability=avail_bools, name=trace_name)


def load_trace(
    path: Union[str, Path],
    start_step: int = 0,
    length: Optional[int] = None,
    max_steps: Optional[int] = None,
    name: Optional[str] = None,
    trace_type: Optional[str] = None,
) -> Trace:
    """Universal trace loader that auto-detects availability vs preemption traces."""
    path_str = str(path)
    if trace_type == "preemption" or "preemption" in path_str.lower():
        return load_preemption_trace(
            path=path_str,
            start_step=start_step,
            length=length,
            max_steps=max_steps,
            name=name,
        )
    else:
        return load_availability_trace(
            path=path_str,
            start_step=start_step,
            length=length,
            max_steps=max_steps,
            name=name,
        )


def discover_trace_files(
    data_dir: Union[str, Path] = "data",
    category: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Auto-discovers trace files in the availability and preemption datasets.
    
    Args:
        data_dir: Base root data directory (default 'data').
        category: Optional filter: 'availability', 'preemption', or None for all.
        
    Returns:
        Dict mapping category name to list of discovered file paths.
    """
    base_path = Path(data_dir)
    discovered: Dict[str, List[str]] = {"availability": [], "preemption": []}

    categories = [category] if category in ("availability", "preemption") else ["availability", "preemption"]

    for cat in categories:
        cat_dir = base_path / cat
        if cat_dir.exists():
            for root, _, files in os.walk(cat_dir):
                for f in sorted(files):
                    if f.endswith((".json", ".csv")):
                        full_path = os.path.join(root, f)
                        discovered[cat].append(full_path)

    return discovered


def load_all_availability_traces(
    data_dir: Union[str, Path] = "data",
    max_steps: Optional[int] = None,
) -> List[Trace]:
    """Loads all discovered availability traces from data/availability/."""
    files = discover_trace_files(data_dir=data_dir, category="availability")["availability"]
    return [load_availability_trace(f, max_steps=max_steps) for f in files]


def load_all_preemption_traces(
    data_dir: Union[str, Path] = "data",
    max_steps: Optional[int] = None,
) -> List[Trace]:
    """Loads all discovered preemption traces from data/preemption/."""
    files = discover_trace_files(data_dir=data_dir, category="preemption")["preemption"]
    return [load_preemption_trace(f, max_steps=max_steps) for f in files]


def load_all_traces(
    data_dir: Union[str, Path] = "data",
    max_steps: Optional[int] = None,
) -> List[Trace]:
    """Loads all discovered traces across both availability and preemption categories."""
    discovered = discover_trace_files(data_dir=data_dir)
    all_traces: List[Trace] = []
    for f in discovered.get("availability", []):
        all_traces.append(load_availability_trace(f, max_steps=max_steps))
    for f in discovered.get("preemption", []):
        all_traces.append(load_preemption_trace(f, max_steps=max_steps))
    return all_traces


def synthetic_trace(
    n_steps: int,
    avg_availability: float = 0.6,
    mean_run_len: float = 10.0,
    seed: Optional[int] = None,
    name: str = "synthetic",
) -> Trace:
    """Generates a synthetic boolean availability trace using a 2-state Markov chain.
    
    Args:
        n_steps: Number of discrete time steps.
        avg_availability: Target fraction of time spot is available (in (0, 1)).
        mean_run_len: Average duration (steps) of consecutive availability.
        seed: Random seed for reproducibility.
        name: Identifier name for the trace.
        
    Returns:
        Trace instance.
    """
    if not (0.0 < avg_availability < 1.0):
        raise ValueError(f"avg_availability must be in (0, 1), got {avg_availability}")
    if mean_run_len < 1.0:
        raise ValueError(f"mean_run_len must be >= 1.0, got {mean_run_len}")

    rng = np.random.default_rng(seed)

    # State 1: Available, State 0: Unavailable
    p10 = 1.0 / mean_run_len
    p11 = 1.0 - p10

    # From stationary distribution: avg_avail * p10 = (1 - avg_avail) * p01
    p01 = (avg_availability / (1.0 - avg_availability)) * p10
    p01 = min(max(p01, 1e-6), 1.0 - 1e-6)

    states = np.zeros(n_steps, dtype=bool)
    current_state = rng.random() < avg_availability
    states[0] = current_state

    for t in range(1, n_steps):
        if current_state:
            current_state = rng.random() < p11
        else:
            current_state = rng.random() < p01
        states[t] = current_state

    return Trace(availability=states, name=name)


def load_trace_csv(
    path: str,
    availability_col: str = "spot_available",
    max_steps: Optional[int] = None,
    name: Optional[str] = None,
) -> Trace:
    """Loads spot availability from a CSV file (backwards compatibility)."""
    return load_availability_trace(
        path=path,
        max_steps=max_steps,
        name=name,
    )
