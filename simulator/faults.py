"""Failure modes and the scheduler that decides when and where they happen.

A fault is not an instant event: it has a *degradation phase* (start -> failure_at)
during which the sensors drift, and then a failure. That lead time is exactly what
the prediction model is supposed to learn to exploit.

Severity of a fault at time t is p = (t - start) / (failure_at - start), in [0, 1].
Each mode maps p to deltas on specific signals; most effects grow non-linearly
(slow onset, fast runaway), which is how real bearings and splices behave.
"""
from dataclasses import dataclass

import numpy as np

from .config import STATIONS, Station

HOUR = 3600
DAY = 24 * HOUR

# mode -> (degradation lead time range in hours, applicable to drive stations only?)
FAULT_MODES = {
    "bearing_failure": ((48, 144), False),
    "belt_misalignment": ((24, 72), False),
    "splice_failure": ((48, 120), False),
    "belt_slip": ((8, 24), True),
    "motor_overload": ((12, 48), True),
}


@dataclass
class Fault:
    id: int
    station_id: str
    mode: str
    start: int          # epoch seconds, degradation begins
    failure_at: int     # epoch seconds, functional failure (unplanned stop)
    direction: float    # +1/-1, used by misalignment to pick a side

    def progress(self, t: np.ndarray) -> np.ndarray:
        """Severity in [0, 1] while degrading, 0 outside the window (component replaced at failure)."""
        p = (t - self.start) / (self.failure_at - self.start)
        return np.where((t >= self.start) & (t < self.failure_at), p, 0.0)


def apply_fault(fault: Fault, t: np.ndarray, sig: dict[str, np.ndarray], running: np.ndarray,
                rng: np.random.Generator) -> None:
    """Add this fault's effect onto the station's signal arrays, in place."""
    p = fault.progress(t)
    if not p.any():
        return
    run = running.astype(float)
    n = len(t)

    if fault.mode == "bearing_failure":
        # friction heat persists a little even when stopped; vibration only when turning
        sig["bearing_temp_c"] += 45 * p ** 2.5 * (0.3 + 0.7 * run)
        sig["vibration_rms_mms"] += 7 * p ** 2 * run
    elif fault.mode == "belt_misalignment":
        sig["belt_offset_mm"] += fault.direction * 70 * p ** 1.5
        sig["vibration_rms_mms"] += 1.5 * p * run
    elif fault.mode == "splice_failure":
        # a weakening splice shows up as tension pulses and impact vibration each time it passes
        sig["belt_tension_kn"] += 25 * p ** 2 * rng.standard_normal(n) * run
        sig["vibration_rms_mms"] += 3 * p ** 2 * np.abs(rng.standard_normal(n)) * run
    elif fault.mode == "belt_slip":
        sig["belt_speed_ms"] -= 0.6 * p ** 3 * run
        sig["belt_tension_kn"] -= 18 * p ** 2 * run
        sig["motor_current_a"] += 60 * p ** 2 * rng.standard_normal(n) * run
    elif fault.mode == "motor_overload":
        sig["motor_current_a"] += 280 * p ** 2 * run
        sig["bearing_temp_c"] += 18 * p ** 2 * run


class FaultScheduler:
    """Pre-plans faults per station as a renewal process, extended lazily as the simulation advances."""

    def __init__(self, rng: np.random.Generator, start: int, mean_days_between: float = 18.0):
        self.rng = rng
        self.mean_days_between = mean_days_between
        self.faults: list[Fault] = []
        self._next_free = {s.id: start + int(rng.uniform(0, 10) * DAY) for s in STATIONS}
        self._next_id = 1

    def _rate_factor(self, st: Station) -> float:
        # harsher terrain fails more often
        return {"flat": 1.0, "incline": 1.6, "curve": 1.6}[st.terrain]

    def _pick_mode(self, st: Station) -> str:
        modes = [m for m, (_, drive_only) in FAULT_MODES.items() if st.has_drive or not drive_only]
        weights = np.array([{"bearing_failure": 0.4, "belt_misalignment": 0.25, "splice_failure": 0.15,
                             "belt_slip": 0.15, "motor_overload": 0.15}[m] for m in modes])
        if st.terrain == "curve":
            weights[modes.index("belt_misalignment")] *= 2.5
        return self.rng.choice(modes, p=weights / weights.sum())

    def add(self, station_id: str, mode: str, start: int, lead_hours: float) -> Fault:
        f = Fault(self._next_id, station_id, mode, int(start), int(start + lead_hours * HOUR),
                  float(self.rng.choice([-1.0, 1.0])))
        self._next_id += 1
        self.faults.append(f)
        self._next_free[station_id] = max(self._next_free[station_id], f.failure_at + DAY)
        return f

    def ensure_until(self, t_end: int) -> None:
        for st in STATIONS:
            while self._next_free[st.id] < t_end:
                gap = self.rng.exponential(self.mean_days_between / self._rate_factor(st)) * DAY
                mode = self._pick_mode(st)
                lo, hi = FAULT_MODES[mode][0]
                self.add(st.id, mode, self._next_free[st.id] + gap, self.rng.uniform(lo, hi))

    def active_for(self, station_id: str, t0: int, t1: int) -> list[Fault]:
        return [f for f in self.faults if f.station_id == station_id and f.start < t1 and f.failure_at > t0]
