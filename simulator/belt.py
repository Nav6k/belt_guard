"""Physics-flavoured generator for *clean* sensor readings.

Not a real conveyor model: it is just enough structure that the signals behave
plausibly and, importantly, that faults are separable from normal variation
(load swings, day/night ambient, duty cycle, terrain, per-sensor calibration offsets).

Generation is vectorised per block of time so 30+ days of history takes seconds,
and stateful between blocks so streaming continues seamlessly from a backfill.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import (BELT_SPEED_MS, MAX_LOAD_TPH, RUN_HOURS, SAMPLE_INTERVAL_S, SIGNALS, STATIONS)
from .faults import FaultScheduler, apply_fault


def _ar1(rng, n, x0, mean, pull, noise, lo, hi):
    """Mean-reverting random walk, used for plant load and dust."""
    out = np.empty(n)
    x = x0
    eps = rng.normal(0, noise, n)
    for i in range(n):
        x = min(hi, max(lo, x + pull * (mean - x) + eps[i]))
        out[i] = x
    return out


class BeltSimulator:
    def __init__(self, start: pd.Timestamp, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.t = int(pd.Timestamp(start).timestamp())
        self.scheduler = FaultScheduler(self.rng, self.t)
        self._load = 0.7
        self._dust = 0.3
        self._thermal_zi = None          # filter state for bearing thermal lag
        # each physical sensor has its own calibration offset: the model must not assume all stations read alike
        self.calib = {s.id: {"bearing_temp_c": self.rng.normal(0, 2.0),
                             "vibration_rms_mms": self.rng.normal(0, 0.15),
                             "belt_offset_mm": self.rng.normal(0, 2.0),
                             "belt_tension_kn": self.rng.normal(0, 1.5)} for s in STATIONS}

    def generate(self, n_steps: int) -> pd.DataFrame:
        """Advance the simulation by n_steps samples and return clean readings (long format, one row per station per step)."""
        rng = self.rng
        times = self.t + np.arange(n_steps) * SAMPLE_INTERVAL_S
        self.t = int(times[-1] + SAMPLE_INTERVAL_S)
        self.scheduler.ensure_until(self.t + 7 * 86400)

        hour = (times % 86400) / 3600
        running = (hour >= RUN_HOURS[0]) & (hour < RUN_HOURS[1])
        run = running.astype(float)

        ambient = 27 + 9 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.3, n_steps)
        load_frac = _ar1(rng, n_steps, self._load, 0.75, 0.02, 0.02, 0.3, 1.0)
        self._load = load_frac[-1]
        lf = load_frac * run
        dust = _ar1(rng, n_steps, self._dust, 0.25, 0.01, 0.03, 0.0, 0.7)
        dust = np.clip(dust + 0.25 * run, 0, 1)
        self._dust = dust[-1] - 0.25 * run[-1]

        # bearing heat rise follows load with ~30 min thermal lag (first-order low-pass)
        a = SAMPLE_INTERVAL_S / 1800
        target = (12 + 14 * lf) * run
        if self._thermal_zi is None:
            self._thermal_zi = np.array([target[0] * (1 - a)])
        heat, self._thermal_zi = lfilter([a], [1, -(1 - a)], target, zi=self._thermal_zi)

        frames = []
        for st in STATIONS:
            c = self.calib[st.id]
            incline = st.terrain == "incline"
            curve = st.terrain == "curve"
            sig = {
                "bearing_temp_c": ambient + heat + (3 if incline else 0) + c["bearing_temp_c"]
                                  + rng.normal(0, 0.4, n_steps),
                "vibration_rms_mms": (1.2 + 1.0 * lf + (0.6 if incline else 0) + (0.4 if curve else 0)) * run
                                     + 0.15 + c["vibration_rms_mms"] + np.abs(rng.normal(0, 0.12, n_steps)),
                "belt_offset_mm": (6 if curve else 0) + c["belt_offset_mm"]
                                  + rng.normal(0, 1.5, n_steps) * (1 + dust),
                "belt_tension_kn": np.where(running, 35 + 25 * lf + (10 if incline else 0) + (15 if st.has_drive else 0), 20)
                                   + c["belt_tension_kn"] + rng.normal(0, 0.8, n_steps),
                "belt_speed_ms": BELT_SPEED_MS * run + rng.normal(0, 0.02, n_steps) * run,
                "load_tph": lf * MAX_LOAD_TPH + rng.normal(0, 40, n_steps) * run,
                "ambient_temp_c": ambient + rng.normal(0, 0.2, n_steps),
                "dust_level": dust,
                "motor_current_a": (150 + 330 * lf) * run + rng.normal(0, 5, n_steps) * run
                                   if st.has_drive else np.full(n_steps, np.nan),
            }
            for f in self.scheduler.active_for(st.id, int(times[0]), self.t):
                apply_fault(f, times, sig, running, rng)

            df = pd.DataFrame({k: sig[k] for k in SIGNALS})
            df.insert(0, "station_id", st.id)
            df.insert(0, "ts", times)
            frames.append(df)

        out = pd.concat(frames, ignore_index=True)
        out["load_tph"] = out["load_tph"].clip(lower=0)
        out["belt_speed_ms"] = out["belt_speed_ms"].clip(lower=0)
        return out.sort_values(["ts", "station_id"], ignore_index=True)

    def ground_truth(self) -> pd.DataFrame:
        """Fault schedule up to the current sim time. Never sent to the backend: it's the answer key."""
        rows = [vars(f) for f in self.scheduler.faults if f.start < self.t]
        return pd.DataFrame(rows, columns=["id", "station_id", "mode", "start", "failure_at", "direction"])
