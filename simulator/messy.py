"""Corrupts clean readings the way field telemetry from a dusty, vibrating mine site actually arrives.

Every corruption here is something the backend must cope with:
  - station outages (comms loss for minutes to hours)  -> gaps
  - random dropped readings / individual NaN fields    -> missing values
  - stuck sensors (same value repeated for hours)      -> plausible-looking but dead data
  - spikes and sentinel garbage (-999, 9999)           -> outliers
  - clock drift per station + jitter                   -> timestamps that don't line up
  - duplicates and locally out-of-order rows           -> retransmits / network reordering
"""
import numpy as np
import pandas as pd

from .config import SAMPLE_INTERVAL_S, STATIONS

MIN = 60
CORRUPTIBLE = ["bearing_temp_c", "vibration_rms_mms", "belt_offset_mm", "belt_tension_kn",
               "belt_speed_ms", "load_tph", "ambient_temp_c", "dust_level", "motor_current_a"]


class MessyLayer:
    def __init__(self, rng: np.random.Generator, t0: int, intensity: float = 1.0):
        self.rng = rng
        self.t0 = t0
        k = intensity
        steps_per_day = 86400 / SAMPLE_INTERVAL_S
        self.p_outage = k / (15 * steps_per_day)          # ~one outage per station every 15 days
        self.p_stuck = k / (40 * steps_per_day)           # per station per signal
        self.p_drop, self.p_nan, self.p_spike, self.p_sentinel = 0.003 * k, 0.005 * k, 0.001 * k, 0.0003 * k
        self.p_dup, self.p_swap = 0.005 * k, 0.005 * k
        self.outage_until = {s.id: 0 for s in STATIONS}
        self.stuck: dict[tuple[str, str], tuple[int, float]] = {}
        # seconds of drift per day; one station has a badly drifting clock
        self.drift = {s.id: rng.normal(0, 2.0) for s in STATIONS}
        self.drift["ST-13"] = 40.0

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        df = df.copy()
        keep = np.ones(len(df), dtype=bool)

        for sid, idx in df.groupby("station_id").indices.items():
            ts = df["ts"].to_numpy()[idx]

            # stuck sensors: continue an episode carried over from the previous block, then start new ones
            for sig in CORRUPTIBLE:
                col = df.columns.get_loc(sig)
                vals = df.iloc[idx, col].to_numpy(copy=True)
                until, value = self.stuck.get((sid, sig), (0, None))
                if value is not None:
                    vals[ts < until] = value
                for i in np.flatnonzero(rng.random(len(idx)) < self.p_stuck):
                    if ts[i] >= until:
                        until, value = ts[i] + int(rng.uniform(1, 6) * 3600), vals[i]
                        vals[(ts >= ts[i]) & (ts < until)] = value
                df.iloc[idx, col] = vals
                if until > ts[-1]:
                    self.stuck[(sid, sig)] = (until, value)
                else:
                    self.stuck.pop((sid, sig), None)

            # comms outages: whole rows missing for 10-180 minutes
            until = self.outage_until[sid]
            mask = ts < until
            for i in np.flatnonzero(rng.random(len(idx)) < self.p_outage):
                end = ts[i] + int(rng.uniform(10, 180)) * MIN
                mask |= (ts >= ts[i]) & (ts < end)
                until = max(until, end)
            keep[idx[mask]] = False
            self.outage_until[sid] = until

        # field-level corruption
        values = df[CORRUPTIBLE].to_numpy(copy=True)
        shape = values.shape
        values[rng.random(shape) < self.p_nan] = np.nan
        spikes = rng.random(shape) < self.p_spike
        values[spikes] *= rng.uniform(3, 8, spikes.sum()) * rng.choice([-1, 1], spikes.sum())
        values[rng.random(shape) < self.p_sentinel] = rng.choice([-999.0, 9999.0])
        df[CORRUPTIBLE] = values

        keep &= rng.random(len(df)) >= self.p_drop
        df = df[keep]

        # clock drift + jitter: the timestamp is what the station *thinks* the time is
        drift = df["station_id"].map(self.drift).to_numpy()
        df["ts"] = (df["ts"] + drift * (df["ts"] - self.t0) / 86400 + rng.normal(0, 1.0, len(df))).round().astype(int)

        # retransmitted duplicates and network reordering
        dups = df[rng.random(len(df)) < self.p_dup]
        df = pd.concat([df, dups]).sort_index(kind="stable").reset_index(drop=True)
        order = np.arange(len(df))
        for i in np.flatnonzero(rng.random(len(df) - 1) < self.p_swap):
            order[i], order[i + 1] = order[i + 1], order[i]
        return df.iloc[order].reset_index(drop=True)
