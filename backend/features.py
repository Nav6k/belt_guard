"""Feature engineering on cleaned 1-minute data -> one row per station per 10 minutes.

Key idea: *fleet-relative* signals. All stations share the same ambient temperature, load,
duty cycle (running / stopped) and belt speed, so subtracting the fleet median at each
timestamp removes most normal variation. Subtracting each station's own rolling 7-day
median then removes its calibration offset and terrain bias. What is left is
"how different is this station from its usual self, relative to its neighbours right now",
which is what a developing fault looks like.

Shared by ml/train.py and the live backend.
"""
import numpy as np
import pandas as pd

from simulator.config import STATIONS

BUCKET_S = 600                         # 10-minute feature resolution
PER_HOUR = 3600 // BUCKET_S
# signal -> short prefix used in feature names
FLEET_SIGNALS = {"bearing_temp_c": "temp", "vibration_rms_mms": "vib", "belt_offset_mm": "offset",
                 "belt_tension_kn": "tension", "belt_speed_ms": "speed"}
BASELINE_BUCKETS = 7 * 24 * PER_HOUR   # 7-day rolling baseline
TERRAIN_CODE = {"flat": 0, "incline": 1, "curve": 2}


def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
    s = wide.stack(future_stack=True)
    s.name = name
    return s


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """clean: long frame (ts epoch s, station_id, signals). Returns features indexed by (ts, station_id)."""
    cols = []
    bucket = lambda idx: idx // BUCKET_S * BUCKET_S

    for sig, short in FLEET_SIGNALS.items():
        wide = clean.pivot_table(index="ts", columns="station_id", values=sig, dropna=False)
        dev = wide.sub(wide.median(axis=1), axis=0)
        std_1h = dev.rolling(60, min_periods=20).std().groupby(bucket).last()
        dev10 = dev.groupby(bucket).mean()
        base = dev10.rolling(BASELINE_BUCKETS, min_periods=3 * PER_HOUR).median().shift(1)
        rel = dev10 - base
        if sig == "belt_offset_mm":
            rel = rel.abs()            # misalignment can go either way; magnitude is what matters
        m1 = rel.rolling(PER_HOUR, min_periods=3).mean()
        m6 = rel.rolling(6 * PER_HOUR, min_periods=2 * PER_HOUR).mean()
        cols += [
            _stack(rel, f"{short}_rel"),
            _stack(m1, f"{short}_rel_1h"),
            _stack(m6, f"{short}_rel_6h"),
            _stack(m1 - m1.shift(6 * PER_HOUR), f"{short}_slope_6h"),
            _stack(m6 - m6.shift(24 * PER_HOUR), f"{short}_slope_24h"),
            _stack(rel.rolling(PER_HOUR, min_periods=3).max(), f"{short}_max_1h"),
            _stack(std_1h, f"{short}_std_1h"),
        ]

    # drive-station signals: motor current against load (only the two drive stations have them)
    cur = clean.pivot_table(index="ts", columns="station_id", values="motor_current_a", dropna=False)
    load = clean.pivot_table(index="ts", columns="station_id", values="load_tph", dropna=False).median(axis=1)
    ratio = cur.div(load.clip(lower=500), axis=0)
    ratio10 = ratio.groupby(bucket).mean()
    ratio_rel = ratio10 - ratio10.rolling(BASELINE_BUCKETS, min_periods=3 * PER_HOUR).median().shift(1)
    cols += [_stack(ratio_rel, "current_per_load_rel"),
             _stack(ratio_rel.rolling(6 * PER_HOUR, min_periods=PER_HOUR).mean(), "current_per_load_rel_6h"),
             _stack(cur.rolling(60, min_periods=20).std().groupby(bucket).last(), "current_std_1h")]

    # data quality and operating context
    temp = clean.pivot_table(index="ts", columns="station_id", values="bearing_temp_c", dropna=False)
    missing = temp.isna().astype(float).groupby(bucket).mean().rolling(6 * PER_HOUR, min_periods=1).mean()
    cols.append(_stack(missing, "missing_frac_6h"))
    speed = clean.pivot_table(index="ts", columns="station_id", values="belt_speed_ms", dropna=False).median(axis=1)
    running = (speed > 1).astype(float).groupby(bucket).mean().rolling(PER_HOUR, min_periods=1).mean()

    feats = pd.concat(cols, axis=1)
    feats.index.names = ["ts", "station_id"]
    feats["running_frac_1h"] = running.reindex(feats.index.get_level_values("ts")).to_numpy()
    meta = {s.id: s for s in STATIONS}
    sids = feats.index.get_level_values("station_id")
    feats["terrain"] = [TERRAIN_CODE[meta[s].terrain] for s in sids]
    feats["has_drive"] = [int(meta[s].has_drive) for s in sids]
    # warm-up rows without a baseline yet carry no information
    return feats[feats["temp_rel"].notna() | feats["vib_rel"].notna()]

