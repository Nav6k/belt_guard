"""Turns messy station telemetry into a clean 1-minute grid per station.

Shared by the live backend and by offline model training, so the model is trained on
exactly what the backend will produce at runtime.

Order matters:
  1. clock alignment   - undo per-station clock drift, snap to the 1-minute sample grid
  2. de-duplication    - retransmits collapse onto the same (station, ts) key
  3. range validation  - physically impossible values (incl. -999 / 9999 sentinels) -> NaN
  4. stuck detection   - long runs of an identical non-zero value -> NaN (sensor is dead, not the belt)
  5. spike removal     - Hampel filter: points far from the rolling median -> NaN
  6. short-gap fill    - interpolate gaps <= 5 min; longer gaps stay NaN and are reported, not invented

Nothing is silently dropped: every rejected value is counted per station and reason.
"""
import numpy as np
import pandas as pd

from simulator.config import DRIVE_STATIONS, SAMPLE_INTERVAL_S, SIGNALS

# physically plausible ranges; anything outside is a sensor/transmission error, not a belt condition
LIMITS = {
    "bearing_temp_c": (-40, 200), "vibration_rms_mms": (0, 50), "belt_offset_mm": (-200, 200),
    "belt_tension_kn": (0, 200), "belt_speed_ms": (0, 6), "load_tph": (0, 8000),
    "ambient_temp_c": (-40, 60), "dust_level": (0, 1), "motor_current_a": (0, 1500),
}
# minimum deviation the Hampel filter will ever call a spike (stops it firing on flat signals)
SPIKE_FLOOR = {
    "bearing_temp_c": 8, "vibration_rms_mms": 3, "belt_offset_mm": 15, "belt_tension_kn": 12,
    "belt_speed_ms": 1.0, "load_tph": 1500, "ambient_temp_c": 5, "dust_level": 0.3, "motor_current_a": 120,
}
HAMPEL_WINDOW, HAMPEL_K = 15, 6
STUCK_MIN_RUN = 15
MAX_INTERP_GAP = 5


class ClockAligner:
    """Tracks each station's clock offset and maps reported timestamps onto the shared sample grid.

    Drift is slow (seconds per day) compared with the 60 s sample period, so a small-gain tracker
    on the residual to the nearest grid point stays locked indefinitely, even for ST-13's bad clock.
    """

    def __init__(self, gain: float = 0.05):
        self.gain = gain
        self.offset: dict[str, float] = {}

    def align(self, station_ids: np.ndarray, ts: np.ndarray) -> np.ndarray:
        out = np.empty(len(ts), dtype=np.int64)
        for i in np.lexsort((ts, station_ids)):
            sid = station_ids[i]
            off = self.offset.get(sid, 0.0)
            corrected = ts[i] - off
            grid = round(corrected / SAMPLE_INTERVAL_S) * SAMPLE_INTERVAL_S
            self.offset[sid] = off + self.gain * (corrected - grid)
            out[i] = grid
        return out


def _run_lengths(x: np.ndarray) -> np.ndarray:
    """Length of the run of identical values each element belongs to (NaN never forms a run)."""
    same = np.r_[False, x[1:] == x[:-1]]
    run_id = np.cumsum(~same)
    return np.bincount(run_id)[run_id]


def clean_station(df: pd.DataFrame, has_drive: bool = True, grid: pd.Index | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean one station's grid-aligned readings. Returns the cleaned frame (indexed by ts) and rejection counts."""
    df = df.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    if grid is None:
        grid = pd.RangeIndex(df.index.min(), df.index.max() + SAMPLE_INTERVAL_S, SAMPLE_INTERVAL_S)
    df = df.reindex(grid)
    counts = {"missing": 0, "out_of_range": 0, "stuck": 0, "spike": 0, "interpolated": 0}

    out = pd.DataFrame(index=df.index)
    for col in SIGNALS:
        if col == "motor_current_a" and not has_drive:
            out[col] = np.nan    # no motor at this station: absent by design, not missing
            continue
        s = df[col].astype(float) if col in df else pd.Series(np.nan, index=df.index)
        counts["missing"] += int(s.isna().sum())

        lo, hi = LIMITS[col]
        bad = (s < lo) | (s > hi)
        counts["out_of_range"] += int(bad.sum())
        s = s.mask(bad)

        v = s.to_numpy()
        stuck = (_run_lengths(v) >= STUCK_MIN_RUN) & (v != 0) & ~np.isnan(v)
        counts["stuck"] += int(stuck.sum())
        s = s.mask(stuck)

        med = s.rolling(HAMPEL_WINDOW, center=True, min_periods=5).median()
        mad = (s - med).abs().rolling(HAMPEL_WINDOW, center=True, min_periods=5).median()
        spike = (s - med).abs() > np.maximum(HAMPEL_K * 1.4826 * mad, SPIKE_FLOOR[col])
        counts["spike"] += int(spike.sum())
        s = s.mask(spike)

        filled = s.interpolate(limit=MAX_INTERP_GAP, limit_area="inside")
        counts["interpolated"] += int(filled.notna().sum() - s.notna().sum())
        out[col] = filled
    out.index.name = "ts"
    return out, counts


def clean_frame(df: pd.DataFrame, aligner: ClockAligner | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline for a multi-station frame of raw readings (ts in epoch seconds)."""
    df = df.copy()
    aligner = aligner or ClockAligner()
    df["ts"] = aligner.align(df["station_id"].to_numpy(), df["ts"].to_numpy())
    frames, quality = [], []
    for sid, g in df.groupby("station_id"):
        clean, counts = clean_station(g.drop(columns="station_id"), has_drive=sid in DRIVE_STATIONS)
        clean.insert(0, "station_id", sid)
        frames.append(clean.reset_index())
        quality.append({"station_id": sid, "rows": len(clean), **counts})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(quality)
