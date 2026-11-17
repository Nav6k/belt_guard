"""The analysis engine: ingest -> clean -> features -> anomalies + predictions -> station status.

"Now" is the latest timestamp the belt has reported, not the wall clock: the simulator runs
faster than real time, and a real site gateway can also deliver data late.
"""
import json
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulator.config import SIGNALS, STATIONS

from .cleaning import ClockAligner, clean_station
from .db import Database
from .features import BUCKET_S, build_features

RECLEAN_WINDOW_S = 2 * 3600      # each ingest re-cleans the last 2 h, so late / out-of-order data is absorbed
BUFFER_S = 8 * 86400             # in-memory history: 7-day feature baseline + margin
OFFLINE_AFTER_S = 15 * 60

# hard limits (ISO 10816-style vibration zones, typical bearing alarm temperatures); (high, critical)
RULES = {
    "bearing_temp_c": ("overtemperature", 85, 95),
    "vibration_rms_mms": ("high_vibration", 7.1, 11.0),
    "motor_current_a": ("motor_overcurrent", 520, 600),
}
OFFSET_RULE = ("belt_misalignment", 40, 60)
# statistical deviation from the station's own fleet-relative baseline: feature -> anomaly type
# (anomaly type, signal, smallest deviation that matters physically, regardless of how many sigmas it is)
DEVIATION_TYPES = {
    "temp_rel_1h": ("temperature_deviation", "bearing_temp_c", 3.0),
    "vib_rel_1h": ("vibration_deviation", "vibration_rms_mms", 0.5),
    "offset_rel_1h": ("alignment_deviation", "belt_offset_mm", 8.0),
    "tension_rel_1h": ("tension_deviation", "belt_tension_kn", 4.0),
    "speed_rel_1h": ("speed_deviation", "belt_speed_ms", 0.1),
}
ONE_SIDED = {"temp_rel_1h", "vib_rel_1h"}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
STATION_META = {s.id: s for s in STATIONS}


class Engine:
    def __init__(self, db_path="data/belt.db", model_path="ml/models/failure_model.joblib"):
        self.db = Database(db_path)
        self.aligner = ClockAligner()
        self.model = joblib.load(model_path) if Path(model_path).exists() else None
        self.lock = threading.RLock()
        self.raw: dict[str, pd.DataFrame] = {}       # grid-aligned raw readings, last RECLEAN_WINDOW_S + margin
        self.clean: dict[str, pd.DataFrame] = {}     # cleaned 1-min readings, last BUFFER_S
        self.quality: dict[str, dict] = {}
        self.last_seen: dict[str, int] = {}
        self.now = 0
        self.analysed_bucket = 0
        self.predictions: dict[str, dict] = {}
        self._above: dict[str, int] = {}             # consecutive buckets above threshold (debounce)
        self.anomalies: dict[tuple[str, str], dict] = {}
        self.status: dict[str, str] = {s.id: "normal" for s in STATIONS}
        self._stop = threading.Event()

    # ---------------------------------------------------------------- ingest
    def ingest(self, readings: list[dict]) -> dict:
        """Validate record by record: one bad record must not cost us the whole batch."""
        good, rejected = [], {}
        for r in readings:
            reason = None
            sid = r.get("station_id")
            if sid not in STATION_META:
                reason = "unknown_station"
            else:
                try:
                    ts = int(pd.Timestamp(r.get("ts")).timestamp())
                except (ValueError, TypeError):
                    reason = "bad_timestamp"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            row = {"station_id": sid, "ts_reported": ts}
            for sig in SIGNALS:
                v = r.get(sig)
                try:
                    row[sig] = float(v) if v is not None else np.nan
                except (TypeError, ValueError):
                    row[sig] = np.nan      # non-numeric junk becomes missing, the rest of the record survives
            good.append(row)
        if not good:
            return {"accepted": 0, "rejected": rejected}

        df = pd.DataFrame(good)
        received = time.time()
        with self.lock:
            df["ts"] = self.aligner.align(df["station_id"].to_numpy(), df["ts_reported"].to_numpy())
            self.db.executemany(
                "INSERT INTO raw_readings VALUES (?,?,?,?,?)",
                [(received, r.station_id, int(r.ts_reported), int(r.ts),
                  json.dumps({s: (None if pd.isna(getattr(r, s)) else getattr(r, s)) for s in SIGNALS}))
                 for r in df.itertuples()])
            for sid, g in df.groupby("station_id"):
                self._reclean(sid, g.drop(columns=["station_id", "ts_reported"]))
            self.now = max(self.now, int(df["ts"].max()))
        return {"accepted": len(good), "rejected": rejected}

    def _reclean(self, sid: str, new: pd.DataFrame) -> None:
        raw = pd.concat([self.raw.get(sid), new]) if sid in self.raw else new
        raw = raw.drop_duplicates("ts", keep="last").sort_values("ts")
        latest = int(raw["ts"].max())
        raw = raw[raw["ts"] > latest - RECLEAN_WINDOW_S - 3600]
        self.raw[sid] = raw
        self.last_seen[sid] = max(self.last_seen.get(sid, 0), latest)

        # re-clean from (earliest new point - 30 min of context); keep only rows from the earliest new point on
        start = int(new["ts"].min())
        window = raw[raw["ts"] >= start - 1800]
        cleaned, counts = clean_station(window, has_drive=STATION_META[sid].has_drive)
        cleaned = cleaned[cleaned.index >= start - 1800]
        self.quality[sid] = counts

        old = self.clean.get(sid)
        merged = cleaned if old is None else pd.concat([old[old.index < cleaned.index.min()], cleaned])
        self.clean[sid] = merged[merged.index > latest - BUFFER_S]
        self.db.executemany(
            f"INSERT OR REPLACE INTO clean_readings VALUES (?,?,{','.join('?' * len(SIGNALS))})",
            [(sid, int(ts), *[None if pd.isna(v) else float(v) for v in row]) for ts, row in
             zip(cleaned.index, cleaned[SIGNALS].to_numpy())])

    # ---------------------------------------------------------------- analysis
    def start_background(self, every_s: float = 2.0) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.analyse_if_due()
                except Exception as e:  # keep the loop alive; surface the error in the log
                    print(f"analysis error: {e!r}")
                self._stop.wait(every_s)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def analyse_if_due(self) -> None:
        bucket = self.now // BUCKET_S * BUCKET_S
        if bucket > self.analysed_bucket:
            self.analyse()
            self.analysed_bucket = bucket

    def analyse(self) -> None:
        with self.lock:
            frames = [df.reset_index().assign(station_id=sid) for sid, df in self.clean.items()]
            now = self.now
        if not frames:
            return
        clean = pd.concat(frames, ignore_index=True)
        feats = build_features(clean)
        if feats.empty:          # first minutes after start-up: not enough history for features yet
            return
        latest_ts = feats.index.get_level_values("ts").max()
        latest = feats.xs(latest_ts, level="ts")

        found: dict[tuple[str, str], dict] = {}
        self._rule_anomalies(clean, now, found)
        self._deviation_anomalies(latest, found)
        self._quality_anomalies(now, found)
        preds = self._predict(latest, latest_ts) if self.model else {}

        with self.lock:
            self._update_anomalies(found, now)
            self.predictions = preds
            for sid in STATION_META:
                sev = [SEVERITY_RANK[a["severity"]] for (s, _), a in self.anomalies.items() if s == sid and a["active"]]
                if preds.get(sid, {}).get("alert"):
                    self.status[sid] = "predicted_failure"
                elif sev and max(sev) >= SEVERITY_RANK["medium"]:
                    self.status[sid] = "abnormal"
                else:
                    self.status[sid] = "normal"

    def _rule_anomalies(self, clean: pd.DataFrame, now: int, found: dict) -> None:
        recent = clean[clean["ts"] > now - BUCKET_S].groupby("station_id")[SIGNALS].median()
        for sid, row in recent.iterrows():
            checks = [(sig, *RULES[sig], row[sig]) for sig in RULES]
            checks.append(("belt_offset_mm", *OFFSET_RULE, abs(row["belt_offset_mm"])))
            for sig, kind, high, crit, v in checks:
                if pd.notna(v) and v >= high:
                    found[(sid, kind)] = {"type": kind, "signal": sig, "value": float(v),
                                          "severity": "critical" if v >= crit else "high",
                                          "message": f"{sig} = {v:.1f} (limit {high})"}

    def _deviation_anomalies(self, latest: pd.DataFrame, found: dict) -> None:
        scales = self.model["anomaly_scales"] if self.model else {}
        for sid, row in latest.iterrows():
            # baselines are learned while running; start-up / shutdown transitions would all look anomalous
            # so while stopped, hold existing deviation anomalies open instead of resolving them
            if row.get("running_frac_1h", 0) < 1:
                for kind, _, _ in DEVIATION_TYPES.values():
                    cur = self.anomalies.get((sid, kind))
                    if cur and cur["active"]:
                        found[(sid, kind)] = {k: cur[k] for k in ("type", "signal", "value", "severity", "message")}
                continue
            for feat, (kind, sig, min_dev) in DEVIATION_TYPES.items():
                scale = scales.get(feat)
                if not scale or pd.isna(row.get(feat)):
                    continue
                # hotter / shakier than usual is a risk; cooler / quieter (e.g. just after a bearing swap) isn't
                if feat in ONE_SIDED and row[feat] < 0:
                    continue
                z = abs(row[feat]) / scale
                if z >= 4 and abs(row[feat]) >= min_dev and (sid, kind) not in found:
                    found[(sid, kind)] = {"type": kind, "signal": sig, "value": float(row[feat]),
                                          "severity": "high" if z >= 8 else "medium",
                                          "message": f"{sig} {row[feat]:+.2f} from station baseline (z={z:.1f})"}

    def _quality_anomalies(self, now: int, found: dict) -> None:
        for sid in STATION_META:
            seen = self.last_seen.get(sid)
            if seen is None or now - seen > OFFLINE_AFTER_S:
                mins = (now - seen) / 60 if seen else None
                found[(sid, "sensor_offline")] = {
                    "type": "sensor_offline", "signal": None, "value": mins, "severity": "medium",
                    "message": f"no data for {mins:.0f} min" if mins else "never reported"}
            q = self.quality.get(sid, {})
            if q.get("stuck", 0) >= 15:
                found[(sid, "sensor_stuck")] = {"type": "sensor_stuck", "signal": None, "value": q["stuck"],
                                                "severity": "low", "message": f"{q['stuck']} stuck values in recent window"}

    def _predict(self, latest: pd.DataFrame, latest_ts: int) -> dict:
        m = self.model
        X = latest.reindex(columns=m["features"])
        proba = pd.DataFrame(m["classifier"].predict_proba(X), index=X.index, columns=m["classifier"].classes_)
        ttf = m["regressor"].predict(X)
        out, rows = {}, []
        for i, sid in enumerate(X.index):
            p_fail = float(1 - proba.loc[sid, "none"])
            modes = proba.loc[sid].drop("none")
            self._above[sid] = self._above.get(sid, 0) + 1 if p_fail >= m["threshold"] else 0
            alert = self._above[sid] >= m["debounce"]
            t = float(np.clip(ttf[i], 0, None))
            out[sid] = {"station_id": sid, "p_fail": round(p_fail, 3), "mode": modes.idxmax(),
                        "mode_prob": round(float(modes.max() / max(modes.sum(), 1e-9)), 3),
                        "ttf_h": round(t, 1) if p_fail >= m["threshold"] else None,
                        "predicted_failure_ts": int(latest_ts + t * 3600) if p_fail >= m["threshold"] else None,
                        "alert": bool(alert), "ts": int(latest_ts)}
            rows.append((int(latest_ts), sid, p_fail, modes.idxmax(), out[sid]["mode_prob"], out[sid]["ttf_h"], int(alert)))
        self.db.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?)", rows)
        return out

    def _update_anomalies(self, found: dict, now: int) -> None:
        for key, a in found.items():
            cur = self.anomalies.get(key)
            if cur and cur["active"]:
                cur.update(severity=a["severity"], value=a["value"], message=a["message"], last_seen=now)
                self.db.execute("UPDATE anomalies SET severity=?, value=?, message=?, last_seen=? WHERE id=?",
                                (a["severity"], a["value"], a["message"], now, cur["id"]))
            else:
                rec = {"station_id": key[0], **a, "first_seen": now, "last_seen": now, "active": True}
                self.db.execute("INSERT INTO anomalies (station_id,type,severity,signal,value,message,first_seen,"
                                "last_seen,active) VALUES (?,?,?,?,?,?,?,?,1)",
                                (key[0], a["type"], a["severity"], a["signal"], a["value"], a["message"], now, now))
                rec["id"] = self.db.query("SELECT last_insert_rowid() AS id")[0]["id"]
                self.anomalies[key] = rec
        for key, cur in self.anomalies.items():
            if cur["active"] and key not in found:
                cur["active"] = False
                self.db.execute("UPDATE anomalies SET active=0 WHERE id=?", (cur["id"],))

    # ---------------------------------------------------------------- read side
    def stations(self) -> list[dict]:
        with self.lock:
            out = []
            for s in STATIONS:
                df = self.clean.get(s.id)
                last = df.iloc[-10:][SIGNALS].median().round(2).to_dict() if df is not None and len(df) else {}
                out.append({
                    "station_id": s.id, "position_m": s.position_m, "terrain": s.terrain, "has_drive": s.has_drive,
                    "status": self.status[s.id], "last_seen": self.last_seen.get(s.id),
                    "latest": {k: (None if pd.isna(v) else v) for k, v in last.items()},
                    "prediction": self.predictions.get(s.id),
                    "active_anomalies": sum(1 for (sid, _), a in self.anomalies.items() if sid == s.id and a["active"]),
                    "data_quality": self.quality.get(s.id, {}),
                })
            return out

    def active_anomalies(self) -> list[dict]:
        with self.lock:
            items = [dict(a, position_m=STATION_META[a["station_id"]].position_m)
                     for a in self.anomalies.values() if a["active"]]
        return sorted(items, key=lambda a: (-SEVERITY_RANK[a["severity"]], a["station_id"]))

    def history(self, sid: str, hours: float) -> list[dict]:
        with self.lock:
            df = self.clean.get(sid)
            if df is None:
                return []
            df = df[df.index > self.now - hours * 3600]
        df = df.groupby(df.index // 300 * 300).mean()     # 5-min resolution is plenty for a chart
        preds = {r["ts"]: r["p_fail"] for r in self.db.query(
            "SELECT ts, p_fail FROM predictions WHERE station_id=? AND ts>?", (sid, self.now - hours * 3600))}
        df["p_fail"] = pd.Series(preds).reindex(df.index // BUCKET_S * BUCKET_S).to_numpy()
        df = df.reset_index().rename(columns={"index": "ts"})
        return json.loads(df.to_json(orient="records"))
