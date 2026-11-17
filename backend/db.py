"""SQLite persistence. One file, no server: enough for one belt; see README for the production swap (TimescaleDB)."""
import sqlite3
import threading
from pathlib import Path

from simulator.config import SIGNALS

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS raw_readings (
    received_at REAL, station_id TEXT, ts_reported INTEGER, ts_grid INTEGER, payload TEXT
);
CREATE TABLE IF NOT EXISTS clean_readings (
    station_id TEXT, ts INTEGER, {", ".join(f"{s} REAL" for s in SIGNALS)},
    PRIMARY KEY (station_id, ts)
);
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, type TEXT, severity TEXT, signal TEXT,
    value REAL, message TEXT, first_seen INTEGER, last_seen INTEGER, active INTEGER
);
CREATE TABLE IF NOT EXISTS predictions (
    ts INTEGER, station_id TEXT, p_fail REAL, mode TEXT, mode_prob REAL, ttf_h REAL, alert INTEGER,
    PRIMARY KEY (station_id, ts)
);
CREATE TABLE IF NOT EXISTS vision_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, defect_id TEXT, lap INTEGER, ts INTEGER, belt_position_m REAL,
    label TEXT, confidence REAL, size_px REAL, image TEXT
);
CREATE TABLE IF NOT EXISTS vision_defects (
    defect_id TEXT PRIMARY KEY, belt_position_m REAL, label TEXT, status TEXT, first_seen INTEGER,
    last_seen INTEGER, laps_seen INTEGER, laps_missed INTEGER, confidence REAL, size_px REAL,
    growth_rate REAL, image TEXT
);
CREATE INDEX IF NOT EXISTS idx_clean_ts ON clean_readings(ts);
"""


class Database:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def executemany(self, sql: str, rows) -> None:
        with self.lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def execute(self, sql: str, params=()) -> None:
        with self.lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def query(self, sql: str, params=()) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
