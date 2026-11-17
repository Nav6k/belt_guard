"""Simulator entry point.

  history:  python -m simulator.run history --days 60 --out data/train
            Writes messy readings + the ground-truth fault schedule to disk (training data for ml/).

  stream:   python -m simulator.run stream --api http://127.0.0.1:8000 --backfill-days 3 --speed 120
            Backfills a few days quickly, then streams "live" to the backend at `speed`x real time.
            Ground truth is written to data/stream_faults.csv for evaluation only, never sent to the API.
"""
import argparse
import math
import time
from pathlib import Path

import pandas as pd
import requests

from .belt import BeltSimulator
from .config import SAMPLE_INTERVAL_S
from .faults import HOUR
from .messy import MessyLayer

SIM_START = pd.Timestamp("2026-10-01 00:00:00", tz="UTC")
STEPS_PER_DAY = 86400 // SAMPLE_INTERVAL_S


def to_payload(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    records = df.to_dict(orient="records")
    # JSON has no NaN; a missing field is sent as null
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in records]


def post_with_retry(session: requests.Session, url: str, readings: list[dict]) -> None:
    """Store-and-forward: if the backend is down, hold the batch and retry rather than lose it."""
    delay = 1.0
    while True:
        try:
            r = session.post(url, json={"readings": readings}, timeout=30)
            r.raise_for_status()
            return
        except requests.RequestException as e:
            print(f"  backend unavailable ({e.__class__.__name__}), retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)


def cmd_history(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = BeltSimulator(SIM_START, seed=args.seed)
    messy = MessyLayer(sim.rng, sim.t)
    frames = []
    for day in range(args.days):
        frames.append(messy.apply(sim.generate(STEPS_PER_DAY)))
        print(f"day {day + 1}/{args.days}", end="\r")
    readings = pd.concat(frames, ignore_index=True)
    readings.to_csv(out / "readings.csv.gz", index=False)
    sim.ground_truth().to_csv(out / "faults.csv", index=False)
    print(f"\nwrote {len(readings):,} readings and {len(sim.ground_truth())} faults to {out}")


def cmd_stream(args):
    sim = BeltSimulator(SIM_START, seed=args.seed)
    messy = MessyLayer(sim.rng, sim.t)
    live_start = sim.t + args.backfill_days * 86400
    if args.demo:
        # guarantee something interesting is developing when the dashboard opens
        sim.scheduler.add("ST-07", "bearing_failure", live_start - 60 * HOUR, 96)
        sim.scheduler.add("ST-17", "belt_misalignment", live_start - 20 * HOUR, 50)
        sim.scheduler.add("ST-11", "motor_overload", live_start - 6 * HOUR, 30)

    url = args.api.rstrip("/") + "/ingest"
    session = requests.Session()
    gt_path = Path("data/stream_faults.csv")
    gt_path.parent.mkdir(exist_ok=True)

    print(f"backfilling {args.backfill_days} days...")
    for _ in range(args.backfill_days * 24):
        post_with_retry(session, url, to_payload(messy.apply(sim.generate(60))))
    sim.ground_truth().to_csv(gt_path, index=False)

    print(f"streaming live at {args.speed}x (one sample every {SAMPLE_INTERVAL_S / args.speed:.2f}s real time)")
    step = 0
    while True:
        t0 = time.monotonic()
        batch = messy.apply(sim.generate(1))
        post_with_retry(session, url, to_payload(batch))
        step += 1
        if step % 60 == 0:
            sim.ground_truth().to_csv(gt_path, index=False)
            print(f"  sim time {pd.Timestamp(sim.t, unit='s', tz='UTC')}")
        time.sleep(max(0.0, SAMPLE_INTERVAL_S / args.speed - (time.monotonic() - t0)))


def main():
    p = argparse.ArgumentParser(description="Conveyor belt sensor simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history")
    h.add_argument("--days", type=int, default=60)
    h.add_argument("--seed", type=int, default=7)
    h.add_argument("--out", default="data/train")
    s = sub.add_parser("stream")
    s.add_argument("--api", default="http://127.0.0.1:8000")
    s.add_argument("--backfill-days", type=int, default=3)
    s.add_argument("--speed", type=float, default=120, help="simulated seconds per real second")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--no-demo", dest="demo", action="store_false")
    args = p.parse_args()
    {"history": cmd_history, "stream": cmd_stream}[args.cmd](args)


if __name__ == "__main__":
    main()
