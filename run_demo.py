"""Start the whole system:  python run_demo.py

backend (8000) -> sensor simulator stream -> vision inspection loop -> dashboard (8501).
Ctrl+C stops everything.
"""
import subprocess
import sys
import time
from pathlib import Path

import requests

PY = sys.executable
FRESH = "--keep-db" not in sys.argv


def wait_for_backend(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            requests.get("http://127.0.0.1:8000/health", timeout=2)
            return
        except requests.RequestException:
            time.sleep(1)
    raise SystemExit("backend did not start")


def main():
    if FRESH:
        for f in Path("data").glob("belt.db*"):
            f.unlink()
    procs = [subprocess.Popen([PY, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--log-level", "warning"])]
    wait_for_backend()
    procs.append(subprocess.Popen([PY, "-m", "simulator.run", "stream", "--backfill-days", "3", "--speed", "60"]))
    if Path("vision/models/belt_patch_resnet18.pt").exists():
        procs.append(subprocess.Popen([PY, "-m", "vision.run", "--laps", "40", "--lap-seconds", "8"]))
    else:
        print("no vision model found: run `python -m vision.train` first; continuing without vision")
    procs.append(subprocess.Popen([PY, "-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true"]))
    print("\nDashboard: http://localhost:8501   API docs: http://localhost:8000/docs   (Ctrl+C to stop)\n")
    try:
        while all(p.poll() is None for p in procs[:2]):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
