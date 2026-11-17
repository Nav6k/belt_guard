"""HTTP API.  Run:  uvicorn backend.main:app --port 8000"""
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .engine import Engine

engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start_background()
    yield
    engine.stop()


app = FastAPI(title="Belt Guard", lifespan=lifespan)


class IngestBatch(BaseModel):
    # records are validated one by one in the engine, so a single malformed record can't reject the batch
    readings: list[dict[str, Any]]


class VisionDetection(BaseModel):
    defect_id: str
    lap: int
    ts: int
    belt_position_m: float
    label: str
    confidence: float
    size_px: float
    image: str


class VisionDefect(BaseModel):
    defect_id: str
    belt_position_m: float
    label: str
    status: str
    first_seen: int
    last_seen: int
    laps_seen: int
    laps_missed: int
    confidence: float
    size_px: float
    growth_rate: float
    image: str


class VisionUpdate(BaseModel):
    detections: list[VisionDetection]
    defects: list[VisionDefect]


@app.get("/health")
def health():
    return {"ok": True, "now": engine.now, "model_loaded": engine.model is not None}


@app.post("/ingest")
def ingest(batch: IngestBatch):
    return engine.ingest(batch.readings)


@app.get("/summary")
def summary():
    stations = engine.stations()
    counts = {k: sum(1 for s in stations if s["status"] == k) for k in ("normal", "abnormal", "predicted_failure")}
    return {"now": engine.now, "status_counts": counts, "active_anomalies": len(engine.active_anomalies())}


@app.get("/stations")
def stations():
    return engine.stations()


@app.get("/stations/{station_id}/history")
def station_history(station_id: str, hours: float = 24):
    if station_id not in {s["station_id"] for s in engine.stations()}:
        raise HTTPException(404, "unknown station")
    return engine.history(station_id, hours)


@app.get("/anomalies")
def anomalies(active: bool = True, limit: int = 200):
    if active:
        return engine.active_anomalies()
    return engine.db.query("SELECT * FROM anomalies ORDER BY last_seen DESC LIMIT ?", (limit,))


@app.get("/predictions")
def predictions():
    preds = [p for p in engine.predictions.values()]
    return sorted(preds, key=lambda p: -p["p_fail"])


@app.post("/vision/update")
def vision_update(update: VisionUpdate):
    engine.db.executemany(
        "INSERT INTO vision_detections (defect_id,lap,ts,belt_position_m,label,confidence,size_px,image) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(d.defect_id, d.lap, d.ts, d.belt_position_m, d.label, d.confidence, d.size_px, d.image)
         for d in update.detections])
    # the tracker owns defect state (incl. rejecting one-off false positives), so its list replaces ours
    engine.db.execute("DELETE FROM vision_defects")
    engine.db.executemany(
        "INSERT INTO vision_defects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [tuple(d.model_dump().values()) for d in update.defects])
    return {"ok": True}


@app.get("/vision/defects")
def vision_defects():
    return engine.db.query("SELECT * FROM vision_defects ORDER BY status, confidence DESC")


@app.get("/vision/detections")
def vision_detections(defect_id: str | None = None, limit: int = 500):
    if defect_id:
        return engine.db.query("SELECT * FROM vision_detections WHERE defect_id=? ORDER BY lap", (defect_id,))
    return engine.db.query("SELECT * FROM vision_detections ORDER BY id DESC LIMIT ?", (limit,))
