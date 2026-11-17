"""Inspection camera simulation + inference + loop-aware tracking, posting results to the backend.

    python -m vision.run --laps 30 --lap-seconds 4

What is simulated (images are real dataset images; the *belt* around them is simulated):
  - damage images from the held-out split are "installed" at fixed belt positions on a 10 km loop
  - some damage exists from lap 0, some appears later, one is repaired mid-run
  - tears grow lap by lap: a growing fraction of the labelled damage region is revealed
    (the rest is blurred to look like intact rubber)
  - the belt stretches lap by lap and slips a little within each lap; the encoder re-syncs at the splice
    and positions are normalised by the measured splice-to-splice length
  - dust haze on every frame, and occasionally a frame so obscured it's unusable
  - each lap the camera also inspects random intact belt sections, which is where false positives come from
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter

from simulator.config import BELT_LENGTH_M, BELT_SPEED_MS

from .data import DATASET, INTACT, split_items
from .tracker import LoopTracker, Observation
from .train import IMG, build_model, eval_tf

LOOP_M = 2 * BELT_LENGTH_M                 # carry side + return side
LAP_S = int(LOOP_M / BELT_SPEED_MS)        # ~42 min per lap
FRAMES = Path("data/vision_frames")
DETECT_THRESHOLD = 0.6


class PatchClassifier:
    def __init__(self, model_dir: Path = Path("vision/models")):
        self.classes = json.loads((model_dir / "classes.json").read_text())
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(torch.load(model_dir / "belt_patch_resnet18.pt", map_location="cpu"))
        self.model.eval()

    def analyse(self, frame: Image.Image) -> tuple[str, float, float]:
        """Classify the whole frame and a 3x3 grid of tiles. Returns (label, confidence, damaged fraction)."""
        w, h = frame.size
        tiles = [frame] + [frame.crop((i * w // 3, j * h // 3, (i + 1) * w // 3, (j + 1) * h // 3))
                           for j in range(3) for i in range(3)]
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.stack([eval_tf(t) for t in tiles])), 1).numpy()
        intact = self.classes.index(INTACT)
        damage = 1 - probs[:, intact]
        best = int(np.argmax(damage))
        cls_probs = probs[best].copy()
        cls_probs[intact] = 0
        size = float((damage[1:] >= DETECT_THRESHOLD).mean())
        return self.classes[int(np.argmax(cls_probs))], float(damage.max()), size


def frame_quality(frame: Image.Image) -> float:
    """Contrast-based usability score: heavy dust or spray flattens the image."""
    g = np.asarray(frame.convert("L"), dtype=float)
    return float(g.std())


def render(item, reveal: float, rng: random.Random) -> Image.Image | None:
    """Camera view around a labelled defect, with only `reveal` of the damage visible (tear growth)."""
    img_path, boxes = item
    im = Image.open(img_path).convert("RGB")
    b = max(boxes, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0))
    bw, bh = b.x1 - b.x0, b.y1 - b.y0
    w, h = im.size
    view = (max(0, b.x0 - bw * 0.5), max(0, b.y0 - bh * 0.5), min(w, b.x1 + bw * 0.5), min(h, b.y1 + bh * 0.5))
    blurred = im.filter(ImageFilter.GaussianBlur(12))
    # hide the not-yet-grown part of the tear along its long axis
    if bw >= bh:
        hide = (int(b.x0 + bw * reveal), int(b.y0), int(b.x1), int(b.y1))
    else:
        hide = (int(b.x0), int(b.y0 + bh * reveal), int(b.x1), int(b.y1))
    if hide[2] > hide[0] and hide[3] > hide[1]:
        im.paste(blurred.crop(hide), hide[:2])
    return add_dust(im.crop(view), rng)


def add_dust(im: Image.Image, rng: random.Random, heavy: bool = False) -> Image.Image:
    alpha = rng.uniform(0.8, 0.95) if heavy else rng.uniform(0.0, 0.35)
    haze = Image.new("RGB", im.size, (150, 140, 125))
    return Image.blend(im, haze, alpha)


def intact_view(item, rng: random.Random) -> Image.Image:
    img_path, boxes = item
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    side = int(min(w, h) * 0.4)
    for _ in range(20):
        x0, y0 = rng.randint(0, w - side), rng.randint(0, h - side)
        if not any(not (x0 + side <= b.x0 or x0 >= b.x1 or y0 + side <= b.y0 or y0 >= b.y1) for b in boxes):
            break
    return add_dust(im.crop((x0, y0, x0 + side, y0 + side)), rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--laps", type=int, default=30)
    p.add_argument("--lap-seconds", type=float, default=4.0, help="real seconds per simulated lap")
    p.add_argument("--defects", type=int, default=6)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    rng = random.Random(args.seed)

    clf = PatchClassifier()
    split = "test" if (DATASET / "test" / "images").exists() else "valid"
    items = split_items(split)
    damaged = [it for it in items if it[1]]
    rng.shuffle(damaged)
    planted = []
    for k, item in enumerate(damaged[:args.defects]):
        planted.append({
            "item": item, "pos": rng.uniform(0, LOOP_M),
            "onset": 0 if k < args.defects // 2 else rng.randint(3, args.laps // 2),
            "reveal0": rng.uniform(0.3, 0.6), "growth": rng.choice([0.0, 0.02, 0.04]),
            "repaired_at": args.laps // 2 if k == 0 else None,
        })
    if planted:
        planted[0]["pos"] = LOOP_M - 4.0      # one defect right before the splice, to exercise wrap-around
    print("planted:", [(round(d["pos"]), d["onset"], d["growth"]) for d in planted])

    FRAMES.mkdir(parents=True, exist_ok=True)
    tracker = LoopTracker(LOOP_M)
    session = requests.Session()
    # align lap timestamps with belt time (sensor stream), waiting briefly for the backfill to start
    t0 = 0
    for _ in range(60):
        try:
            t0 = session.get(args.api + "/health", timeout=5).json()["now"]
        except requests.RequestException:
            pass
        if t0:
            break
        time.sleep(2)
    t0 = t0 or int(time.time())
    stretch = 0.0

    for lap in range(args.laps):
        try:   # stamp with current belt time so vision and sensor events share one clock
            lap_ts = session.get(args.api + "/health", timeout=5).json()["now"] or t0 + lap * LAP_S
        except requests.RequestException:
            lap_ts = t0 + lap * LAP_S
        stretch += 0.0002                                  # belt slowly lengthens (0.6 % over 30 laps)
        lap_scale = 1 + stretch                            # encoder metres per nominal belt metre this lap
        measured_lap = LOOP_M * lap_scale                  # splice-to-splice encoder length, measured every lap
        obs, obscured, detections = [], [], []

        views = []
        for d in planted:
            active = lap >= d["onset"] and (d["repaired_at"] is None or lap < d["repaired_at"])
            if active:
                reveal = min(1.0, d["reveal0"] + d["growth"] * (lap - d["onset"]))
                views.append((d["pos"], render(d["item"], reveal, rng)))
        for _ in range(15):                                # random intact belt sections
            views.append((rng.uniform(0, LOOP_M), intact_view(rng.choice(items), rng)))

        for i, (true_pos, frame) in enumerate(views):
            if rng.random() < 0.08:
                frame = add_dust(frame, rng, heavy=True)
            # encoder distance since splice, with slip error that grows along the lap; then normalise
            raw = true_pos * lap_scale + rng.gauss(0, 0.5 + 0.0003 * true_pos)
            measured = (raw / measured_lap * LOOP_M) % LOOP_M
            if frame_quality(frame) < 8:
                obscured.append(measured)
                continue
            label, conf, size = clf.analyse(frame)
            if conf < DETECT_THRESHOLD:
                continue
            path = FRAMES / f"lap{lap:03d}_{i:02d}.jpg"
            frame.resize((IMG * 2, int(IMG * 2 * frame.size[1] / frame.size[0]))).save(path, quality=85)
            obs.append(Observation(lap, lap_ts, measured, label, conf, size, str(path).replace("\\", "/")))

        tracker.update_lap(lap, obs, obscured)
        id_by_obs = {id(o): t.defect_id for t in tracker.tracks for o in t.history}
        detections = [{"defect_id": id_by_obs.get(id(o), "rejected"), "lap": o.lap, "ts": o.ts,
                       "belt_position_m": round(o.belt_position_m, 1), "label": o.label,
                       "confidence": round(o.confidence, 3), "size_px": round(o.size, 3), "image": o.image}
                      for o in obs]
        defects = tracker.defects()
        try:
            session.post(args.api + "/vision/update", json={"detections": detections, "defects": defects}, timeout=10)
        except requests.RequestException as e:
            print(f"  backend unavailable: {e.__class__.__name__}")
        summary = {s: sum(d["status"] == s for d in defects) for s in ("candidate", "confirmed", "growing", "resolved")}
        print(f"lap {lap:3d}: {len(obs)} detections, {len(obscured)} obscured frames -> {summary}")
        time.sleep(args.lap_seconds)


if __name__ == "__main__":
    main()
