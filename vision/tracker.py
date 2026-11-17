"""Loop-aware defect tracking.

A fixed camera over a continuous belt sees every metre of belt once per lap. Consequences:
  1. The same tear is detected every lap. Without association, one tear = an alert every ~42 min forever.
     -> Detections are tied to a *belt coordinate* (metres from the reference splice, measured by the
        encoder since the splice last passed the camera) and associated across laps by position.
  2. The belt coordinate is circular: 9,998 m and 3 m are 5 m apart, across the splice.
     -> All distances are computed modulo the loop length.
  3. The belt stretches over its life, so "metres from the splice" by encoder drifts lap to lap, and
     error accumulates with distance travelled since the splice (worst just *before* the splice returns).
     -> Positions are normalised by the splice-to-splice length measured each lap (cancels stretch);
        the remaining slip error is handled by a tolerance that grows with distance since the splice,
        and each track's position follows recent laps (EMA) rather than an all-time mean.
  4. A one-off false positive (dust, spillage, a shadow) won't be at the same belt position next lap.
     -> Tracks start as `candidate` and need N sightings in the last M laps to be `confirmed`.
        The loop is an advantage here: every lap is a free second opinion.
  5. A frame obscured by dust/spray tells us nothing about that spot.
     -> Unusable frames don't count as misses; only a clear view with no detection does.
  6. Repeated sightings let us measure growth.
     -> Size proxy per lap; a positive trend over recent laps marks the defect `growing`.
  7. A tear that stops being seen was probably repaired (or the belt section replaced).
     -> After K consecutive clear misses a confirmed defect becomes `resolved`.
"""
from dataclasses import dataclass, field

import numpy as np

CONFIRM_N, CONFIRM_M = 3, 4      # chosen with vision/tune_tracker.py (Monte Carlo over measured hit / false-positive rates)
RESOLVE_AFTER = 5
GROWTH_WINDOW = 5
GROWTH_MIN_SLOPE = 0.02   # size-fraction per lap


@dataclass
class Observation:
    lap: int
    ts: int
    belt_position_m: float
    label: str
    confidence: float
    size: float           # fraction of the frame classified as damaged (crude extent proxy)
    image: str


@dataclass
class Track:
    defect_id: str
    position_m: float
    history: list[Observation] = field(default_factory=list)
    lap_results: dict[int, str] = field(default_factory=dict)   # lap -> "seen" | "missed" | "obscured"
    status: str = "candidate"

    @property
    def label(self) -> str:
        labels = [o.label for o in self.history]
        return max(set(labels), key=labels.count)

    @property
    def growth_rate(self) -> float:
        recent = self.history[-GROWTH_WINDOW:]
        if len(recent) < 3:
            return 0.0
        return float(np.polyfit([o.lap for o in recent], [o.size for o in recent], 1)[0])

    def consecutive_misses(self) -> int:
        n = 0
        for lap in sorted(self.lap_results, reverse=True):
            r = self.lap_results[lap]
            if r == "missed":
                n += 1
            elif r == "seen":
                break
        return n


class LoopTracker:
    def __init__(self, loop_length_m: float, base_tol_m: float = 5.0, rel_tol: float = 0.001, ema: float = 0.3):
        self.L = loop_length_m
        self.base_tol, self.rel_tol, self.ema = base_tol_m, rel_tol, ema
        self.tracks: list[Track] = []
        self._next = 1

    def circ_dist(self, a: float, b: float) -> float:
        d = abs(a - b) % self.L
        return min(d, self.L - d)

    def tolerance(self, pos: float) -> float:
        # encoder error accumulates with distance travelled since the splice re-sync
        return self.base_tol + self.rel_tol * (pos % self.L)

    def update_lap(self, lap: int, observations: list[Observation], obscured_positions: list[float]) -> None:
        matched = set()
        for obs in sorted(observations, key=lambda o: -o.confidence):
            best, best_d = None, None
            for t in self.tracks:
                if t.status == "resolved" or id(t) in matched:
                    continue
                d = self.circ_dist(t.position_m, obs.belt_position_m)
                if d <= self.tolerance(obs.belt_position_m) and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                best = Track(f"D-{self._next:03d}", obs.belt_position_m)
                self._next += 1
                self.tracks.append(best)
            best.history.append(obs)
            best.lap_results[lap] = "seen"
            # follow recent laps; step along the shorter way round the loop so 9,998 m -> 2 m doesn't jump
            delta = (obs.belt_position_m - best.position_m + self.L / 2) % self.L - self.L / 2
            best.position_m = (best.position_m + self.ema * delta) % self.L
            matched.add(id(best))

        for t in self.tracks:
            if id(t) in matched or t.status == "resolved":
                continue
            hidden = any(self.circ_dist(t.position_m, p) <= self.tolerance(p) for p in obscured_positions)
            t.lap_results[lap] = "obscured" if hidden else "missed"
        self._update_status(lap)

    def _update_status(self, lap: int) -> None:
        for t in self.tracks:
            if t.status == "resolved":
                continue
            recent = [t.lap_results.get(lap - k) for k in range(CONFIRM_M)]
            seen_recent = sum(r == "seen" for r in recent)
            if t.status == "candidate":
                if seen_recent >= CONFIRM_N:
                    t.status = "confirmed"
                elif t.consecutive_misses() >= CONFIRM_M:
                    t.status = "rejected"        # transient false positive, never repeated
            if t.status in ("confirmed", "growing"):
                if t.consecutive_misses() >= RESOLVE_AFTER:
                    t.status = "resolved"
                else:
                    t.status = "growing" if t.growth_rate >= GROWTH_MIN_SLOPE else "confirmed"
        # rejected candidates are dropped from memory: they were noise
        self.tracks = [t for t in self.tracks if t.status != "rejected"]

    def defects(self) -> list[dict]:
        out = []
        for t in self.tracks:
            last = t.history[-1]
            out.append({
                "defect_id": t.defect_id, "belt_position_m": round(t.position_m, 1), "label": t.label,
                "status": t.status, "first_seen": t.history[0].ts, "last_seen": last.ts,
                "laps_seen": len(t.history), "laps_missed": t.consecutive_misses(),
                "confidence": round(float(np.mean([o.confidence for o in t.history[-3:]])), 3),
                "size_px": round(last.size, 3), "growth_rate": round(t.growth_rate, 4), "image": last.image,
            })
        return out
