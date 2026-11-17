# Belt Guard: design document

This covers what the system monitors, how it monitors it, and why. For setup, results and screenshots, see the [README](../README.md).

## 1. The problem, restated

A conveyor several kilometres long fails in a few predictable ways, and most of them build up over hours or days before the actual failure:

- **Idler / pulley bearing failure.** This is the most common. A failing bearing gets hot and shakes before it seizes, and a seized idler then cuts the belt.
- **Belt misalignment (mistracking).** The belt drifts sideways, especially on curves. It damages its own edges, spills material, and can rip the belt on the structure.
- **Splice degradation.** The joint where two belt ends are bonded weakens. Every time it passes, it shows up as tension pulses and impacts, until it parts. When a splice parts, the whole belt is gone.
- **Belt slip at a drive.** The drive pulley loses grip. Belt speed and tension drop, and motor current becomes unstable.
- **Drive motor overload.** Motor current climbs against load, and the motor heats up.
- **Surface wear and tears.** Cuts, gouges, edge wear and longitudinal rips from abrasive or sharp material. Point sensors on the structure can't see these, but a camera can.

The system aims to catch these **early enough to plan maintenance** during the six daily hours the belt isn't running, instead of losing an 18-hour shift to an unplanned stop.

## 2. What is monitored, and where

**Layout (assumed):**
- a 5 km conveyor, i.e. a 10 km belt loop counting the carry and return sides;
- 20 sensor stations, one every 250 m (ST-01 to ST-20);
- drive motors at ST-01 (the head drive) and ST-11 (a booster drive halfway along);
- inclines at ST-04 to ST-06 and ST-14/15, curves at ST-09 and ST-17.

Each station reports once a minute:

| Signal | Failure modes it reveals | Why this signal |
|---|---|---|
| Bearing temperature (°C) | bearing failure, motor overload | friction heat is the earliest cheap indicator of bearing distress |
| Vibration RMS (mm/s) | bearing, splice, misalignment | the industry-standard overall machine-condition measure (ISO 10816 zones) |
| Belt lateral offset (mm) | misalignment | measures tracking directly; edge damage follows from it |
| Belt tension (kN) | splice, slip | a weak splice shows up as tension pulses; slip as lost tension |
| Belt speed (m/s) | slip | the belt moving slower than the drive means it is slipping |
| Load (t/h) | context | overload speeds up every failure mode; also used to normalise motor current |
| Ambient temperature, dust | context | so the system doesn't mistake a hot afternoon or a dust storm for a fault |
| Motor current (A), drive stations only | overload, slip | current relative to load is the direct measure of drive health |

**Not monitored, deliberately:**
- *Acoustic sensing* (microphones along the belt). It's good for spotting failing idlers, but it duplicates what vibration already tells us.
- *Thermal cameras.* These would be the next thing to add.
- *X-ray / magnetic scanning of the belt's steel cords.* Expensive, and not something I could simulate meaningfully.

## 3. Architecture

```
 simulator/ ──HTTP batches──► backend/ (FastAPI)
  belt physics                 ingest ─► clean ─► features ─► rules + deviations ─► anomalies
  fault injection                                        └──► gradient boosting ─► predictions
  messy-data layer             SQLite (raw, clean, anomalies, predictions, vision)
                                   ▲                        │
 vision/ ──HTTP per lap────────────┘                        ▼
  dataset images on a              dashboard/ (Streamlit) reads the API every 5 s
  simulated 10 km loop
  patch CNN + loop tracker
```

- **Single source of data.** The simulator is the only source of sensor data. It talks to the backend over HTTP, the same way a site gateway would, and has no back door into the database.
- **The answer key is never sent.** The simulator's record of when each fault really happens (`data/stream_faults.csv`) stays out of the backend. It's used only to evaluate the model offline.
- **Shared code.** The cleaning and feature code (`backend/cleaning.py`, `backend/features.py`) is used **both** by model training and by the live backend. A model trained on differently prepared data than it sees in production is the most common way ML in operations goes wrong.
- **"Now" means belt time.** Everything is timestamped with the time the belt reports, not the wall clock. The simulator runs faster than real time, and real field data also arrives late.

## 4. Messy input: how bad data is handled

The simulator corrupts its own output on purpose (`simulator/messy.py`). The backend deals with each kind of corruption explicitly and never hides what it did:

| Problem | Handling |
|---|---|
| Per-station clock drift (one station drifts 40 s/day) plus jitter | A small tracker per station estimates its clock offset and snaps readings onto the shared 1-minute grid. Verified: after 60 days of drift, cleaning yields exactly 86,400 minutes per station with no misplaced readings. |
| Duplicate readings (re-sent after a network hiccup) | They collapse onto the same (station, minute) key; the last one wins. |
| Readings arriving out of order or late | Every incoming batch triggers a re-clean of the last 2 hours, so late data gets slotted into the right place. |
| Garbage values (`-999`, `9999`) and impossible readings | Physical range checks turn them into missing values. |
| Spikes | A Hampel filter (rolling median ± k·MAD, with a minimum threshold per signal). |
| Stuck sensors | 15 or more identical non-zero readings in a row means the sensor is dead, not that the belt is steady. These are blanked and reported as a `sensor_stuck` anomaly. |
| Short gaps | Gaps of 5 minutes or less are interpolated. |
| Long gaps / outages | Left empty, never invented. The model handles missing values natively, and a station silent for 15 minutes raises a `sensor_offline` anomaly. |
| Malformed records in a batch | Each record is validated on its own; one bad record never rejects the whole batch. Rejections are counted by reason. |
| Backend down | The simulator holds the batch and retries with backoff instead of dropping data. |

Every reading that gets rejected is counted per station and reason (missing, out of range, stuck, spike, interpolated), and the dashboard shows these counts.

## 5. Anomaly detection (what's wrong *now*)

There are three layers, from simplest to most statistical:

1. **Hard limits.** Bearing temperature ≥ 85 °C (critical at 95 °C), vibration ≥ 7.1 mm/s (critical at 11), lateral offset ≥ 40 mm, motor current ≥ 520 A. These are simple and explainable, and they would trigger alarms on a real site even without any ML.
2. **Fleet-relative deviation.** This is the core idea.
   - All 20 stations share the same weather, load, on/off schedule and belt speed.
   - So for each signal, subtract the **fleet median at that minute**. What's left is how this station differs from its neighbours right now.
   - Then subtract the station's own **rolling 7-day median** of that difference. That removes calibration offsets and terrain effects.
   - The remainder is turned into a z-score, using a spread estimate learned from healthy training data.
   - A deviation is flagged only if it is **both** statistically unusual (z ≥ 4) **and** physically meaningful (≥ 3 °C, ≥ 0.5 mm/s, ≥ 8 mm, ≥ 4 kN, ≥ 0.1 m/s).
   - Two refinements came from mistakes in the first version (see the README):
     - Deviations are only assessed while the belt has been running steadily for a full hour. Start-up and shut-down periods are skipped.
     - Temperature and vibration only count when they're *higher* than baseline. A freshly replaced bearing runs cooler, which isn't a risk.
3. **Data quality.** Offline stations and stuck sensors are anomalies too, at low or medium severity. A blind spot on the belt is an operational risk.

**Anomaly lifecycle.** Anomalies are tracked as episodes, not as individual alerts. Each is keyed by (station, type), keeps `first_seen` / `last_seen`, and closes when the condition clears. While the belt is stopped overnight, open deviation anomalies are kept open rather than closed and reopened.

**Station status:**
- `predicted_failure` if the model has raised an alert;
- otherwise `abnormal` if there is any active anomaly of medium severity or higher;
- otherwise `normal`.

## 6. Failure prediction (what will go wrong, where, when)

- **Unit of prediction:** one station, every 10 minutes.
- **Features** (`backend/features.py`), computed for each fleet-relative signal:
  - its level, plus its 1-hour and 6-hour means;
  - its 6-hour and 24-hour trends (slopes);
  - its 1-hour maximum and 1-hour standard deviation. The standard deviation catches splice pulses that the mean misses.
  - For drive stations: motor current per unit of load, compared with that station's baseline.
  - Context: the share of the last hour the belt was running, the share of missing data, terrain type, and whether the station has a drive.
- **Targets:**
  - (a) a classifier for "which failure mode, if any, will happen within 48 h" (`none` or one of the five modes);
  - (b) a regressor estimating hours-to-failure, trained only on stations that are already degrading.
- **Model:** scikit-learn `HistGradientBoosting`. Three reasons:
  - it handles missing values natively, so gaps don't need to be filled in;
  - it trains in seconds on a CPU;
  - it's strong on tabular features like these.

  Deep sequence models would add opacity and training cost without a clear benefit at this data size.
- **Alerting:** an alert fires when P(failure) ≥ 0.5 for two 10-minute checks in a row (debounced), so a single noisy reading can't trigger one.
- **Evaluation:**
  - Split by time: train on days 1 to 38 (a 48-hour gap stops labels leaking across the split) and test on days 41 to 60. Never a random split: adjacent 10-minute rows are nearly identical, so a random split would leak the answers.
  - Metrics are reported per failure event (was it caught, and how early?) and as false alarms per station per day, not just row-level accuracy. See [ml/REPORT.md](../ml/REPORT.md).
- **Why recall matters more than precision:**
  - A missed failure means an unplanned stop of the whole operation, a possible belt rip and a safety exposure.
  - A false alarm means someone walks out to check an idler during the overnight maintenance window.
  - The threshold is set with that asymmetry in mind.

## 7. Vision: wear and tear on a continuous loop

### Model
- A ResNet18, pretrained on ImageNet, with only its last block and classifier head fine-tuned. It classifies **image patches** as `intact` or a damage class.
- The training patches come from a public dataset of labelled belt-damage images: crops around each labelled damage box, plus intact crops taken from areas with no damage. See the README for the dataset and why it was chosen.
- Why a patch classifier rather than an object detector:
  - an inspection camera over a moving belt sees a continuous strip, and classifying patches of that strip fits naturally;
  - it fine-tunes on a CPU in minutes;
  - the share of damaged tiles gives a rough measure of how big the damage is.

### Why the loop matters
A fixed camera sees every metre of belt once per lap, about every 42 minutes. These are the consequences, and how each one is handled (`vision/tracker.py`):

1. **The same tear is detected every lap.** Naively, one tear means an alert every 42 minutes forever.
   → Each detection is tagged with a **position on the belt itself**: metres from a reference splice, measured by the encoder since the splice last passed the camera. Detections are then grouped across laps by that position into one *tracked defect*.
2. **Positions wrap around.** The belt position is circular: 9,998 m and 3 m are 5 m apart, across the splice.
   → All distances are calculated around the loop, and position updates take the shorter way round.
3. **The belt stretches, and the encoder drifts.** Over its life a belt lengthens by a percent or more, so encoder metres don't stay constant. The error accumulates with distance since the splice re-sync, so it's worst *just before* the splice comes back round.
   → Each lap's positions are **divided by that lap's measured splice-to-splice length**, which cancels stretch. The remaining slip error is handled by a tolerance that grows with distance from the splice. *(My first version got this wrong; the README explains.)*
4. **One-off false positives.** Dust, spilled ore or a shadow won't be at the same belt position next lap.
   → New tracks start as `candidate` and need sightings in **3 of the last 4 laps** (tuned by Monte Carlo in `vision/tune_tracker.py`) to become `confirmed`. A candidate that never repeats is dropped. Here the loop works *for* us: every lap is a free second opinion.
5. **Obscured frames aren't evidence.** A frame washed out by dust says nothing about the damage underneath.
   → A low-contrast frame is marked "obscured" and doesn't count as a miss.
6. **Growth is measurable.** Seeing the same defect every lap lets us track its size over time.
   → If the size trend over the last 5 laps is rising, the defect is marked `growing`. That is wear progression, which is the vision side's own form of prediction.
7. **Repairs.**
   → A confirmed defect clearly missed for 5 laps in a row becomes `resolved`. It was most likely patched, or the belt section was replaced.
8. **Two coordinate systems.** A tear travels with the belt; a sensor station is bolted to the structure. A tear at "belt position 1,830 m" passes *every* station each lap.
   → The dashboard shows the two on separate axes and says so, rather than drawing tears as if they sat at a station.

### Deliberately not solved
- **The underside of the belt.** One camera on the carrying side never sees the bottom cover, which gets worn by pulleys and idlers. That would need a second camera on the return side, plus association between the two cameras.
- **True defect size.** The size measure is the share of damaged tiles in the frame, not millimetres. A real system would use a calibrated line-scan camera and segmentation.
- **Splice localisation from images.** I assume a splice marker (RFID or magnetic) gives a reliable once-per-lap reference. Finding the splice visually is its own problem.
- **Lateral position across the belt width.** Not tracked: position is along the belt only.
- **Linking vision with sensors.** For example, a growing tear seen just after a station whose vibration rises once per lap. Possible, because both are timestamped against belt motion, but not built.

## 8. Dashboard

The dashboard (Streamlit) reads the API every 5 seconds and shows:
1. a belt map of all 20 stations, coloured *and* shaped by status: green ● normal, amber ▲ abnormal, red ✖ predicted failure;
2. active anomalies, with location, type, severity and since when;
3. predictions, with failure mode, location, probability, estimated time and an alert flag;
4. vision defects on their own 10 km belt-coordinate axis, with the latest frame, laps seen, status and growth trend;
5. a drill-down chart for each station, a predicted-probability timeline, and a data-quality table.

## 9. "Tool vs. agent"

All of the detection and prediction is **deterministic**: rules, statistics and trained models with tested thresholds. These are the "tools". They are cheap, repeatable and easy to audit, which matters when a decision means stopping a mine.

I chose **not** to put an LLM agent in the detection path. An agent's natural place is *on top* of these tools, for example:
- a shift-handover summary ("ST-07 bearing likely to fail in about 40 h; schedule a change in tonight's window");
- answering ad-hoc questions against the API.

That layer is optional and was skipped for time. The API is already shaped so that each endpoint could be exposed directly as an agent tool.

## 10. Assumptions

- The brief says "multi-kilometre"; I chose 5 km and 20 stations every 250 m.
- One reading per station per minute. Real vibration hardware samples at kHz and reports summary values like these.
- The belt runs 06:00 to 24:00 and is stopped 00:00 to 06:00.
- A "failure" means an event that forces an unplanned stop. After a failure, the component is assumed to be replaced and the station returns to healthy.
- One inspection camera on the carrying side near the head end, with a splice reference marker.
- All timestamps are UTC.
