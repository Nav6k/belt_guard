# Belt Guard: Project Plan

**Goal:** Build one working system that monitors a simulated multi-kilometre mining conveyor belt and predicts failures before they happen. Data flows from the simulator to the backend, then the models, then the dashboard.

**Deadline:** Tue 17 Nov 2026, 11:59 PM IST
**Target:** Finish in one day (16 Nov 2026), and keep 17 Nov for polish, screenshots and the README.

---

## 1. What I'm building (the 5 required components)

| # | Component | What I'll build | Tech |
|---|-----------|-----------------|------|
| 1 | Design document | `docs/DESIGN.md`, linked from the README: what I monitor, how and why | Markdown |
| 2 | Backend | Takes sensor readings in, validates and cleans them, stores them, runs anomaly detection and prediction, and serves results | FastAPI + SQLite |
| 3 | Prediction model | Anomaly detection, plus a "will this station fail in the next N hours, and from what" predictor | scikit-learn |
| 4 | Vision component | Classifies or detects belt damage in images, and handles the problems caused by the belt being a loop | PyTorch / torchvision (pretrained CNN, fine-tuned) |
| 5 | Dashboard | Shows the status of every sensor point, active anomalies, predictions and vision detections | Streamlit + Plotly |

**Why this stack:** I can build all of it in Python in one day and run it locally with no cloud setup. SQLite needs no database server. Streamlit is the fastest way to get a dashboard that's good enough. I'll note in the README that in production I'd swap in TimescaleDB, MQTT and a React frontend.

---

## 2. The simulated belt

- **Belt:** a 5 km loop running at about 4 m/s, with 18 hours/day running and 6 hours/day stopped.
- **Sensor points:** a sensor suite every 250 m, giving **20 stations** (`ST-01` … `ST-20`). Some stations sit on inclines or curves, and those wear faster.
- **Signals at each station:**

| Signal | Why it matters |
|--------|----------------|
| Idler/pulley bearing temperature (°C) | Bearing failure is the most common conveyor failure |
| Vibration RMS (mm/s) | Shows bearing wear, idler seizure and imbalance early |
| Belt alignment / lateral offset (mm) | Tracking problems lead to edge damage and spillage |
| Belt tension (kN) | Slip, splice failure and take-up problems |
| Belt speed (m/s) | Slip, i.e. a mismatch between the drive and the belt |
| Load (t/h) | Context: overload speeds up all failure modes |
| Ambient temperature / dust | Environmental context that can confuse the other readings |
| Drive motor current (A) | Only at drive stations; shows overload and jams |

- **Failure modes I inject:** bearing degradation (slow rise in temperature and vibration, then a runaway), belt misalignment drift, belt slip, motor overload, splice/tear events. Each one has a **ground-truth failure time**, so the model gets labels to train on.
- **Messy data I inject on purpose** (the assessment calls this out explicitly):
  - missing readings and sensor dropouts
  - stuck/frozen sensors
  - spikes and outliers
  - duplicate and out-of-order timestamps
  - clock drift
  - noise that depends on dust and temperature
- The simulator streams readings to the backend over HTTP. It can also replay a fast-forwarded history (weeks of data generated in seconds) to train the model.

---

## 3. Backend

- `POST /ingest`: takes batched readings, validates them with Pydantic, removes duplicates, reorders by timestamp, flags bad readings rather than silently dropping them, and stores both raw and cleaned values.
- **Analysis pipeline:**
  1. Rule checks: hard limits, such as bearing temperature above 90 °C.
  2. Anomaly detection: per-station rolling z-scores, plus an Isolation Forest over the rolling features.
  3. Failure prediction (section 4).
  4. Combine the results into a status for each station: `normal` / `abnormal` / `predicted_failure`.
- Endpoints the dashboard uses: `GET /stations`, `GET /anomalies`, `GET /predictions`, `GET /vision/detections`, `GET /stations/{id}/history`.

---

## 4. Prediction model

- **Features:** rolling mean, std and slope over 1 h, 6 h and 24 h for each signal; temperature rise above ambient; time since maintenance; terrain type.
- **Targets:**
  - (a) *Will this station fail within the next 24 h?* (classification)
  - (b) *Which failure mode is it?* (multi-class)
  - (c) a rough *time-to-failure* estimate (regression)
- **Model:** Gradient Boosting / Random Forest. It's fast, easy to interpret, and works well on tabular data. Deep learning would be overkill here.
- **Evaluation:** split by time, never randomly, to avoid leakage. I'll report precision and recall, and explain why recall matters more than precision here (a missed failure costs far more than a false alarm).
- **Output:** "Station ST-07, bearing failure, ~14 h, 82% confidence". This covers the *what, where and when* the dashboard has to show.

---

## 5. Vision component

**Dataset:** I'll search Kaggle and Roboflow Universe for a conveyor-belt damage dataset (tears, edge damage, holes, wear). If nothing suitable is public, I'll fall back to a general surface-defect dataset (e.g. NEU surface defects) and explain why that's a weaker substitute. The README will document the source, the license and why I chose it.

**Model:** fine-tune a pretrained ResNet18 or MobileNet to classify damage vs. no damage and the damage type. A detector (YOLO) only if the dataset has bounding boxes and time allows.

### The continuous-loop challenge (the key thinking part)

A fixed camera sees the same belt over and over, once per lap (about 21 min for 5 km at 4 m/s). This causes several problems:

1. **Duplicate detections:** the same tear gets detected every lap. Without deduplication, one defect turns into dozens of alerts.
   → *Fix:* give every detection a **position along the belt**, measured from a reference splice and the distance the belt has travelled since. Then group detections at the same belt position into one tracked defect.
2. **Two coordinate systems:** a damaged patch on the belt moves; a sensor station stays put. A defect at "belt position 1,830 m" and a bearing problem at "station ST-07" are different kinds of location.
   → *Fix:* show vision defects in belt coordinates and sensor faults in structure coordinates, and explain the difference on the dashboard.
3. **Tracking how damage grows:** seeing the same defect again on every lap is actually *useful*. If its size or confidence keeps growing from lap to lap, that's wear progression, which is a trend we can predict from.
   → *Fix:* keep a history of each defect across laps and flag it when it's growing.
4. **Belt stretch and drift:** the belt stretches over time, so lap length changes and position estimates drift.
   → *Fix:* re-sync to the reference splice every lap. (Stretch compensation will be simplified and documented as such.)
5. **Top vs. bottom surface:** a camera on the carrying run never sees the underside, which gets worn by the pulleys.
   → I'll document this as a limitation and suggest a second camera on the return run.
6. **Dust and occlusion:** a bad frame on one lap shouldn't erase a defect we're confident about.
   → *Fix:* only confirm a defect after it's seen on N of M laps, and only clear it after it's been missing for several laps in a row.

To demonstrate this, I'll write a simulated camera that assigns dataset images to positions on the belt and "re-observes" them each lap, adding noise and gradual growth.

---

## 6. Dashboard (the minimum 4, plus extras)

1. **Belt map:** a line diagram of the 5 km belt with all 20 stations coloured 🟢 normal / 🟡 abnormal / 🔴 predicted failure.
2. **Active anomalies table:** location, type, severity, first seen.
3. **Predictions panel:** what might fail, where, when, and how confident the model is.
4. **Vision detections:** image thumbnails, defect type, position on the belt, laps seen, growth trend.
5. *Extras:* charts of each station's sensor history, a data-quality panel (dropouts and stuck sensors).

---

## 7. Repo structure

```
belt-guard/
├── README.md              # first-class deliverable
├── docs/DESIGN.md         # design document
├── simulator/             # belt + sensor + fault + messy-data simulator
├── backend/               # FastAPI app, DB models, analysis pipeline
├── ml/                    # feature engineering, training, saved models
├── vision/                # dataset prep, training, loop-aware defect tracker
├── dashboard/             # Streamlit app
├── docs/screenshots/      # dashboard screenshots for README
├── ai_usage/              # AI chat logs (this Claude Code session + others)
└── requirements.txt
```

---

## 8. Today's schedule

| Time block | Task |
|------------|------|
| 1 h | Design doc skeleton, repo setup, simulator: belt model + sensors |
| 1.5 h | Simulator: fault injection + messy data + history generator |
| 1.5 h | Backend: ingest, cleaning, storage, endpoints |
| 1.5 h | Prediction model: features, training, evaluation, wiring into backend |
| 2 h | Vision: pick dataset, fine-tune classifier, loop-aware defect tracker |
| 1.5 h | Dashboard |
| 1 h | End-to-end run, screenshots, README, ai_usage export |

---

## 9. What I'll deliberately skip (and say so in the README)

- A real message broker or cloud deployment. A local HTTP stream is enough to prove the design.
- Acoustic sensors and thermal imaging. I'll mention them as future work.
- Accurate belt-stretch physics. I'll use a simplified model with splice re-sync instead.
- Bottom-surface vision.
- Authentication and multiple users on the dashboard.
- An LLM agent layer. If there's time, a stretch goal is an LLM that writes a plain-English shift summary from the anomaly and prediction data. Otherwise I'll explain why the core detection stays deterministic and tool-based.

---

## 10. Assumptions (to record in the README)

- 5 km belt and 20 stations, 250 m apart. I chose these numbers; the brief only says "multi-kilometre".
- Readings every 10 s, aggregated into 1-minute features.
- One fixed camera on the carrying run, near the head pulley.
- A failure means an event that would require an unplanned stop.

---

## 11. Submission checklist

- [ ] All code: simulator, backend, model, vision, dashboard
- [ ] README: design decisions, vision dataset + why, approach, screenshots, what I'd do differently
- [ ] `docs/DESIGN.md` linked from the README
- [ ] Section in the README on the continuous-loop vision problem
- [ ] `ai_usage/` with chat logs, including where I pushed back on AI suggestions
- [ ] Push to GitHub and send the link before 17 Nov, 11:59 PM IST
