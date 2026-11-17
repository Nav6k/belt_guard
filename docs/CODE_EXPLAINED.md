# Belt Guard: every file explained simply

This guide walks through every file in the project, in the order it was built. For each file it covers:

- **What it is for**, in one sentence
- **A real-life analogy** to make the idea stick
- **The code, section by section**, with small snippets and plain-English explanations
- **Why it was built this way**
- **How it connects** to the other files

---

## The big picture first

Think of the project as a **hospital for a conveyor belt**:

| Hospital | Belt Guard | Folder |
|---|---|---|
| The patient (who sometimes gets sick) | The simulated belt with its sensors | `simulator/` |
| Lab that cleans up and checks blood samples | Cleaning + feature code | `backend/cleaning.py`, `backend/features.py` |
| The doctor who learned from past patients | The prediction model | `ml/` |
| The hospital's records system and nurses' station | The backend API + database | `backend/` |
| The X-ray department | The vision component | `vision/` |
| The screen on the ward wall | The dashboard | `dashboard/` |

The data flows in one direction:

```
simulator  ──►  backend (clean → analyse → predict)  ──►  database  ──►  dashboard
                     ▲
vision  ─────────────┘
```

---

# Part 0: Project setup files

## 📄 `PROJECT_PLAN.md`

**What it is for:** the to-do list and battle plan written before any code existed.

**What's inside:** the five required components, the technology choices, the belt layout, the list of failure types, the "messy data" ideas, the vision loop problems, a time schedule, what we skip, and a submission checklist.

**Why it matters:** the assessment grades *how you break down the problem*. This file shows the breakdown happened before the coding.

---

## 📄 `requirements.txt`

**What it is for:** the shopping list of Python libraries the project needs.

```
numpy, pandas        → work with numbers and tables
requests             → send data over HTTP (simulator → backend)
fastapi, uvicorn     → the web server (backend)
pydantic             → checks the shape of incoming data
scikit-learn, joblib → the prediction model, and saving it to a file
streamlit, plotly    → the dashboard and its charts
torch, torchvision   → the vision neural network
pillow               → open and edit images
tabulate             → lets pandas write markdown tables (for reports)
pyyaml               → reads the vision dataset's data.yaml file
```

> **In plain words:** anyone can run `pip install -r requirements.txt` and get exactly the tools this project uses.

---

## 📄 `.gitignore`

**What it is for:** tells Git which files **not** to upload to GitHub.

```
__pycache__/  *.pyc     → Python's temporary compiled files
.venv/                  → the virtual environment (huge, and rebuilt by anyone)
data/                   → generated data (can be re-created by the simulator)
*.db                    → the database file
vision/dataset/         → the downloaded images (they have their own licence; reviewers download them)
vision/models/*.pt      → (to be revisited: we may commit the trained vision model)
```

> **Why:** a repo should contain *source*, not things that can be regenerated or that are too big.
> The trained prediction model (`ml/models/failure_model.joblib`) **is** kept, so reviewers can run the system without retraining.

---

# Part 1: The simulator (`simulator/`)

The simulator **pretends to be the real belt**. It's the only data source the system has; the assessment requires this.

## 📄 `simulator/__init__.py`

An empty file. Its only job is to tell Python "this folder is a package", so other code can write `from simulator.config import ...`.
(The same is true of `backend/__init__.py`, `ml/__init__.py` and `vision/__init__.py`.)

---

## 📄 `simulator/config.py`

**What it is for:** one place holding all the *facts* about the belt.

**Analogy:** the architect's drawing of the conveyor. Every other file looks at this drawing instead of inventing its own numbers.

### The constants

```python
BELT_LENGTH_M = 5000          # the conveyor is 5 km long
BELT_SPEED_MS = 4.0           # the belt moves at 4 metres per second
STATION_SPACING_M = 250       # one sensor station every 250 m
N_STATIONS = 5000 // 250      # = 20 stations
SAMPLE_INTERVAL_S = 60        # each station sends one reading per minute
RUN_HOURS = (6, 24)           # belt runs 06:00–24:00 (18 h), stopped 00:00–06:00
MAX_LOAD_TPH = 6000           # maximum 6000 tonnes per hour
```

> **In plain words:** "multi-kilometre" in the brief was turned into concrete numbers. These are *assumptions* and the README must say so.

### Special stations

```python
DRIVE_STATIONS = {"ST-01", "ST-11"}                         # stations with a motor
INCLINE_STATIONS = {"ST-04", "ST-05", "ST-06", "ST-14", "ST-15"}  # uphill sections
CURVE_STATIONS = {"ST-09", "ST-17"}                         # bends
```

Why: the brief says "steep inclines and sharp turns". Hills and bends wear faster, so later code makes those stations fail more often.

### The `Station` data class

```python
@dataclass(frozen=True)
class Station:
    id: str            # "ST-07"
    position_m: float  # 1625.0 metres from the start
    terrain: str       # "flat", "incline" or "curve"
    has_drive: bool    # does it have a motor?
```

A `dataclass` is just a neat container for related values. `frozen=True` means it can't be changed by accident.

### `build_stations()`

Loops 20 times and creates `ST-01` … `ST-20`. Each station sits in the **middle** of its 250 m section (125 m, 375 m, 625 m …).

### `SIGNALS`

The list of the 9 measurements every reading contains: bearing temperature, vibration, belt offset (sideways drift), tension, speed, load, ambient temperature, dust and motor current.

---

## 📄 `simulator/faults.py`

**What it is for:** decides **what breaks, where and when**, and how a breaking part changes the sensor readings.

**Analogy:** a script for a medical drama. It decides in advance "patient in bed 7 will get a fever starting Tuesday and collapse on Friday".

### The key idea: faults develop slowly

A real bearing doesn't fail instantly. It gets slightly warmer for days, then shakes, then seizes. So every fault has:
- a `start`: the moment it begins getting worse
- a `failure_at`: the moment it actually breaks

Between those two moments the fault has a **severity `p`** that goes from 0 to 1:

```python
p = (t - start) / (failure_at - start)
```

> At the start p = 0 (barely sick). Halfway through, p = 0.5. Just before failure, p ≈ 1.

### `FAULT_MODES`: the five failure types

```python
"bearing_failure":   ((48, 144), False)   # develops over 2–6 days, any station
"belt_misalignment": ((24, 72),  False)   # 1–3 days, any station
"splice_failure":    ((48, 120), False)   # 2–5 days, any station
"belt_slip":         ((8, 24),   True)    # 8–24 hours, drive stations only
"motor_overload":    ((12, 48),  True)    # 12–48 hours, drive stations only
```

The `True`/`False` means "only possible where there's a motor".

### The `Fault` class

Stores one fault: its id, station, type, start time, failure time and a `direction` (+1 or −1; misalignment can drift left or right).

`progress(t)` calculates `p` for many timestamps at once. Outside the fault window it returns 0, meaning the part is fine (or has already been replaced).

### `apply_fault(...)`: how a fault changes the readings

This is where each failure gets its "symptoms". Example for a bearing:

```python
sig["bearing_temp_c"]    += 45 * p ** 2.5 * (0.3 + 0.7 * run)
sig["vibration_rms_mms"] += 7 * p ** 2 * run
```

Reading this slowly:
- `45 * p ** 2.5` means up to **+45 °C** at the end. The power 2.5 makes it rise **slowly at first, then very fast**, just like real bearings.
- `(0.3 + 0.7 * run)`: `run` is 1 when the belt is moving and 0 when stopped. A hot bearing stays a bit warm even when stopped (30%).
- Vibration only rises **while running** (`* run`): a stopped machine doesn't shake.

The other symptoms:

| Fault | Symptom in the data |
|---|---|
| Misalignment | belt offset drifts up to ±70 mm, vibration rises a little |
| Splice failure | tension *jumps around randomly* (pulses) + impact vibration |
| Belt slip | belt speed drops, tension drops, motor current becomes jumpy |
| Motor overload | motor current climbs by up to +280 A, temperature rises |

### `FaultScheduler`: who gets sick and when

```python
gap = rng.exponential(18 / rate_factor) days
```

- Each station waits a random time (on average 18 days) before its next fault. "Exponential" is the standard maths for "random events that happen at some average rate".
- `_rate_factor`: inclines and curves get ×1.6, so they fail **more often**.
- `_pick_mode`: picks which fault type, with weights (bearing 40%, misalignment 25% …). Curves get ×2.5 misalignment, because bends make belts drift.
- `ensure_until(t_end)`: schedules faults "lazily", only as far ahead as needed.
- `add(...)`: lets us manually plant a fault (used for the demo).

> **Important:** the scheduler's list of faults is the **answer key**. It's used to train and grade the model, but it's **never sent to the backend**. Otherwise the system would be cheating.

---

## 📄 `simulator/belt.py`

**What it is for:** creates **healthy, realistic** sensor readings, then adds the fault symptoms on top.

**Analogy:** a healthy person's normal day. Heart rate goes up when walking, temperature changes between morning and evening. A disease only makes sense against that normal background.

### Why "normal" must be complicated

If healthy data were a flat line, any fault would be trivially easy to spot and the project would prove nothing. So normal readings change because of:

| Normal cause | How it's coded |
|---|---|
| Day/night temperature | `ambient = 27 + 9 * sin(...)`: peaks around 15:00 |
| Belt stopped at night | `running = (hour >= 6) & (hour < 24)` |
| Load goes up and down | `_ar1(...)`: a random walk that drifts back toward 75% |
| Dust changes | another `_ar1` random walk, higher while running |
| Bearings heat up slowly | `lfilter(...)`: a ~30-minute delay, like a kettle warming |
| Terrain | inclines run warmer and shake more; curves drift sideways |
| Every sensor is slightly different | `self.calib`: a small permanent offset per sensor |

### `_ar1(...)`: the "wandering but sensible" number

```python
x = x + pull * (mean - x) + random_noise
```

Every step the value moves randomly but is gently **pulled back toward the average**. Like a dog on a long leash: it wanders but never runs off.

### `BeltSimulator.generate(n_steps)`: the main function

1. Makes a list of timestamps, one per minute.
2. Works out the time of day → is the belt running?
3. Builds the shared conditions (ambient temperature, load, dust, bearing heat).
4. **For each of the 20 stations**, builds the 9 signals from formulas, e.g.
   ```python
   "belt_speed_ms": 4.0 * run + small noise
   "motor_current_a": (150 + 330 * load) * run   if the station has a motor, else empty (NaN)
   ```
5. Applies every fault active at that station (`apply_fault`).
6. Glues everything into one table: columns `ts, station_id, bearing_temp_c, ...`.

It's **stateful**: it remembers the last load, dust and heat values, so calling it again continues smoothly. That's how "3 days of history, then live streaming" joins up without a jump.

### `ground_truth()`

Returns the answer-key table of faults. Used for training and evaluation only.

---

## 📄 `simulator/messy.py`

**What it is for:** **deliberately damages** the clean data, the way real mine-site data arrives damaged.

**Analogy:** a postal service that sometimes loses letters, delivers some twice, delivers them in the wrong order, and lets a few get wet and unreadable.

### The eight kinds of damage

| Damage | Rate | Real-world cause |
|---|---|---|
| **Outage** (station silent 10–180 min) | ~1 per station every 15 days | radio link lost, power dip |
| **Stuck sensor** (same value for 1–6 h) | ~1 per signal every 40 days | frozen sensor electronics |
| **Single dropped reading** | 0.3% | packet loss |
| **One empty field** | 0.5% | sensor timeout |
| **Spike** (value × 3–8) | 0.1% | electrical interference |
| **Garbage** (`-999` or `9999`) | 0.03% | device error codes |
| **Clock drift** (±2 s/day, ST-13 = 40 s/day) | always | cheap clocks in field devices |
| **Duplicates / swapped order** | 0.5% each | network retransmits |

### How the stuck sensor is coded

```python
for i in where(random < p_stuck):          # pick random moments a sensor freezes
    if ts[i] >= until:                     # not already frozen
        until, value = ts[i] + 1..6 hours, vals[i]
        vals[(ts >= ts[i]) & (ts < until)] = value    # copy the same value forward
```

It also **remembers** freezes that continue into the next batch (`self.stuck`), so a 6-hour freeze isn't cut short at a batch boundary.

### Clock drift

```python
df["ts"] = ts + drift_per_day * days_elapsed + random_jitter
```

> The timestamp becomes *what the station thinks* the time is, not the true time. The backend has to fix this.

---

## 📄 `simulator/run.py`

**What it is for:** the "on switch" of the simulator, with two modes.

### Mode 1: `history`

```
python -m simulator.run history --days 60 --out data/train
```

- Generates 60 days, one day at a time, and messes it up.
- Saves `readings.csv.gz` (the messy data) and `faults.csv` (the answer key).
- That's the **training data** for the prediction model.

### Mode 2: `stream`

```
python -m simulator.run stream --backfill-days 3 --speed 120
```

1. **Demo faults:** plants three faults so something is happening when you open the dashboard:
   - ST-07 bearing failure (started 60 h before "live", fails 96 h after its start)
   - ST-17 misalignment
   - ST-11 motor overload
2. **Backfill:** quickly sends 3 days of history in 1-hour batches, so the dashboard isn't empty.
3. **Live:** sends one minute of readings every `60 / speed` real seconds. At speed 120 that's one simulated minute every 0.5 s.

### Helper functions

- `to_payload(df)`: turns the table into JSON. JSON has no "NaN", so missing values become `null`.
- `post_with_retry(...)`: **store-and-forward**. If the backend is down, it waits 1 s, 2 s, 4 s … (up to 30 s) and tries again. **Data is never thrown away.** We saw this work for real: the simulator started before the backend and simply retried.

---

# Part 2: Cleaning and features (shared code)

These two files live in `backend/`, but **both** model training **and** the live backend use them. That guarantees the model learns from data prepared *exactly* the way it will see it live.

## 📄 `backend/cleaning.py`

**What it is for:** turns messy readings into a clean table: one row per station per minute.

**Analogy:** a lab technician who receives blood samples, throws out broken tubes, fixes mislabelled ones and writes down what was wrong.

The steps happen in a fixed order:

### Step 1: `ClockAligner` fixes clock drift

```python
corrected = ts - my_estimate_of_this_station's_offset
grid      = round(corrected / 60) * 60          # snap to the nearest whole minute
offset   += 0.05 * (corrected - grid)           # learn a little from the leftover error
```

> **In plain words:** for every station it keeps a guess of "how wrong is this station's clock?". Each new reading nudges the guess a little (5%). Because drift is slow (seconds per day) and readings are frequent, the guess keeps up. Even ST-13's bad clock stays locked on.
> **Proof it works:** 60 days in → exactly 86,400 minutes per station out, with no reading landing on the wrong minute.

### Step 2: remove duplicates

`drop_duplicates("ts", keep="last")`: two copies of the same minute become one.

### Step 3: `reindex` onto a full minute grid

Every minute gets a row, even missing ones (which get empty values). Now a gap is *visible* instead of invisible.

### Step 4: range check (`LIMITS`)

```python
"bearing_temp_c": (-40, 200)   # outside this is impossible → mark as missing
```

`-999` and `9999` get caught here.

### Step 5: stuck detection

```python
stuck = (run_length >= 15) & (value != 0)
```

`_run_lengths` counts how many identical values in a row each point belongs to. 15+ identical readings means a dead sensor.
`value != 0` matters because a stopped belt *really* reads exactly 0 speed for 6 hours, which isn't a fault.

### Step 6: spike removal (the Hampel filter)

```python
median = rolling median of 15 neighbours
mad    = typical distance of neighbours from that median
spike  = |value - median| > max(6 × 1.4826 × mad, SPIKE_FLOOR)
```

> **In plain words:** compare each point to its 15 neighbours. If it's wildly further away than its neighbours usually are, it's a spike.
> `SPIKE_FLOOR` is a minimum, e.g. "never call a temperature change under 8 °C a spike", so flat signals don't produce false spikes.

### Step 7: fill only short gaps

```python
s.interpolate(limit=5, limit_area="inside")
```

Gaps of 5 minutes or less are filled by drawing a straight line between neighbours. **Longer gaps stay empty.** Inventing 3 hours of data would be lying.

### Counting everything

`counts = {"missing", "out_of_range", "stuck", "spike", "interpolated"}`: every fix is counted per station. The dashboard shows these, so **nothing is hidden**.

### `has_drive` special case

Non-motor stations have no motor current *by design*. They're not counted as "missing".

### `clean_frame(...)`

Runs all of the above for every station and returns the clean table plus the quality report.

---

## 📄 `backend/features.py`

**What it is for:** summarises each station's recent behaviour into ~45 numbers, every 10 minutes. These numbers are what the model looks at.

**Analogy:** a doctor doesn't read every heartbeat. They look at summaries: "average pulse today", "is it rising?", "how irregular is it?".

### The most important idea: compare with the neighbours

```python
dev = station_value - median_of_all_20_stations_at_that_minute
```

All 20 stations experience the **same** weather, load, start/stop and speed. Subtracting the median of all stations cancels all of that. What's left is **only what is unusual about this station**.

Then:

```python
base = rolling 7-day median of dev (shifted so it never sees the future)
rel  = dev - base
```

This removes each station's *permanent* personality (calibration offset, being on a hill).
**`rel` ≈ 0 means "this station is behaving like its usual self".**

### The features for each signal

For temperature, vibration, offset, tension and speed:

| Feature | Meaning | Why it helps |
|---|---|---|
| `temp_rel` | difference right now | the current state |
| `temp_rel_1h`, `_6h` | averages | ignore momentary noise |
| `temp_slope_6h`, `_24h` | is it rising? | faults *grow*: the trend is the early warning |
| `temp_max_1h` | worst value in the last hour | short peaks |
| `temp_std_1h` | how jumpy it is | splice failures show up as **jumpiness**, not a higher average |

Offset uses the **absolute** value, because drifting left is as bad as drifting right.

### Motor features (drive stations only)

```python
ratio = motor_current / load
```

A motor drawing more current *for the same load* is struggling.

### Context features

- `missing_frac_6h`: how much data was missing recently (the model can learn to be careful)
- `running_frac_1h`: was the belt running?
- `terrain`, `has_drive`: what kind of station this is

The last line drops the first ~3 hours, which have no baseline yet.

---

# Part 3: The prediction model (`ml/`)

## 📄 `ml/train.py`

**What it is for:** teaches a model to answer *"will this station fail within 48 hours, from what, and in how many hours?"*, then grades it honestly.

**Analogy:** a medical student studies 40 days of patient files, then takes an exam on 20 *new* days they've never seen.

### Step 1: `make_labels`: the correct answers

For every (station, 10-minute) row:
- `label`: the fault type if that station fails within the next 48 h, otherwise `"none"`
- `ttf_h`: hours until failure (while the fault is developing)
- `degrading`: is the fault already in progress?

### Step 2: two models

```python
HistGradientBoostingClassifier   # "which failure (or none)?"
HistGradientBoostingRegressor    # "how many hours left?"
```

"Gradient boosting" = hundreds of small decision trees, each one fixing the mistakes of the previous ones.
Why this model:
- it **handles empty values on its own**, which is perfect for messy data
- it trains in seconds on a laptop
- it's very good with table data like ours

`class_weight="balanced"`: failures are rare (~10% of rows), so they're given more weight, otherwise the model could just always say "no failure".

### Step 3: honest split by time

```python
train = first 38 days      (48 h gap so training labels can't peek into the test period)
test  = days 40–60
```

> **Why not a random split?** Row 10:00 and row 10:10 are almost identical. If one is in training and the other in the test, the model "has seen the answer". Splitting by time is how it will really be used: learn from the past, predict the future.

### Step 4: `evaluate`: grading like an operator would

- Row-level precision and recall (standard ML scores)
- **Event-level:** for every real failure in the test period, *was it caught, and how many hours early?*
- **False alarms per station per day:** how often operators would be bothered for nothing
- `alert_series`: an alert only counts if probability ≥ 50% for **2 checks in a row** (debounce), so one noisy value can't raise an alarm.

**Result:** 20 of 20 failures caught, a median of 56 h early, ~0.06 false alarms per station per day. Saved in `ml/REPORT.md`.
**Honest caveat:** these numbers are good because the data is simulated with clean symptoms. Real data would score lower.

### Step 5: anomaly scales

```python
scale = 1.4826 × median(|x - median(x)|)    # on healthy rows only
```

This is a "robust standard deviation". It tells the backend how big a *normal* wobble is for each signal, so it can say "this is 8× bigger than normal".

### Step 6: retrain on everything and save

The split was only for honest grading. The final model learns from all 60 days and is saved to `ml/models/failure_model.joblib` along with the feature list, threshold and scales.

---

# Part 4: The backend (`backend/`)

## 📄 `backend/db.py`

**What it is for:** stores everything in a SQLite database (one file: `data/belt.db`).

**Analogy:** the hospital's filing cabinet.

### The tables

| Table | What's stored |
|---|---|
| `raw_readings` | every reading **exactly as received**, plus when it arrived (an audit trail) |
| `clean_readings` | the cleaned 1-minute data |
| `anomalies` | each anomaly episode: station, type, severity, first/last seen, active? |
| `predictions` | the model's output every 10 minutes |
| `vision_detections` | every single camera detection |
| `vision_defects` | the tracked defects (one row per real tear) |

### The `Database` class

- `PRAGMA journal_mode=WAL`: lets reading and writing happen at the same time.
- `threading.Lock()`: the API and the background analysis thread use the database simultaneously. The lock makes them take turns so they can't corrupt it.
- `execute`, `executemany`, `query`: small helpers. `query` returns rows as dictionaries, e.g. `{"station_id": "ST-07", ...}`.

> **Why SQLite:** zero setup, one file. For a real site, the README suggests TimescaleDB (a database built for time-series).

---

## 📄 `backend/engine.py`

**What it is for:** the **brain** of the backend. It receives data, cleans it, finds anomalies, runs the model and decides each station's status.

**Analogy:** the nurses' station. Results come in, get checked, and the patient board is updated.

### Settings at the top

```python
RECLEAN_WINDOW_S = 2 hours   # each new batch re-cleans the last 2 h (to absorb late data)
BUFFER_S = 8 days            # keep 8 days in memory (features need a 7-day baseline)
OFFLINE_AFTER_S = 15 min     # no data for 15 min → "sensor offline"
```

`RULES`: hard limits. For example, vibration ≥ 7.1 mm/s is "high" and ≥ 11 is "critical", based on the ISO 10816 vibration-severity zones.

`DEVIATION_TYPES`: for each statistical feature, a **minimum physical size** (e.g. 3 °C) that must also be exceeded.

`ONE_SIDED`: temperature and vibration only count if **higher** than usual.

### `ingest(readings)`: receiving data

1. **Check each record separately.** Unknown station → reject. Unreadable timestamp → reject. Non-number value → that one field becomes empty, but the rest of the record survives. **One bad record never ruins the whole batch.**
2. Fix the clock with the `ClockAligner`.
3. Save the raw data to the database (the audit trail).
4. Call `_reclean` for each station.
5. Update `self.now`, the latest **belt time** seen.

### `_reclean(sid, new)`: cleaning incrementally

- Adds the new readings to a 3-hour raw buffer.
- Re-cleans from 30 minutes **before** the earliest new reading. The 30 minutes give the spike filter context.
- Replaces those rows in the clean buffer and the database.

> **Why re-clean instead of cleaning once?** A late reading from 20 minutes ago still lands in the right place, and the newest minutes get cleaned again once their "future neighbours" arrive.

### `start_background` / `analyse_if_due`

A background thread checks every 2 seconds: *"has a new 10-minute block of belt time started?"* If yes, it runs `analyse()`. This keeps `/ingest` fast, because the heavy maths happens separately.

### `analyse()`: the main check-up

1. Combine all stations' clean data → `build_features(...)`.
2. Take the newest feature row per station.
3. Collect anomalies from three checks (below).
4. Run the model (`_predict`).
5. Update the anomaly records and each station's status:

```
model alert                       → "predicted_failure"
any active anomaly ≥ medium       → "abnormal"
otherwise                         → "normal"
```

### The three anomaly checks

**`_rule_anomalies`:** median of the last 10 minutes vs the hard limits. Simple, explainable, would exist on any real site.

**`_deviation_anomalies`:** the statistical check.

```python
z = |rel_1h| / normal_wobble
anomaly if z >= 4 AND |rel_1h| >= minimum physical size
severity "high" if z >= 8 else "medium"
```

Plus two lessons learned from testing:
- **Only judged after the belt has run for a full hour.** When the belt stops, all tensions drop to the same resting value and *every* station looked "abnormal". While stopped, existing anomalies are **held open**, not closed.
- **One-sided for temperature and vibration.** A freshly replaced bearing running cooler was being flagged "high". That's not a risk.

**`_quality_anomalies`:** a station silent > 15 min → `sensor_offline`. Many stuck values → `sensor_stuck`. A blind spot is an operational risk too.

### `_predict(...)`: using the model

- `p_fail = 1 - P("none")`: the chance of *any* failure within 48 h.
- `mode`: the most likely failure type.
- `ttf_h`: hours until failure, from the regressor (only shown when p_fail ≥ 50%).
- **Debounce:** `self._above[sid]` counts consecutive checks above the threshold; an alert needs 2 in a row.
- Every prediction is saved to the database, which makes the probability-over-time chart possible.

### `_update_anomalies(...)`: the anomaly lifecycle

- If the anomaly is already open → update its severity and `last_seen`.
- If it's new → insert it with `first_seen = now`.
- If an open anomaly wasn't found this time → mark it **resolved** (`active = 0`).

> **Why episodes:** operators want *"ST-07 has had high temperature since 12:12"*, not 50 identical alerts.

### Read functions for the API

- `stations()`: one summary per station (status, latest values, prediction, anomaly count, data quality)
- `active_anomalies()`: sorted, most severe first
- `history(sid, hours)`: the chart data, averaged to 5 minutes, with the failure probability attached

---

## 📄 `backend/main.py`

**What it is for:** the **web API**: the "doors" through which the simulator, vision and dashboard talk to the engine.

**Analogy:** the hospital reception desk. Each window serves one kind of request.

### Startup

```python
engine = Engine()                 # create the brain once
lifespan → engine.start_background()   # start the analysis thread when the server starts
```

### The data shapes (Pydantic models)

`IngestBatch`, `VisionDetection`, `VisionDefect`, `VisionUpdate` describe what valid incoming JSON looks like. FastAPI checks it automatically.
Note: `IngestBatch.readings` is deliberately loose (`list[dict]`). Strict checking happens **per record** in the engine, so one bad record can't reject a whole batch.

### The endpoints

| Method + path | Who uses it | What it does |
|---|---|---|
| `GET /health` | everyone | is it alive? what's the current belt time? |
| `POST /ingest` | simulator | receive sensor readings |
| `GET /summary` | dashboard | counts of normal/abnormal/predicted |
| `GET /stations` | dashboard | all 20 stations with status |
| `GET /stations/{id}/history` | dashboard | chart data for one station |
| `GET /anomalies` | dashboard | active anomalies (or full history with `active=false`) |
| `GET /predictions` | dashboard | predictions, highest risk first |
| `POST /vision/update` | vision | detections + current defect list |
| `GET /vision/defects` | dashboard | tracked defects |
| `GET /vision/detections` | dashboard | raw detections (optionally for one defect) |

In `/vision/update`, the defect table is **replaced** each time (`DELETE` then `INSERT`). The vision tracker is the single source of truth, so false positives it has rejected disappear from the database too.

> Bonus: FastAPI auto-generates interactive docs at `http://localhost:8000/docs`.

---

# Part 5: Vision (`vision/`)

## 📄 `vision/data.py`

**What it is for:** reads the Roboflow dataset and cuts the images into small **patches** labelled "intact" or a damage type.

**Analogy:** cutting a big photo album into labelled flash cards for studying.

### How the dataset is laid out (YOLOv8 format)

```
vision/dataset/data.yaml            ← class names, e.g. ["tear", "edge wear"]
vision/dataset/train/images/x.jpg
vision/dataset/train/labels/x.txt   ← one line per damage: class cx cy w h  (all 0–1)
```

### Functions

- `class_names()`: reads the class names from `data.yaml`.
- `read_boxes(...)`: reads a label file and converts it to pixel boxes. It handles both formats:
  - **4 numbers** = a rectangle (centre x, centre y, width, height)
  - **many numbers** = a polygon outline → we take its bounding box

  Broken lines are skipped instead of crashing (messy input again).
- `split_items("train")`: lists all images with their boxes. Unreadable images are skipped.
- `make_patches(...)`:
  - **damage patches:** crop around each box, plus 15% margin so the patch shows the edge between good and damaged rubber
  - **intact patches:** random squares that **don't overlap** any damage box, two per image, with a bounded number of attempts so it can't loop forever

> **Why patches?** A camera over a moving belt sees a continuous strip. Asking "is this piece damaged?" fits that naturally, and it's quick to train on a laptop CPU.

`BELT_DATASET` environment variable: lets us point to a different dataset folder (used for the fake-data test).

---

## 📄 `vision/train.py`

**What it is for:** teaches a neural network to recognise damaged belt patches.

**Analogy:** someone who already knows how to see (shapes, edges, textures) gets a short course in "what a belt tear looks like".

### Transfer learning with ResNet18

```python
m = models.resnet18(weights=ImageNet)     # already trained on 1.2 million everyday photos
for name, p in m.named_parameters():
    p.requires_grad = name.startswith("layer4")   # only the last block keeps learning
m.fc = nn.Linear(512, n_classes)          # new final layer: intact / tear / ...
```

> **In plain words:** the early layers already detect edges and textures. We only retrain the last block and the final "decision" layer. That needs far less data and time.

### Augmentation: making training harder on purpose

```python
RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15)   # damage can face any way
ColorJitter(brightness, contrast)                             # dust haze, lighting changes
GaussianBlur                                                  # motion blur from a moving belt
```

Each epoch, the model sees slightly different versions of every image, so it learns *damage*, not "this exact photo".

### Training loop

- `class weights`: rare classes count more, so the model doesn't ignore them.
- `AdamW`, learning rate 0.001: a standard, reliable optimiser.
- 8 epochs. After each one it checks accuracy on the **validation** images (never used for training) and **keeps the best version**.

### Output

- `vision/models/belt_patch_resnet18.pt`: the trained weights
- `vision/models/classes.json`: the class names, in order
- `vision/REPORT.md`: per-class precision/recall and a confusion matrix ("what got mistaken for what")

---

## 📄 `vision/tracker.py`

**What it is for:** the answer to the assessment's **continuous-loop challenge**. It turns many per-lap detections into a short list of real defects.

**Analogy:** a security guard walking the same round every 42 minutes. On the first round they spot a crack in a wall. On the next round they **don't** report a "new crack". They recognise the same one, check whether it's bigger, and only believe a strange shadow if it's still there next round.

### The seven loop problems and their fixes

The file's top comment lists them:

1. **The same tear is seen every lap.** → Detections are grouped by **position on the belt** (metres from the splice).
2. **Position wraps around.** 9,998 m and 3 m are only 5 m apart. → `circ_dist` measures around the loop.
3. **The belt stretches, and the encoder drifts.** → Positions are divided by each lap's measured length (done in `run.py`), the tolerance grows with distance since the splice, and each track's position follows recent laps.
4. **One-off false positives.** → New tracks start as `candidate` and need **3 sightings in 4 laps** (tested with `vision/tune_tracker.py`).
5. **Dust-covered frames.** → Counted as "obscured", **not** as a miss.
6. **Growth.** → A trend line through the last 5 sizes. Rising = `growing`.
7. **Repairs.** → 5 clear misses in a row = `resolved`.

### The building blocks

**`Observation`:** one detection on one lap: lap, time, belt position, label, confidence, size, image path.

**`Track`:** one real defect followed over time:
- `history`: all its observations
- `lap_results`: for each lap, "seen", "missed" or "obscured"
- `status`: candidate → confirmed / growing → resolved
- `label`: the most common label across its history (a majority vote)
- `growth_rate`: the slope of a straight line through the last 5 sizes (`np.polyfit`)
- `consecutive_misses()`: counts recent "missed" laps, ignoring "obscured"

**`LoopTracker`:**

```python
def circ_dist(a, b):            # distance around the loop
    d = |a - b| % L
    return min(d, L - d)

def tolerance(pos):              # how far apart can two sightings be and still be "the same"?
    return 5 m + 0.001 × distance since splice
```

`update_lap(lap, observations, obscured_positions)`:
1. For each detection (most confident first), find the **nearest** existing track within tolerance. None found → create a new candidate `D-001`, `D-002` …
2. Update the track's position: move **30% of the way** toward the new measurement (an exponential moving average), going the **short way round** the loop.
3. Tracks with no detection this lap → "obscured" if a dusty frame covered that spot, otherwise "missed".
4. `_update_status`:
   ```
   candidate + seen 2 of last 3 laps        → confirmed
   candidate + missed 3 laps                → rejected (noise, deleted)
   confirmed + growth ≥ 0.02/lap            → growing
   confirmed/growing + 5 clear misses       → resolved (repaired)
   ```

`defects()`: exports each track as a dictionary for the API.

> **A mistake found by testing:** the first version assumed position error was largest at *both ends* of the loop. In reality the encoder resets at the splice, so error is largest **just before the splice comes back round**. The defect planted at 9,996 m got split into two "different" defects. The fix: divide by the measured lap length (which cancels stretch) and make the tolerance grow with distance since the splice.

---

## 📄 `vision/run.py`

**What it is for:** simulates the inspection camera lap after lap, runs the neural network on each frame, feeds the tracker and sends the results to the backend.

**Analogy:** a flight simulator. The *pictures* are real, from the dataset; the *belt they're placed on* is simulated.

### Constants

```python
LOOP_M = 2 × 5000 = 10,000 m      # carry side + return side
LAP_S  = 10,000 / 4 = 2,500 s     # ~42 minutes per lap
DETECT_THRESHOLD = 0.6            # 60% sure it's damage → detection
```

### `PatchClassifier.analyse(frame)`

Looks at the **whole frame + a 3×3 grid of tiles** (10 views in total):
- `confidence` = the highest damage probability of any view
- `label` = the most likely damage class for that view
- `size` = the fraction of the 9 tiles that look damaged (a rough size measure)

### `frame_quality(frame)`

Measures the image's **contrast** (standard deviation of brightness). Heavy dust makes everything one flat colour, so contrast is low → the frame is unusable → "obscured".

### `render(item, reveal, rng)`: simulating a growing tear

1. Takes the biggest labelled damage box in the image.
2. Hides part of it by pasting a **blurred** copy over it, so it looks like intact rubber.
3. `reveal` (0–1) controls how much of the tear is visible. It increases each lap, so the **tear grows** along its long side, like a real rip.
4. Crops a camera view around it and adds dust (`add_dust` blends the image with a grey-brown haze).

`intact_view(...)`: crops a damage-free square, for the random "healthy belt" frames.

### `main()`: the lap loop

**Setup:**
- Picks 6 damage images from the held-out split and "installs" them at random belt positions.
- Half exist from the start; half appear later (new damage).
- Growth speed: 0, 0.02 or 0.04 per lap.
- Defect #0 is **repaired halfway** through the run (to test "resolved").
- Defect #0 is placed at **9,996 m**, just before the splice, to test the wrap-around.
- Waits for the backend's belt time, so vision and sensors share one clock.

**Every lap:**
1. `stretch += 0.0002`: the belt gets slightly longer each lap.
2. Builds the frames: active defects + 15 random intact frames.
3. 8% of frames get **heavy dust**.
4. Measured position:
   ```python
   raw      = true_pos × stretch + slip_noise        # what the encoder says
   measured = raw / measured_lap_length × 10,000     # normalise → cancels stretch
   ```
5. Low-quality frame → obscured. Otherwise → classify. Confidence ≥ 0.6 → save the image and create an `Observation`.
6. `tracker.update_lap(...)`.
7. POST the detections + defects to `/vision/update`.
8. Print a summary, e.g. `lap 7: 3 detections → {confirmed: 3, resolved: 1}`.

---

# Part 6: The dashboard

## 📄 `dashboard/app.py`

**What it is for:** shows the belt's health to a human, refreshing every 5 seconds.

**Analogy:** the monitor screen next to a hospital bed.

### Colour and symbol rules

```python
"normal":            green  ●  "Normal"
"abnormal":          amber  ▲  "Abnormal"
"predicted_failure": red    ✖  "Predicted failure"
```

Colour is **never the only signal**: there's always a shape and a word as well, so colour-blind users and black-and-white printouts still work.

### Helper functions

- `get(path)`: calls the API; returns `None` if the backend is down, so the page doesn't crash.
- `fmt_ts(ts)`: turns `1791073440` into `05 Oct 15:51`.

### The charts

**`belt_map(stations)`:** the 5 km belt drawn as a grey line, with 20 markers.
- Colour + shape = status. The number above each marker = the station.
- Light grey bands mark inclines and curves.
- Hovering shows details: failure probability, type, hours left, anomaly count.

**`vision_map(defects)`:** a **separate** 0–10,000 m line for the belt itself, with the splice marked.
> Why separate? A tear **moves with the belt**; a sensor station is **bolted to the structure**. Putting a tear "at station 7" would be wrong. This is consequence #8 of the loop, made visible.

**`station_chart(...)`:** a simple line chart for one signal of one station, with a hover tooltip.

### `live()`: the auto-refreshing part

`@st.fragment(run_every=5)`: only this section re-runs every 5 seconds.

It shows the four **required** sections:
1. **Sensor points:** KPI numbers + the belt map
2. **Active anomalies:** station, location, type, severity (with icon), detail, since when
3. **Predictions:** stations with ≥ 20% probability, showing what, where, probability bar, estimated time, hours left, ALERT or "watch"
4. **Vision:** the defect map + photo cards (ID, label, status, belt position, laps seen, confidence, size, growth trend)

Plus a collapsible **data quality** table (missing, out-of-range, stuck, spikes per station).

### `drilldown()`

Pick a station (the riskiest are listed first) and a time window (1 d / 3 d / 7 d). It shows:
- 4 charts: temperature, vibration, offset, tension
- the **predicted failure probability over time**, with the 50% alert line

> This is where you can *watch* a fault develop: temperature creeping up while the probability line climbs toward the threshold.

---

# Part 7: Running it all

## 📄 `run_demo.py`

**What it is for:** starts everything with one command: `python run_demo.py`

**Analogy:** the master switch that turns on every machine in the right order.

```
1. Delete the old database (fresh start), unless --keep-db is given
2. Start the backend → wait until /health answers
3. Start the simulator stream (3 days backfill, then live at 60× speed)
4. Start vision, if a trained vision model exists (otherwise print a hint)
5. Start the dashboard → http://localhost:8501
6. Ctrl+C → stop everything cleanly
```

---

# Part 8: Documents

## 📄 `docs/DESIGN.md`

Required deliverable #1. It explains **what** is monitored and **why** (each signal → which failure it reveals), the architecture, how messy data is handled, the anomaly logic, the model, the eight consequences of the looping belt, what was deliberately not solved, the "tool vs agent" decision, and all assumptions.

## 📄 `ml/REPORT.md` and `vision/REPORT.md`

Generated automatically by the training scripts. They hold the honest evaluation numbers.

## 📄 `docs/CODE_EXPLAINED.md`

This file.

---

# Summary: one line per file

| File | One-line job |
|---|---|
| `simulator/config.py` | the belt's blueprint: length, stations, terrain, signals |
| `simulator/faults.py` | which parts break, when, and how the readings change |
| `simulator/belt.py` | realistic healthy readings + fault symptoms |
| `simulator/messy.py` | breaks the data the way real field data breaks |
| `simulator/run.py` | saves history to files, or streams live to the backend |
| `backend/cleaning.py` | fixes clocks, duplicates, garbage, stuck sensors, spikes, short gaps |
| `backend/features.py` | "how unusual is this station vs its neighbours and its usual self?" |
| `ml/train.py` | learns to predict failures; graded honestly by time split |
| `backend/db.py` | the SQLite filing cabinet |
| `backend/engine.py` | the brain: clean → anomalies → predictions → status |
| `backend/main.py` | the web API doors |
| `vision/data.py` | dataset → labelled patches |
| `vision/train.py` | fine-tunes ResNet18 to spot damage |
| `vision/tracker.py` | handles the loop: same tear across laps, wrap-around, growth, repairs |
| `vision/run.py` | simulated camera laps with real images |
| `dashboard/app.py` | the screen showing everything |
| `run_demo.py` | one command to start it all |
