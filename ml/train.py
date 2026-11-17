"""Train the failure-prediction models on simulator history.

    python -m simulator.run history --days 60 --out data/train
    python -m ml.train

Two models, both gradient-boosted trees (they handle NaN natively, which matters with messy data):
  - classifier: for each station every 10 min -> P(failure within HORIZON_H) per failure mode ("none" = healthy)
  - regressor:  for stations already degrading -> hours until failure

Evaluation uses a time split (train on the first 40 days, test on the rest), never a random
split: neighbouring 10-minute rows are nearly identical, so a random split would leak the answer.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_score, recall_score

from backend.cleaning import clean_frame
from backend.features import build_features

DATA = Path("data/train")
MODEL_DIR = Path("ml/models")
HORIZON_H = 48
TRAIN_DAYS = 40
THRESHOLD = 0.5
DEBOUNCE = 2      # consecutive 10-min buckets above threshold before an alert counts


def make_labels(feats: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    ts = feats.index.get_level_values("ts").to_numpy()
    sid = feats.index.get_level_values("station_id").to_numpy()
    lab = pd.DataFrame({"label": "none", "ttf_h": np.nan, "degrading": False}, index=feats.index)
    for f in faults.itertuples():
        m = (sid == f.station_id) & (ts < f.failure_at)
        in_horizon = m & (ts >= f.failure_at - HORIZON_H * 3600)
        lab.loc[in_horizon, "label"] = f.mode
        window = m & (ts >= f.start)
        lab.loc[window, "ttf_h"] = (f.failure_at - ts[window]) / 3600
        lab.loc[window, "degrading"] = True
    return lab


def make_models():
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         class_weight="balanced", random_state=0)
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
    return clf, reg


def alert_series(p: pd.Series) -> pd.Series:
    """Debounced alert: True once p has stayed above THRESHOLD for DEBOUNCE consecutive buckets."""
    above = (p >= THRESHOLD).astype(int)
    return above.rolling(DEBOUNCE, min_periods=DEBOUNCE).sum() >= DEBOUNCE


def evaluate(clf, reg, X, lab, faults, t_split, t_end) -> dict:
    proba = pd.DataFrame(clf.predict_proba(X), index=X.index, columns=clf.classes_)
    p_fail = 1 - proba["none"]
    y = lab["label"] != "none"
    out = {
        "rows": int(len(X)), "positive_rows": int(y.sum()),
        "average_precision": round(average_precision_score(y, p_fail), 3),
        "row_precision": round(precision_score(y, p_fail >= THRESHOLD), 3),
        "row_recall": round(recall_score(y, p_fail >= THRESHOLD), 3),
    }
    pos = y & (p_fail >= THRESHOLD)
    pred_mode = proba.drop(columns="none").idxmax(axis=1)
    out["mode_accuracy_on_detected"] = round(float((pred_mode[pos] == lab.loc[pos, "label"]).mean()), 3)

    deg = lab["degrading"] & lab["ttf_h"].le(HORIZON_H * 1.5)
    if deg.any():
        out["ttf_mae_hours"] = round(float(np.abs(reg.predict(X[deg]) - lab.loc[deg, "ttf_h"]).mean()), 1)

    # event level: was each test-period failure flagged in time, and how early?
    events, false_alarms = [], 0
    for sid, p in p_fail.groupby(level="station_id"):
        p = p.droplevel("station_id").sort_index()
        alert = alert_series(p)
        st_faults = faults[(faults.station_id == sid) & (faults.failure_at > t_split) & (faults.failure_at <= t_end)]
        for f in st_faults.itertuples():
            window = alert[(alert.index >= f.failure_at - 7 * 86400) & (alert.index < f.failure_at)]
            window = window[window.index >= max(f.start, f.failure_at - HORIZON_H * 3600 * 2)]
            hit = window[window]
            lead = (f.failure_at - hit.index[0]) / 3600 if len(hit) else None
            events.append({"station": sid, "mode": f.mode, "detected": lead is not None,
                           "lead_time_h": None if lead is None else round(lead, 1)})
        # false alarm = alert episode starting while the station is healthy and not degrading
        healthy = lab.xs(sid, level="station_id")["label"].eq("none") & ~lab.xs(sid, level="station_id")["degrading"]
        starts = alert & ~alert.shift(1, fill_value=False)
        false_alarms += int((starts & healthy.reindex(starts.index, fill_value=False)).sum())

    ev = pd.DataFrame(events)
    days = (t_end - t_split) / 86400
    out["test_failures"] = len(ev)
    out["failures_detected"] = int(ev["detected"].sum()) if len(ev) else 0
    out["median_lead_time_h"] = float(ev["lead_time_h"].median()) if len(ev) and ev["detected"].any() else None
    out["false_alarms_per_station_day"] = round(float(false_alarms / (20 * days)), 4)
    out["events"] = ev
    return out


def write_report(m: dict, path: Path) -> None:
    ev = m.pop("events")
    lines = ["# Prediction model: evaluation report", "",
             f"Time split: train on first {TRAIN_DAYS} days, test on the remainder. "
             f"Horizon: failure within {HORIZON_H} h. Alert threshold {THRESHOLD}, debounced over {DEBOUNCE} x 10 min.", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in m.items()]
    lines += ["", "## Per-failure results (test period)", "", ev.to_markdown(index=False) if len(ev) else "_none_"]
    path.write_text("\n".join(lines) + "\n")


def main():
    raw = pd.read_csv(DATA / "readings.csv.gz")
    faults = pd.read_csv(DATA / "faults.csv")
    print("cleaning...")
    clean, quality = clean_frame(raw)
    print("features...")
    feats = build_features(clean)
    lab = make_labels(feats, faults)
    X = feats

    ts = X.index.get_level_values("ts")
    t0, t_end = ts.min(), ts.max()
    t_split = t0 + TRAIN_DAYS * 86400
    train = ts < t_split - HORIZON_H * 3600     # gap so train labels never look into the test period
    test = ts >= t_split

    clf, reg = make_models()
    print(f"training on {train.sum():,} rows, testing on {test.sum():,}")
    clf.fit(X[train], lab.loc[train, "label"])
    deg_train = train & lab["degrading"].to_numpy()
    reg.fit(X[deg_train], lab.loc[deg_train, "ttf_h"])

    metrics = evaluate(clf, reg, X[test], lab[test], faults, t_split, t_end)
    print({k: v for k, v in metrics.items() if k != "events"})
    print(metrics["events"].to_string())

    # anomaly scales: robust spread of each fleet-relative feature on healthy training rows
    healthy = train & (lab["label"] == "none").to_numpy() & ~lab["degrading"].to_numpy()
    rel_cols = [c for c in X.columns if c.endswith("_rel_1h")]
    scales = {c: float(1.4826 * (X.loc[healthy, c] - X.loc[healthy, c].median()).abs().median()) for c in rel_cols}

    # deploy on all data: the split was for honest evaluation, the live model should see everything
    clf_full, reg_full = make_models()
    clf_full.fit(X, lab["label"])
    deg_all = lab["degrading"].to_numpy()
    reg_full.fit(X[deg_all], lab.loc[deg_all, "ttf_h"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf_full, "regressor": reg_full, "features": list(X.columns),
                 "threshold": THRESHOLD, "debounce": DEBOUNCE, "horizon_h": HORIZON_H, "anomaly_scales": scales},
                MODEL_DIR / "failure_model.joblib")
    write_report(metrics, Path("ml/REPORT.md"))
    print("saved ml/models/failure_model.joblib and ml/REPORT.md")


if __name__ == "__main__":
    main()
