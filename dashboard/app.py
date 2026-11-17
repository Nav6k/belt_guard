"""Belt health dashboard.  Run:  streamlit run dashboard/app.py

Reads everything from the backend API; holds no state of its own.
"""
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("BELT_API", "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parent.parent

# status palette: reserved for state, always paired with an icon + label
STATUS = {
    "normal": ("#0ca30c", "●", "Normal"),
    "abnormal": ("#fab219", "▲", "Abnormal"),
    "predicted_failure": ("#d03b3b", "✖", "Predicted failure"),
}
SEVERITY_ICON = {"low": "○ low", "medium": "▲ medium", "high": "◆ high", "critical": "✖ critical"}
VISION_ICON = {"candidate": "◌ candidate", "confirmed": "● confirmed", "growing": "▲ growing", "resolved": "✓ resolved"}

st.set_page_config(page_title="Belt Guard", page_icon="🛠️", layout="wide")


def get(path: str, **params):
    try:
        r = requests.get(API + path, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def fmt_ts(ts) -> str:
    return "-" if ts is None or pd.isna(ts) else pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%d %b %H:%M")


def belt_map(stations: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=5000, y0=0, y1=0, line=dict(color="#8a8a86", width=6))
    for s in stations:
        if s["terrain"] != "flat":
            fig.add_vrect(x0=s["position_m"] - 125, x1=s["position_m"] + 125, fillcolor="#8a8a86",
                          opacity=0.12, line_width=0, annotation_text=s["terrain"], annotation_position="bottom",
                          annotation_font_size=10)
    for status, (color, symbol, label) in STATUS.items():
        pts = [s for s in stations if s["status"] == status]
        if not pts:
            continue
        hover = []
        for s in pts:
            p = s.get("prediction") or {}
            line = f"<b>{s['station_id']}</b> @ {s['position_m']:.0f} m<br>{label}"
            if p:
                line += f"<br>P(fail 48h) = {p['p_fail']:.0%}"
                if p.get("ttf_h") is not None:
                    line += f"<br>{p['mode'].replace('_', ' ')} in ~{p['ttf_h']:.0f} h"
            line += f"<br>active anomalies: {s['active_anomalies']}"
            hover.append(line)
        fig.add_trace(go.Scatter(
            x=[s["position_m"] for s in pts], y=[0] * len(pts), mode="markers+text", name=f"{symbol} {label}",
            marker=dict(size=22, color=color, symbol={"normal": "circle", "abnormal": "triangle-up",
                                                      "predicted_failure": "x"}[status],
                        line=dict(color="white", width=2)),
            text=[s["station_id"].replace("ST-", "") for s in pts], textposition="top center",
            hovertext=hover, hoverinfo="text"))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                      legend=dict(orientation="h", y=1.25, x=0),
                      xaxis=dict(title="distance from head pulley (m)", range=[-100, 5100], showgrid=False),
                      yaxis=dict(visible=False, range=[-1, 1.2]), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def vision_map(defects: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=10000, y0=0, y1=0, line=dict(color="#8a8a86", width=6))
    fig.add_vline(x=0, line_dash="dot", annotation_text="splice (ref.)", annotation_position="top right")
    colors = {"candidate": "#8a8a86", "confirmed": "#fab219", "growing": "#d03b3b", "resolved": "#0ca30c"}
    for status, color in colors.items():
        pts = [d for d in defects if d["status"] == status]
        if pts:
            fig.add_trace(go.Scatter(
                x=[d["belt_position_m"] for d in pts], y=[0] * len(pts), mode="markers+text",
                name=VISION_ICON[status], marker=dict(size=16, color=color, symbol="diamond",
                                                       line=dict(color="white", width=2)),
                text=[d["defect_id"] for d in pts], textposition="top center",
                hovertext=[f"<b>{d['defect_id']}</b> {d['label']}<br>belt pos {d['belt_position_m']:.0f} m"
                           f"<br>seen {d['laps_seen']} laps, conf {d['confidence']:.0%}" for d in pts],
                hoverinfo="text"))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.3, x=0),
                      xaxis=dict(title="position on the belt itself, metres from splice (10 km loop)",
                                 range=[-150, 10150], showgrid=False),
                      yaxis=dict(visible=False, range=[-1, 1.2]), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def station_chart(hist: pd.DataFrame, col: str, title: str, unit: str) -> go.Figure:
    t = pd.to_datetime(hist["ts"], unit="s", utc=True)
    fig = go.Figure(go.Scatter(x=t, y=hist[col], mode="lines", line=dict(width=2, color="#2a78d6"),
                               name=title, hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} " + unit + "<extra></extra>"))
    fig.update_layout(title=dict(text=f"{title} ({unit})", font_size=14), height=220,
                      margin=dict(l=10, r=10, t=35, b=10), hovermode="x unified", showlegend=False)
    return fig


@st.fragment(run_every=5)
def live():
    summary = get("/summary")
    if summary is None:
        st.error(f"Backend not reachable at {API}. Start it with `uvicorn backend.main:app --port 8000`.")
        return
    stations = get("/stations") or []
    anomalies = get("/anomalies") or []
    predictions = get("/predictions") or []
    defects = get("/vision/defects") or []

    c = summary["status_counts"]
    k = st.columns(5)
    k[0].metric("Belt time (UTC)", fmt_ts(summary["now"]))
    k[1].metric("● Normal", c["normal"])
    k[2].metric("▲ Abnormal", c["abnormal"])
    k[3].metric("✖ Predicted failure", c["predicted_failure"])
    k[4].metric("Vision defects (confirmed/growing)", sum(d["status"] in ("confirmed", "growing") for d in defects))

    # 1. status of every sensor point
    st.subheader("1 · Sensor points along the belt")
    st.plotly_chart(belt_map(stations), width="stretch", key="belt_map")

    left, right = st.columns(2)
    # 2. active anomalies
    with left:
        st.subheader("2 · Active anomalies")
        if anomalies:
            df = pd.DataFrame(anomalies)
            st.dataframe(pd.DataFrame({
                "station": df["station_id"], "location (m)": df["position_m"].round(0).astype(int),
                "type": df["type"].str.replace("_", " "), "severity": df["severity"].map(SEVERITY_ICON),
                "detail": df["message"], "since": df["first_seen"].map(fmt_ts)}),
                hide_index=True, width="stretch", height=320)
        else:
            st.success("No active anomalies.")

    # 3. model predictions
    with right:
        st.subheader("3 · Failure predictions (next 48 h)")
        preds = [p for p in predictions if p["p_fail"] >= 0.2]
        if preds:
            meta = {s["station_id"]: s for s in stations}
            df = pd.DataFrame(preds)
            st.dataframe(pd.DataFrame({
                "station": df["station_id"],
                "location (m)": df["station_id"].map(lambda s: int(meta[s]["position_m"])),
                "what": df["mode"].str.replace("_", " "),
                "probability": df["p_fail"],
                "when (est.)": df["predicted_failure_ts"].map(fmt_ts),
                "in (h)": df["ttf_h"],
                "alert": df["alert"].map({True: "✖ ALERT", False: "watch"})}),
                hide_index=True, width="stretch", height=320,
                column_config={"probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
            st.caption("ALERT = probability ≥ 50 % for two consecutive 10-minute checks. "
                       "'Watch' rows are shown from 20 %.")
        else:
            st.success("No station above 20 % failure probability.")

    # 4. vision detections
    st.subheader("4 · Vision: belt wear & tear")
    st.caption("Vision positions are on the **belt itself** (metres from the splice, a 10 km loop), "
               "not along the structure: a tear moves with the belt and passes every station.")
    if defects:
        st.plotly_chart(vision_map(defects), width="stretch", key="vision_map")
        shown = [d for d in defects if d["status"] != "candidate"] or defects
        cols = st.columns(min(4, len(shown)))
        for i, d in enumerate(sorted(shown, key=lambda d: (d["status"] == "resolved", -d["confidence"]))):
            with cols[i % len(cols)]:
                img = ROOT / d["image"]
                if img.exists():
                    st.image(str(img), width="stretch")
                st.markdown(f"**{d['defect_id']}** · {d['label']} · {VISION_ICON[d['status']]}  \n"
                            f"belt pos **{d['belt_position_m']:.0f} m** · seen {d['laps_seen']} laps · "
                            f"conf {d['confidence']:.0%}  \n"
                            f"extent {d['size_px']:.0%} of frame · trend {d['growth_rate']:+.3f}/lap")
    else:
        st.info("No vision data yet. Start `python -m vision.run`.")

    with st.expander("Data quality (last re-clean window per station)"):
        st.dataframe(pd.DataFrame([{"station": s["station_id"], "last seen": fmt_ts(s["last_seen"]),
                                    **s["data_quality"]} for s in stations]), hide_index=True, width="stretch")


def drilldown():
    st.subheader("Station drill-down")
    stations = get("/stations") or []
    if not stations:
        return
    order = sorted(stations, key=lambda s: -((s.get("prediction") or {}).get("p_fail") or 0))
    sid = st.selectbox("Station", [s["station_id"] for s in order],
                       format_func=lambda x: f"{x} ({next(s['status'] for s in stations if s['station_id'] == x).replace('_', ' ')})")
    hours = st.radio("Window", [24, 72, 168], horizontal=True, format_func=lambda h: f"{h // 24} d" if h >= 24 else f"{h} h")
    hist = pd.DataFrame(get(f"/stations/{sid}/history", hours=hours) or [])
    if hist.empty:
        st.info("No history yet.")
        return
    a, b = st.columns(2)
    a.plotly_chart(station_chart(hist, "bearing_temp_c", "Bearing temperature", "°C"), width="stretch")
    b.plotly_chart(station_chart(hist, "vibration_rms_mms", "Vibration RMS", "mm/s"), width="stretch")
    a.plotly_chart(station_chart(hist, "belt_offset_mm", "Belt lateral offset", "mm"), width="stretch")
    b.plotly_chart(station_chart(hist, "belt_tension_kn", "Belt tension", "kN"), width="stretch")
    if hist["p_fail"].notna().any():
        fig = station_chart(hist.dropna(subset=["p_fail"]), "p_fail", "Predicted P(failure within 48 h)", "")
        fig.add_hline(y=0.5, line_dash="dot", annotation_text="alert threshold")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, width="stretch")


st.title("Belt Guard: conveyor health")
st.caption("5 km mining conveyor · 20 sensor stations · simulated telemetry, real-image vision · refreshes every 5 s")
live()
st.divider()
drilldown()
