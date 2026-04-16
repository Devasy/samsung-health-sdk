#!/usr/bin/env python3
"""
Brahmatal Trek Health Dashboard — March 2026
Run:  python scripts/build_trek_dashboard.py
Out:  trek_dashboard.html
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from samsung_health_sdk import SamsungHealthParser
from samsung_health_sdk.metrics.exercise import ExerciseMetric

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "samsunghealth_patel.devasy.23_20260328222493"
OUTPUT   = Path(__file__).parent.parent / "trek_dashboard.html"
IST      = "Asia/Kolkata"
PACK_KG  = 12
BODY_KG  = 70

PHASES = [
    ("baseline",  "Baseline",           "2026-02-15", "2026-03-01", "#52B788",  341),
    ("lohajung",  "Lohajung 2,335 m",   "2026-03-02", "2026-03-02", "#95D5B2", 2335),
    ("bekaltal",  "Bekaltal 2,954 m",   "2026-03-03", "2026-03-03", "#F4D03F", 2954),
    ("brahmatal", "Brahmatal 3,182 m",  "2026-03-04", "2026-03-04", "#E67E22", 3182),
    ("summit",    "Summit 3,734 m",     "2026-03-05", "2026-03-05", "#C0392B", 3734),
    ("descent",   "Descent 2,335 m",    "2026-03-06", "2026-03-06", "#F39C12", 2335),
    ("recovery",  "Recovery",           "2026-03-07", "2026-03-15", "#3498DB",  341),
]

TREK_TYPES    = {13001, 13003, 15002}
BACKPACK_TYPE = 13003

SESSION_META = [
    ("Mar 3 — Lohajung to Bekaltal",    "#F4D03F"),
    ("Mar 4 — Bekaltal to Brahmatal",   "#E67E22"),
    ("Mar 5a — Camp to Summit",         "#C0392B"),
    ("Mar 5b — Summit to Camp",         "#E74C3C"),
    ("Mar 5c — Camp area return",       "#F39C12"),
    ("Mar 6 — Brahmatal to Lohajung",   "#3498DB"),
]

# Trek timeline milestones (IST)
MILESTONES = [
    ("2026-03-02 20:00", "Lohajung\nHomestay",    "#95D5B2"),
    ("2026-03-03 17:00", "Bekaltal\nCamp",         "#F4D03F"),
    ("2026-03-04 17:00", "Brahmatal\nBase Camp",   "#E67E22"),
    ("2026-03-05 08:00", "Summit\n3,734 m",        "#C0392B"),
    ("2026-03-06 02:15", "2:15 AM\nWake up",       "#a78bfa"),
    ("2026-03-06 06:00", "Jhandi Top\nSunrise",    "#fbbf24"),
    ("2026-03-06 18:00", "Lohajung\n(descent)",    "#F39C12"),
]

# ── Palette ────────────────────────────────────────────────────────────────────
BG   = "#080d18"
SURF = "#0f1728"
SUR2 = "#162035"
BDR  = "#1e2d45"
TEXT = "#e2e8f0"
MUT  = "#64748b"
MUT2 = "#475569"

# ── Helpers ────────────────────────────────────────────────────────────────────

def hex_rgba(h: str, a: float = 0.12) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"


def o2_frac(alt_m: float) -> float:
    """Fraction of sea-level O2 pressure via standard atmosphere."""
    L, T0 = 0.0065, 288.15
    exp = 9.80665 * 0.0289644 / (8.31447 * L)   # ≈ 5.256
    return max(0.50, (1.0 - L * alt_m / T0) ** exp)


def to_ist(df: pd.DataFrame, col: str = "start_time") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].dt.tz_convert(IST)
    return df


def tag_phase(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df["phase"]        = "unknown"
    df["phase_label"]  = "-"
    df["phase_color"]  = MUT
    df["phase_elev_m"] = 0
    for pid, lbl, s, e, col, elv in PHASES:
        m = (df[date_col] >= pd.Timestamp(s).date()) & (df[date_col] <= pd.Timestamp(e).date())
        df.loc[m, "phase"]        = pid
        df.loc[m, "phase_label"]  = lbl
        df.loc[m, "phase_color"]  = col
        df.loc[m, "phase_elev_m"] = elv
    return df


def dark_fig(fig: go.Figure, title: str = "", height: int = 480) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=15,
                   family="Inter,'Segoe UI',system-ui"), x=0.01),
        paper_bgcolor=SURF, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter,'Segoe UI',system-ui", size=12),
        height=height,
        margin=dict(l=60, r=24, t=55, b=50),
        legend=dict(bgcolor=SUR2, bordercolor=BDR, borderwidth=1, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=BDR, zerolinecolor=BDR, linecolor=BDR, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=BDR, zerolinecolor=BDR, linecolor=BDR, tickfont=dict(size=11))
    return fig


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(p: SamsungHealthParser, em: ExerciseMetric):
    s, e = "2026-02-15", "2026-03-15"

    hr = to_ist(p.get_heart_rate(s, e))
    hr["date"] = hr["start_time"].dt.date
    hr["hour"] = hr["start_time"].dt.hour
    hr["heart_rate"] = pd.to_numeric(hr["heart_rate"], errors="coerce")
    hr = tag_phase(hr)

    sleep = to_ist(p.get_sleep(s, e))
    sleep["date"] = sleep["start_time"].dt.date
    sleep["dur_min"] = (sleep["end_time"] - sleep["start_time"]).dt.total_seconds() / 60
    sleep = tag_phase(sleep)

    hrv = to_ist(p.get_hrv(s, e))
    hrv["date"] = hrv["start_time"].dt.date
    hrv["sdnn"]  = pd.to_numeric(hrv["sdnn"],  errors="coerce")
    hrv["rmssd"] = pd.to_numeric(hrv["rmssd"], errors="coerce")
    hrv = tag_phase(hrv)

    spo2 = to_ist(p.get_spo2(s, e))
    spo2["date"] = spo2["start_time"].dt.date
    spo2["spo2"] = pd.to_numeric(spo2["spo2"], errors="coerce")
    spo2 = tag_phase(spo2)

    rr = to_ist(p.get_respiratory_rate(s, e))
    rr["date"] = rr["start_time"].dt.date
    rr["average"] = pd.to_numeric(rr["average"], errors="coerce")
    rr = tag_phase(rr)

    ex = to_ist(p.get_exercise("2026-03-01", "2026-03-07"))
    ex["date"] = ex["start_time"].dt.date
    trek = ex[ex["exercise_type"].isin(TREK_TYPES)].copy()

    night_hr = hr[hr["hour"].isin(list(range(22, 24)) + list(range(0, 6)))].copy()

    return hr, sleep, hrv, spo2, rr, trek, night_hr


def load_gps_sessions(em: ExerciseMetric, trek: pd.DataFrame) -> list[dict]:
    bp = (trek[trek["exercise_type"] == BACKPACK_TYPE]
          .dropna(subset=["datauuid"])
          .sort_values("start_time")
          .reset_index(drop=True))

    d6 = (trek[(trek["exercise_type"] == 13001) &
               (trek["start_time"].dt.date == pd.Timestamp("2026-03-06").date())]
          .dropna(subset=["datauuid"]))

    rows = list(bp.iterrows()) + list(d6.iterrows())

    sessions, meta_idx = [], 0
    seen: set[str] = set()
    for _, row in rows:
        uuid = row["datauuid"]
        if uuid in seen:
            continue
        gps = em.load_run_locationdata(uuid)
        if gps.empty:
            continue
        seen.add(uuid)
        lbl, color = SESSION_META[meta_idx] if meta_idx < len(SESSION_META) else (f"Session {meta_idx+1}", MUT)
        meta_idx += 1
        step = max(1, len(gps) // 1200)
        sessions.append({
            "label":    lbl,
            "color":    color,
            "gps":      gps.iloc[::step].reset_index(drop=True),
            "gps_full": gps,
            "mean_hr":  row.get("mean_heart_rate"),
            "max_hr":   row.get("max_heart_rate"),
            "dist_km":  row.get("distance_km"),
            "dur_min":  row.get("duration_min"),
            "alt_gain": row.get("altitude_gain"),
            "calorie":  row.get("calorie"),
            "datauuid": uuid,
            "is_bp":    int(row["exercise_type"]) == BACKPACK_TYPE,
        })
    return sessions


# ── Section 1: MapLibre 3D Terrain Map ────────────────────────────────────────
# Returns raw HTML string (not Plotly) — uses template to avoid brace escaping

_MAP_TMPL = """\
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<div style="position:relative;">
<div id="trek-map3d" style="width:100%;height:700px;border-radius:0;"></div>
<div id="trek-legend" style="position:absolute;bottom:30px;left:12px;
  background:rgba(15,23,40,0.9);border:1px solid #1e2d45;border-radius:8px;
  padding:12px 14px;z-index:10;font-family:Inter,system-ui,sans-serif;"></div>
<div style="position:absolute;top:12px;right:12px;background:rgba(15,23,40,0.88);
  border:1px solid #1e2d45;border-radius:8px;padding:10px 14px;z-index:10;
  font-size:11px;color:#64748b;line-height:1.7;font-family:Inter,system-ui,sans-serif;">
  <b style="color:#e2e8f0">3D Terrain Map</b><br>
    Esri World Imagery · Satellite<br>Drag to rotate · Scroll to zoom
</div>
<button id="orbit-btn" style="position:absolute;bottom:30px;right:12px;
  background:#162035;border:1px solid #1e2d45;border-radius:6px;
  padding:7px 16px;color:#e2e8f0;cursor:pointer;font-size:12px;z-index:10;
  font-family:Inter,system-ui,sans-serif;">
  &#9654; Orbit (screen-record)
</button>
</div>
<script>
(function(){
var sessions = __SESSIONS__;
var centerLon = __LON__;
var centerLat = __LAT__;
var camps = [
  {name:"Lohajung<br>2,335 m",  lon:79.6039,lat:30.0823,c:"#95D5B2"},
  {name:"Bekaltal Camp<br>2,954 m",   lon:79.5980,lat:30.1388,c:"#F4D03F"},
  {name:"Brahmatal Camp<br>3,182 m",  lon:79.5877,lat:30.1591,c:"#E67E22"},
  {name:"Brahmatal Summit<br>3,734 m",lon:79.5931,lat:30.1789,c:"#C0392B"}
];

var map = new maplibregl.Map({
  container:"trek-map3d",
  style:{
    version:8,
    glyphs:"https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources:{
      "basemap":{
        type:"raster",
                tiles:["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
        tileSize:256,
                attribution:"Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        maxzoom:19
      },
      "terrain-src":{
        type:"raster-dem",
        tiles:["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
        tileSize:256,
        encoding:"terrarium",
        maxzoom:13
      }
    },
    layers:[{id:"basemap",type:"raster",source:"basemap"}],
    terrain:{source:"terrain-src",exaggeration:1.8},
    sky:{
      "sky-color":"#0a1a2e","sky-horizon-blend":0.4,
      "horizon-color":"#1e3a5f","horizon-fog-blend":0.3
    }
  },
  center:[centerLon,centerLat],
  zoom:11.2,pitch:62,bearing:-20,antialias:true
});

map.addControl(new maplibregl.NavigationControl(),"top-left");

map.on("load",function(){
  sessions.forEach(function(s){
    map.addSource(s.id,{
      type:"geojson",
      data:{type:"Feature",properties:{label:s.label},
            geometry:{type:"LineString",coordinates:s.coords}}
    });
    map.addLayer({id:s.id+"-glow",type:"line",source:s.id,
      layout:{"line-join":"round","line-cap":"round"},
      paint:{"line-color":s.color,"line-width":12,"line-opacity":0.15}});
    map.addLayer({id:s.id+"-line",type:"line",source:s.id,
      layout:{"line-join":"round","line-cap":"round"},
      paint:{"line-color":s.color,"line-width":3.5,"line-opacity":0.95}});
  });

  camps.forEach(function(c){
    var el=document.createElement("div");
    el.style.cssText="width:11px;height:11px;border-radius:50%;background:"+c.c+
      ";border:2px solid #fff;cursor:pointer;box-shadow:0 0 10px "+c.c;
    new maplibregl.Marker({element:el})
      .setLngLat([c.lon,c.lat])
      .setPopup(new maplibregl.Popup({offset:14})
        .setHTML('<div style="color:#fff;font-size:12px;font-weight:600;line-height:1.5">'+c.name+'</div>'))
      .addTo(map);
  });

  var leg=document.getElementById("trek-legend");
  sessions.forEach(function(s){
    var d=document.createElement("div");
    d.style.cssText="display:flex;align-items:center;gap:8px;font-size:11px;color:#e2e8f0;margin-bottom:5px";
    d.innerHTML='<div style="width:22px;height:3px;border-radius:2px;background:'+s.color+'"></div>'+s.label;
    leg.appendChild(d);
  });
});

var orbiting=false,orbitRaf=null,orbitBearing=map.getBearing();
function doOrbit(){orbitBearing=(orbitBearing+0.28)%360;map.setBearing(orbitBearing);orbitRaf=requestAnimationFrame(doOrbit);}
document.getElementById("orbit-btn").addEventListener("click",function(){
  if(!orbiting){orbiting=true;this.textContent="Pause orbit";doOrbit();}
  else{orbiting=false;this.textContent="\u25B6 Orbit (screen-record)";cancelAnimationFrame(orbitRaf);}
});
})();
</script>"""


def section_map3d(sessions: list[dict]) -> str:
    all_gps = pd.concat([s["gps"] for s in sessions], ignore_index=True)
    center_lon = float(all_gps.longitude.mean())
    center_lat = float(all_gps.latitude.mean())

    sess_data = []
    for s in sessions:
        g = s["gps"].dropna(subset=["latitude", "longitude"])
        step = max(1, len(g) // 500)
        coords = [[round(float(r.longitude), 6), round(float(r.latitude), 6)]
                  for _, r in g.iloc[::step].iterrows()]
        safe_id = "s" + str(len(sess_data))
        sess_data.append({"id": safe_id, "label": s["label"],
                          "color": s["color"], "coords": coords})

    html = _MAP_TMPL
    html = html.replace("__SESSIONS__", _json.dumps(sess_data))
    html = html.replace("__LON__", f"{center_lon:.5f}")
    html = html.replace("__LAT__", f"{center_lat:.5f}")
    return html


# ── Section 2: Continuous Elevation Timeline ───────────────────────────────────

def chart_elevation_timeline(sessions: list[dict]) -> go.Figure:
    fig = go.Figure()

    # Draw each session as a separate filled area on the shared IST time axis
    for s in sessions:
        g = (s["gps_full"]
             .dropna(subset=["altitude", "start_time"])
             .sort_values("start_time"))
        step = max(1, len(g) // 700)
        g = g.iloc[::step]
        ist_times = g["start_time"].dt.tz_convert(IST)
        fig.add_trace(go.Scatter(
            x=ist_times, y=g["altitude"],
            mode="lines", name=s["label"],
            line=dict(color=s["color"], width=2.5),
            fill="tozeroy", fillcolor=hex_rgba(s["color"], 0.06),
            hovertemplate="Alt: %{y:.0f} m  |  %{x|%b %d %H:%M}<extra>" + s["label"] + "</extra>",
        ))

    # Camp reference lines
    for name, elv in [("Bekaltal 2,954 m", 2954),
                       ("Brahmatal Camp 3,182 m", 3182),
                       ("Summit 3,734 m", 3734)]:
        fig.add_hline(y=elv, line_dash="dot", line_color="#2a3a4a", line_width=1,
                      annotation_text=name, annotation_position="right",
                      annotation_font=dict(color=MUT, size=10))

    # Milestone vertical markers — add_vline annotation is broken in Plotly 6 for
    # datetime axes, so use add_shape + add_annotation separately.
    y_positions = [0.98, 0.90, 0.82, 0.74, 0.66, 0.58, 0.50]  # stagger to avoid overlap
    for i, (ts_str, label, color) in enumerate(MILESTONES):
        ts_utc = pd.Timestamp(ts_str, tz=IST).tz_convert("UTC").isoformat()
        fig.add_shape(
            type="line", x0=ts_utc, x1=ts_utc,
            y0=0, y1=1, yref="paper", xref="x",
            line=dict(color=color, width=1.2, dash="dot"),
        )
        fig.add_annotation(
            x=ts_utc, y=y_positions[i % len(y_positions)],
            xref="x", yref="paper",
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(color=color, size=9),
            xanchor="left", yanchor="middle",
            bgcolor="rgba(8,13,24,0.7)", borderpad=2,
        )

    dark_fig(fig, "Trek Elevation Journey — Continuous IST Timeline", height=480)
    fig.update_yaxes(title_text="Altitude (m)", range=[2100, 3950])
    fig.update_xaxes(title_text="Date & Time (IST)", tickformat="%b %d\n%H:%M")
    return fig


# ── Section 3: Distribution Shifts ────────────────────────────────────────────

def chart_distributions(hr, night_hr, hrv, rr) -> go.Figure:
    KEY = ["baseline", "bekaltal", "brahmatal", "summit", "descent", "recovery"]
    phase_lookup = {p[0]: p for p in PHASES}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["All-Day Heart Rate (bpm)", "Sleeping HR — 22:00-06:00 (bpm)",
                        "HRV — SDNN (ms)", "Respiratory Rate (breaths / min)"],
        horizontal_spacing=0.14, vertical_spacing=0.22,
    )

    def _add_kde(df, col, row, c_idx, x_unit):
        for pid in KEY:
            if pid not in phase_lookup:
                continue
            _, lbl, *_, color, _ = phase_lookup[pid]
            vals = df.loc[df["phase"] == pid, col].dropna().values.astype(float)
            if len(vals) < 5:
                continue
            mu, sigma = float(vals.mean()), float(vals.std())
            bw = 0.45 if len(vals) < 40 else "scott"
            kde = stats.gaussian_kde(vals, bw_method=bw)
            span = max(3.5 * sigma, 3.0)
            x = np.linspace(mu - span, mu + span, 500)
            y = kde(x)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                name=lbl, legendgroup=pid,
                showlegend=(row == 1 and c_idx == 1),
                line=dict(color=color, width=2.5),
                fill="tozeroy", fillcolor=hex_rgba(color, 0.10),
                hovertemplate=(f"<b>{lbl}</b><br>{x_unit}: %{{x:.1f}}<br>"
                               f"mu={mu:.1f}  sigma={sigma:.1f}  n={len(vals)}<extra></extra>"),
            ), row=row, col=c_idx)
            fig.add_vline(x=mu, line_dash="dot", line_color=color,
                          line_width=1.5, row=row, col=c_idx)

    _add_kde(hr,       "heart_rate", 1, 1, "bpm")
    _add_kde(night_hr, "heart_rate", 1, 2, "bpm")
    _add_kde(hrv,      "sdnn",       2, 1, "ms")
    _add_kde(rr,       "average",    2, 2, "br/min")

    fig.update_layout(
        paper_bgcolor=SURF, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter,'Segoe UI',system-ui", size=12),
        height=760,
        title=dict(text="Physiological Distribution Shifts — Shift / Flatten / Narrow across Elevation Phases",
                   font=dict(size=15, color=TEXT), x=0.01),
        margin=dict(l=60, r=24, t=85, b=55),
        legend=dict(bgcolor=SUR2, bordercolor=BDR, borderwidth=1,
                    font=dict(size=11), title=dict(text="Phase", font=dict(color=MUT))),
    )
    fig.update_xaxes(gridcolor=BDR, zerolinecolor=BDR, linecolor=BDR, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=BDR, zerolinecolor=BDR, linecolor=BDR,
                     tickfont=dict(size=11), title_text="Density", tickformat=".3f")
    return fig


# ── Section 4: HR × Altitude — Oxygen Factor & Adaptation ────────────────────

def chart_hr_altitude_adaptation(em: ExerciseMetric, sessions: list[dict]) -> go.Figure:
    """
    Left panel: scatter HR vs altitude per session, bubble size = O2 availability,
                dashed trendline per session — shows cardiac load at elevation.
    Right panel: avg HR per 100 m altitude band, overlaid per session —
                 lower curve in a later session = body adapted.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "HR vs Altitude  (bubble = O2 availability at that height)",
            "Adaptation Index — Avg HR per 100 m band by Session",
        ],
        horizontal_spacing=0.13,
    )

    adapt_rows: list[dict] = []

    for s in sessions:
        # Full-res GPS for altitude timestamps
        gps_t = (s["gps_full"]
                 .dropna(subset=["altitude", "start_time"])
                 .sort_values("start_time")
                 .reset_index(drop=True))

        live = em.load_run_livedata(s["datauuid"])
        if live.empty or "heart_rate" not in live.columns:
            continue
        live = (live.dropna(subset=["heart_rate", "start_time"])
                    .sort_values("start_time")
                    .reset_index(drop=True))

        # Nearest-timestamp merge: GPS altitude → live HR
        merged = pd.merge_asof(
            live[["start_time", "heart_rate"]],
            gps_t[["start_time", "altitude"]],
            on="start_time", direction="nearest",
            tolerance=pd.Timedelta("60s"),
        ).dropna(subset=["altitude", "heart_rate"])

        if len(merged) < 20:
            continue

        # Downsample for scatter
        step = max(1, len(merged) // 500)
        m = merged.iloc[::step].copy()
        m["o2_pct"] = m["altitude"].apply(lambda a: o2_frac(a) * 100)
        # Bubble size scaled: higher O2 = bigger bubble (ironic — more O2 = less stress)
        m["bsize"] = ((m["o2_pct"] - 55) / 20).clip(3, 18)

        # Left panel scatter
        fig.add_trace(go.Scatter(
            x=m["altitude"], y=m["heart_rate"],
            mode="markers", name=s["label"],
            legendgroup=s["label"],
            marker=dict(color=s["color"], size=m["bsize"],
                        opacity=0.45, line=dict(width=0)),
            customdata=m["o2_pct"],
            hovertemplate=(f"<b>{s['label']}</b><br>"
                           "Alt: %{x:.0f} m<br>HR: %{y:.0f} bpm<br>"
                           "O2 avail: %{customdata:.1f}%<extra></extra>"),
        ), row=1, col=1)

        # Trendline
        if len(m) >= 10:
            sl, ic, *_ = stats.linregress(m["altitude"], m["heart_rate"])
            xf = np.linspace(m["altitude"].min(), m["altitude"].max(), 80)
            yf = sl * xf + ic
            fig.add_trace(go.Scatter(
                x=xf, y=yf, mode="lines",
                name=s["label"] + " trend",
                legendgroup=s["label"], showlegend=False,
                line=dict(color=s["color"], width=2.2, dash="dash"),
            ), row=1, col=1)

        # Right panel: bin by 100 m altitude bands
        merged["alt_band"] = (merged["altitude"] // 100) * 100
        for band, grp in merged.groupby("alt_band"):
            if len(grp) >= 3:
                adapt_rows.append({
                    "session": s["label"],
                    "color":   s["color"],
                    "alt_band": float(band),
                    "mean_hr":  float(grp["heart_rate"].mean()),
                })

    # Right panel: adaptation curves
    if adapt_rows:
        adapt_df = pd.DataFrame(adapt_rows)
        # Only keep altitude bands shared by >= 2 sessions
        shared = adapt_df.groupby("alt_band")["session"].nunique()
        shared_bands = shared[shared >= 2].index
        adapt_df = adapt_df[adapt_df["alt_band"].isin(shared_bands)]

        for lbl in adapt_df["session"].unique():
            sub = adapt_df[adapt_df["session"] == lbl].sort_values("alt_band")
            if sub.empty:
                continue
            color = sub["color"].iloc[0]
            fig.add_trace(go.Scatter(
                x=sub["alt_band"], y=sub["mean_hr"],
                mode="lines+markers", name=lbl,
                legendgroup=lbl, showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                hovertemplate=(f"<b>{lbl}</b><br>"
                               "Band: %{x:.0f}-" + "%{x:.0f}" + " m<br>"
                               "Avg HR: %{y:.1f} bpm<extra></extra>"),
            ), row=1, col=2)

    # O2 legend annotation
    fig.add_annotation(
        x=0.49, y=-0.12, xref="paper", yref="paper",
        text="Bubble size = O2 availability at altitude  |  Smaller bubble = thinner air = more cardiac strain",
        font=dict(size=10, color=MUT), showarrow=False, align="center",
    )

    dark_fig(fig, "HR-Altitude Cardiac Adaptation — Did the Body Learn to Work Smarter?", height=520)
    fig.update_xaxes(title_text="Altitude (m)", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=1, col=1)
    fig.update_xaxes(title_text="Altitude Band (m)", row=1, col=2)
    fig.update_yaxes(title_text="Avg HR (bpm)", row=1, col=2)
    return fig


# ── Section 5: SpO2 Arc ────────────────────────────────────────────────────────

def chart_spo2_arc(spo2: pd.DataFrame) -> go.Figure:
    daily = (spo2.groupby("date")
             .agg(spo2_mean=("spo2", "mean"), spo2_min=("spo2", "min"))
             .reset_index())
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["p_color"] = daily["date"].map(
        lambda d: next((p[4] for p in PHASES
                        if pd.Timestamp(p[2]).date() <= d <= pd.Timestamp(p[3]).date()), MUT))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["SpO2 (%)", "Camp Elevation (m)"],
                        row_heights=[0.68, 0.32], vertical_spacing=0.06)

    fig.add_trace(go.Scatter(
        x=daily["date_dt"], y=daily["spo2_mean"],
        mode="lines+markers", name="SpO2",
        line=dict(color="#60a5fa", width=2.5),
        marker=dict(size=9, color=daily["p_color"],
                    line=dict(color="#0f1728", width=1.5)),
        fill="tozeroy", fillcolor=hex_rgba("#60a5fa", 0.06),
        hovertemplate="<b>SpO2:</b> %{y:.1f}%  --  %{x|%b %d}<extra></extra>",
    ), row=1, col=1)
    fig.add_hrect(y0=80, y1=94, fillcolor=hex_rgba("#f87171", 0.06), line_width=0, row=1, col=1)
    fig.add_hline(y=94, line_dash="dot", line_color="#f87171", line_width=1,
                  annotation_text="94% -- clinical alert", annotation_position="bottom right",
                  annotation_font=dict(color="#f87171", size=10), row=1, col=1)

    trek_dates = [pd.Timestamp(p[2]) for p in PHASES[1:6]]
    fig.add_trace(go.Bar(
        x=trek_dates, y=[p[5] for p in PHASES[1:6]],
        name="Camp Elevation",
        marker_color=[p[4] for p in PHASES[1:6]],
        customdata=[p[1] for p in PHASES[1:6]],
        hovertemplate="<b>%{customdata}</b><br>%{y:,} m<extra></extra>",
    ), row=2, col=1)

    dark_fig(fig, "SpO2 Arc -- Blood Oxygen Drop at Altitude & Recovery", height=500)
    fig.update_yaxes(title_text="SpO2 (%)", range=[80, 101], row=1, col=1)
    fig.update_yaxes(title_text="Elevation (m)", row=2, col=1)
    return fig


# ── Section 6: Sleeping HR ────────────────────────────────────────────────────

def chart_sleep_hr(hr: pd.DataFrame) -> go.Figure:
    night = hr[hr["hour"].isin(list(range(22, 24)) + list(range(0, 6)))].copy()
    daily = (night.groupby(["date", "phase_label", "phase_color", "phase_elev_m"])
             .agg(sleep_hr=("heart_rate", "mean")).reset_index())
    daily["date_dt"] = pd.to_datetime(daily["date"])
    baseline_avg = float(night.loc[night["phase"] == "baseline", "heart_rate"].mean())

    fig = go.Figure()
    fig.add_vrect(x0="2026-03-01", x1="2026-03-07",
                  fillcolor=hex_rgba("#E67E22", 0.07), line_width=0,
                  annotation_text="Trek", annotation_position="top left",
                  annotation_font=dict(color=TEXT, size=10))
    fig.add_trace(go.Scatter(
        x=daily["date_dt"], y=daily["sleep_hr"],
        mode="lines+markers", name="Sleeping HR",
        line=dict(color="#f87171", width=2.5),
        marker=dict(size=9, color=daily["phase_color"],
                    line=dict(color="#0f1728", width=1.5)),
        hovertemplate="<b>%{x|%b %d}</b><br>Sleeping HR: %{y:.1f} bpm<extra></extra>",
    ))
    fig.add_hline(y=baseline_avg, line_dash="dot", line_color="#52B788", line_width=1.5,
                  annotation_text=f"Baseline avg {baseline_avg:.0f} bpm",
                  annotation_font=dict(color="#52B788", size=10),
                  annotation_position="bottom right")
    peak_row = daily.loc[daily["sleep_hr"].idxmax()]
    fig.add_annotation(
        x=peak_row["date_dt"], y=peak_row["sleep_hr"],
        text=f"Peak {peak_row['sleep_hr']:.0f} bpm<br>{peak_row['phase_label']}",
        showarrow=True, arrowhead=2, arrowcolor="#f87171",
        font=dict(color="#f87171", size=10), ax=30, ay=-45,
    )
    dark_fig(fig, "Sleeping HR -- Nocturnal Tachycardia Driven by Low SpO2", height=400)
    fig.update_yaxes(title_text="Avg HR during sleep (bpm)")
    fig.update_xaxes(title_text="Date (IST)")
    return fig


# ── Section 7: HRV Arc ────────────────────────────────────────────────────────

def chart_hrv_arc(hrv: pd.DataFrame) -> go.Figure:
    daily = (hrv.groupby("date")
             .agg(sdnn=("sdnn", "mean"), rmssd=("rmssd", "mean"))
             .reset_index())
    daily["date_dt"] = pd.to_datetime(daily["date"])

    fig = go.Figure()
    fig.add_vrect(x0="2026-03-01", x1="2026-03-07",
                  fillcolor=hex_rgba("#E67E22", 0.07), line_width=0,
                  annotation_text="Trek", annotation_position="top left",
                  annotation_font=dict(color=TEXT, size=10))
    for col, color, name in [("sdnn", "#fbbf24", "SDNN"), ("rmssd", "#a78bfa", "RMSSD")]:
        fig.add_trace(go.Scatter(
            x=daily["date_dt"], y=daily[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{name}:</b> %{{y:.1f}} ms  --  %{{x|%b %d}}<extra></extra>",
        ))
    fig.add_annotation(
        x="2026-03-04", y=85,
        text="Peak physiological stress<br>SDNN 85 ms  (-35% vs baseline)",
        showarrow=True, arrowhead=2, arrowcolor="#f87171",
        font=dict(color="#f87171", size=10), ax=50, ay=-55,
    )
    fig.add_annotation(
        x="2026-03-09", y=172,
        text="Supercompensation rebound<br>SDNN 172 ms  (+32% above baseline)",
        showarrow=True, arrowhead=2, arrowcolor="#34d399",
        font=dict(color="#34d399", size=10), ax=-50, ay=-55,
    )
    dark_fig(fig, "HRV -- Stress Dip During Trek & Supercompensation Rebound After", height=440)
    fig.update_yaxes(title_text="HRV (ms)")
    fig.update_xaxes(title_text="Date (IST)")
    return fig


# ── Section 8: Sleep Quality ──────────────────────────────────────────────────

def chart_sleep_quality(sleep: pd.DataFrame) -> go.Figure:
    STAGES  = ["Deep", "REM", "Light", "Awake"]
    SCOLORS = {"Deep": "#2980B9", "REM": "#8E44AD", "Light": "#16A085", "Awake": "#f87171"}

    daily = (sleep.groupby(["date", "stage_label"])
             .agg(mins=("dur_min", "sum")).reset_index())
    daily["date_dt"] = pd.to_datetime(daily["date"])
    pivot = daily.pivot(index="date_dt", columns="stage_label", values="mins").fillna(0)
    for col in STAGES:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["total"] = pivot[STAGES].sum(axis=1)
    pivot["efficiency"] = (pivot["total"] - pivot["Awake"]) / pivot["total"] * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Sleep Stage Minutes", "Sleep Efficiency (%)"],
                        row_heights=[0.65, 0.35], vertical_spacing=0.07)
    for stage in STAGES:
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[stage], name=stage,
            marker_color=SCOLORS[stage],
            hovertemplate=f"<b>{stage}:</b> %{{y:.0f}} min<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pivot.index, y=pivot["efficiency"],
        mode="lines+markers", name="Efficiency %",
        line=dict(color="#fbbf24", width=2.5), marker=dict(size=6),
        hovertemplate="Efficiency: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    for r in [1, 2]:
        fig.add_vrect(x0="2026-03-01", x1="2026-03-07",
                      fillcolor=hex_rgba("#E67E22", 0.07), line_width=0,
                      row=r, col=1)
    fig.update_layout(barmode="stack", paper_bgcolor=SURF, plot_bgcolor=BG,
                      font=dict(color=TEXT, family="Inter,'Segoe UI',system-ui", size=12),
                      height=540,
                      title=dict(text="Sleep Quality -- REM Suppression & Altitude Insomnia",
                                 font=dict(size=15, color=TEXT), x=0.01),
                      margin=dict(l=60, r=24, t=60, b=50),
                      legend=dict(bgcolor=SUR2, bordercolor=BDR, borderwidth=1, font=dict(size=11)))
    fig.update_xaxes(gridcolor=BDR, zerolinecolor=BDR, linecolor=BDR)
    fig.update_yaxes(gridcolor=BDR, zerolinecolor=BDR, linecolor=BDR)
    fig.update_yaxes(title_text="Minutes", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", range=[60, 102], row=2, col=1)
    return fig


# ── Section 9: Live HR per Session ────────────────────────────────────────────

def chart_live_hr(em: ExerciseMetric, sessions: list[dict]) -> go.Figure:
    fig = go.Figure()
    for s in sessions:
        live = em.load_run_livedata(s["datauuid"])
        if live.empty or "heart_rate" not in live.columns:
            continue
        live = live.dropna(subset=["heart_rate"])
        if live.empty:
            continue
        step = max(1, len(live) // 800)
        live = live.iloc[::step]
        fig.add_trace(go.Scatter(
            x=live["elapsed_min"], y=live["heart_rate"],
            mode="lines", name=s["label"],
            line=dict(color=s["color"], width=2.0),
            hovertemplate="HR: %{y:.0f} bpm  |  T+%{x:.0f} min<extra>" + s["label"] + "</extra>",
        ))
    extra_pct = PACK_KG / BODY_KG * 100
    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text=(f"Backpack approx {PACK_KG} kg  ->  +{extra_pct:.0f}% effective load"
              f"  ->  ~+{extra_pct*0.15:.0f}% calorie burn on backpacking sessions"),
        font=dict(size=11, color=MUT), showarrow=False, align="left",
        bgcolor=SUR2, bordercolor=BDR, borderwidth=1, borderpad=6,
    )
    dark_fig(fig, "Live Heart Rate per Trek Session", height=440)
    fig.update_yaxes(title_text="Heart Rate (bpm)")
    fig.update_xaxes(title_text="Elapsed Time (min)")
    return fig


# ── HTML Assembly ─────────────────────────────────────────────────────────────

KPI_CARDS = [
    ("Trek Route",        "Lohajung - Bekaltal - Brahmatal", "Uttarakhand, India"),
    ("Dates",             "Mar 1-7, 2026",                   "7 days"),
    ("Max Altitude",      "3,734 m  (12,250 ft)",            "Brahmatal Summit"),
    ("Trek Distance",     "approx 40 km",                    "GPS-derived"),
    ("Elevation Gain",    "approx 1,900 m",                  "backpacking sessions"),
    ("SpO2 at Summit",    "85-87 %",                         "-12 pts vs baseline 97%"),
    ("O2 Availability",   "approx 63%",                      "of sea-level at 3,734 m"),
    ("Sleeping HR Peak",  "115 bpm",                         "vs 67 bpm baseline"),
    ("HRV Stress Low",    "SDNN 85 ms",                      "-35% vs baseline 130 ms"),
    ("HRV Rebound",       "SDNN 172 ms",                     "+32% above baseline Mar 9"),
    ("Sleep Eff. Low",    "79.9%  (Mar 4)",                  "vs 93% baseline"),
    ("Body Optimization", "approx 80%",                      "peak altitude composite"),
]

# Sections: (html_id, nav_label, content_key)
SECTIONS = [
    ("map3d",         "3D Terrain Map",          "map3d"),
    ("elevation",     "Elevation Timeline",      "elevation"),
    ("distributions", "Distribution Analysis",   "distributions"),
    ("hr_adaptation", "HR-Altitude Adaptation",  "hr_adaptation"),
    ("spo2",          "SpO2 Arc",                "spo2"),
    ("sleep_hr",      "Sleeping HR",             "sleep_hr"),
    ("hrv",           "HRV Arc",                 "hrv"),
    ("sleep",         "Sleep Quality",           "sleep"),
    ("live_hr",       "Live Trek HR",            "live_hr"),
]

_PLOTLYJS_URL = "https://cdn.plot.ly/plotly-2.32.0.min.js"
_DIV_CTR = 0


def _render_fig(fig: go.Figure, emit_cdn: bool) -> str:
    global _DIV_CTR
    fig_dict  = _json.loads(fig.to_json())
    frames    = fig_dict.get("frames", [])
    cdn_tag   = f'<script src="{_PLOTLYJS_URL}" charset="utf-8"></script>\n' if emit_cdn else ""

    if not frames:
        incl = False
        return cdn_tag + pio.to_html(
            fig, full_html=False, include_plotlyjs=incl,
            config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
        )

    _DIV_CTR += 1
    did   = f"plotly-anim-{_DIV_CTR}"
    h     = fig_dict.get("layout", {}).get("height", 700)
    return (
        cdn_tag
        + f'<div id="{did}" style="width:100%;height:{h}px;"></div>\n'
        + "<script>\n(function(){\n"
        + f"  var data={_json.dumps(fig_dict.get('data',[]))};\n"
        + f"  var layout={_json.dumps(fig_dict.get('layout',{}))};\n"
        + f"  var frames={_json.dumps(frames)};\n"
        + f"  var cfg={{displayModeBar:true,scrollZoom:true,responsive:true}};\n"
        + f"  Plotly.newPlot('{did}',data,layout,cfg).then(function(){{\n"
        + f"    Plotly.addFrames('{did}',frames);\n  }});\n"
        + "})();\n</script>"
    )


def build_html(content: dict[str, "go.Figure | str"]) -> str:
    rendered: dict[str, str] = {}
    plotly_cdn_emitted = False
    for key, item in content.items():
        if isinstance(item, str):
            rendered[key] = item
        else:
            rendered[key] = _render_fig(item, emit_cdn=not plotly_cdn_emitted)
            plotly_cdn_emitted = True

    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-val">{val}</div>'
        f'<div class="kpi-hint">{hint}</div></div>'
        for lbl, val, hint in KPI_CARDS
    )
    nav_html = "".join(
        f'<a href="#{sid}" class="nav-link">{name}</a>'
        for sid, name, _ in SECTIONS
    )
    sections_html = "".join(
        f'<section id="{sid}" class="section">'
        f'<div class="section-title">{name}</div>'
        f'<div class="chart-wrap">{rendered[key]}</div>'
        f'</section>'
        for sid, name, key in SECTIONS
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Brahmatal Trek -- Health Dashboard -- March 2026</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:{BG};--surf:{SURF};--sur2:{SUR2};--bdr:{BDR};
  --text:{TEXT};--mut:{MUT};
  --blue:#60a5fa;--green:#34d399;--orange:#fbbf24;--red:#f87171;}}
body{{background:var(--bg);color:var(--text);font-family:Inter,'Segoe UI',system-ui,sans-serif;font-size:13px;line-height:1.5;min-height:100vh}}
#app{{max-width:1500px;margin:0 auto;padding:28px 24px 80px}}
.header{{padding-bottom:18px;margin-bottom:20px;border-bottom:1px solid var(--bdr)}}
.header h1{{font-size:24px;font-weight:700;letter-spacing:-.4px;color:#fff}}
.header .sub{{color:var(--mut);font-size:12px;margin-top:5px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));gap:10px;margin-bottom:22px}}
.kpi{{background:var(--surf);border:1px solid var(--bdr);border-radius:10px;padding:14px 16px}}
.kpi-label{{color:var(--mut);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}}
.kpi-val{{font-size:18px;font-weight:700;color:#fff;line-height:1.15}}
.kpi-hint{{color:var(--mut);font-size:10px;margin-top:3px}}
nav{{position:sticky;top:0;z-index:100;background:rgba(8,13,24,0.92);backdrop-filter:blur(12px);
  padding:9px 0;border-bottom:1px solid var(--bdr);display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0}}
.nav-link{{color:var(--mut);text-decoration:none;font-size:12px;padding:4px 12px;
  border-radius:20px;border:1px solid var(--bdr);transition:all .15s;white-space:nowrap}}
.nav-link:hover{{color:#fff;border-color:#475569;background:var(--sur2)}}
.section{{padding:28px 0;border-bottom:1px solid var(--bdr)}}
.section-title{{font-size:10px;font-weight:600;color:var(--mut);text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:14px}}
.chart-wrap{{border-radius:10px;overflow:hidden;border:1px solid var(--bdr)}}
.maplibregl-popup-content{{background:#0f1728!important;color:#e2e8f0!important;
  border:1px solid #1e2d45!important;border-radius:8px!important;padding:10px 14px!important}}
.maplibregl-popup-tip{{border-top-color:#1e2d45!important}}
</style>
</head>
<body>
<div id="app">
<div class="header">
  <h1>Brahmatal Trek -- Health Dashboard</h1>
  <div class="sub">March 1-7, 2026  &middot;  Uttarakhand, India  &middot;  Lohajung - Bekaltal - Brahmatal Summit (3,734 m / 12,250 ft)  &middot;  Samsung Galaxy Watch</div>
</div>
<div class="kpi-grid">{kpi_html}</div>
<nav>{nav_html}</nav>
{sections_html}
</div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Initialising parser...")
    p  = SamsungHealthParser(DATA_DIR)
    em = ExerciseMetric(DATA_DIR)

    print("Loading health data...")
    hr, sleep, hrv, spo2, rr, trek, night_hr = load_data(p, em)

    print("Loading GPS sessions...")
    sessions = load_gps_sessions(em, trek)
    print(f"  {len(sessions)} GPS sessions loaded")
    for s in sessions:
        g = s["gps_full"]
        print(f"  {s['label']}: {len(g)} pts  alt {g.altitude.min():.0f}-{g.altitude.max():.0f} m")

    print("Building sections...")
    content: dict[str, "go.Figure | str"] = {
        "map3d":         section_map3d(sessions),
        "elevation":     chart_elevation_timeline(sessions),
        "distributions": chart_distributions(hr, night_hr, hrv, rr),
        "hr_adaptation": chart_hr_altitude_adaptation(em, sessions),
        "spo2":          chart_spo2_arc(spo2),
        "sleep_hr":      chart_sleep_hr(hr),
        "hrv":           chart_hrv_arc(hrv),
        "sleep":         chart_sleep_quality(sleep),
        "live_hr":       chart_live_hr(em, sessions),
    }

    print("Assembling HTML...")
    html = build_html(content)
    OUTPUT.write_text(html, encoding="utf-8")
    size_mb = OUTPUT.stat().st_size / 1_048_576
    print(f"Done -> {OUTPUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
