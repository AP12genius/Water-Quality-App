"""Streamlit WQI Forecaster — premium dark-mode UI."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT              = Path(__file__).parent
MODEL_PATH        = ROOT / "models" / "rf.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"

FEATURE_COLUMNS = [
    "WQI","pH","DO","BOD","TC","FC","Nitrate",
    "DO_BOD_ratio","pH_deviation","log_fecal_coliform","log_total_coliform",
]

QUALITY_BANDS = [
    (90, "Excellent",  "#10b981", "#064e3b"),
    (70, "Good",       "#3b82f6", "#1e3a5f"),
    (50, "Medium",     "#f59e0b", "#451a03"),
    (25, "Poor",       "#f97316", "#431407"),
    ( 0, "Very Poor",  "#ef4444", "#450a0a"),
]

QUALITY_EMOJI = {
    "Excellent": "🟢",
    "Good": "🔵",
    "Medium": "🟡",
    "Poor": "🟠",
    "Very Poor": "🔴",
}

def compute_wqi(pH, DO, BOD, FC, Nitrate):
    q_do      = float(np.clip(100 - (10 - DO) * 10,                         0, 100))
    q_ph      = float(np.clip(100 - abs(pH - 7.0) * 20,                     0, 100))
    q_bod     = float(np.clip(100 - BOD * 12,                                0, 100))
    q_fc      = float(np.clip(100 - np.log1p(FC) / np.log1p(10_000) * 100,  0, 100))
    q_nitrate = float(np.clip(100 - Nitrate * 15,                            0, 100))
    return 0.31*q_do + 0.11*q_ph + 0.19*q_bod + 0.28*q_fc + 0.11*q_nitrate

def build_feature_row(pH, DO, BOD, TC, FC, Nitrate):
    wqi = compute_wqi(pH, DO, BOD, FC, Nitrate)
    row = {
        "WQI": wqi, "pH": pH, "DO": DO, "BOD": BOD,
        "TC": TC, "FC": FC, "Nitrate": Nitrate,
        "DO_BOD_ratio":       DO / (BOD + 1e-5),
        "pH_deviation":       abs(pH - 7.0),
        "log_fecal_coliform": np.log1p(max(FC, 0)),
        "log_total_coliform": np.log1p(max(TC, 0)),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]

def quality_label(wqi):
    for threshold, label, color, bg in QUALITY_BANDS:
        if wqi >= threshold:
            return label, color, bg
    last = QUALITY_BANDS[-1]
    return last[1], last[2], last[3]

@st.cache_resource(show_spinner=False)
def load_artifacts():
    for path, name in [(MODEL_PATH,"models/rf.joblib"),(PREPROCESSOR_PATH,"models/preprocessor.joblib")]:
        if not path.exists():
            return None, None, f"**File not found:** `{name}`\n\nRun `python save_models.py` first."
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH), None

st.set_page_config(
    page_title="WQI Forecaster",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Force entire app dark */
.stApp {
    background: #050c1a !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(14,165,233,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(79,70,229,0.12) 0%, transparent 60%) !important;
    min-height: 100vh;
}

/* Kill the white Streamlit toolbar */
header[data-testid="stHeader"] {
    background: rgba(5,12,26,0.85) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* Main block padding */
.block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
}

/* ── Hero ─────────────────────────────────────────────── */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 3.5rem 2rem 3rem;
    text-align: center;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg,
        rgba(14,165,233,0.15) 0%,
        rgba(99,102,241,0.15) 50%,
        rgba(168,85,247,0.12) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
        rgba(14,165,233,0.08),
        rgba(99,102,241,0.08));
    border-radius: inherit;
}
.hero-glow {
    position: absolute;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: rgba(14,165,233,0.15);
    filter: blur(80px);
    top: -100px; left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
}
.hero-icon { font-size: 4rem; line-height: 1; display: block; margin-bottom: 1rem; }
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0 0 0.75rem;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero h1 span {
    background: linear-gradient(135deg, #0ea5e9, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    margin-top: 1.4rem;
    flex-wrap: wrap;
}
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.9rem;
    border-radius: 99px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
}

/* ── Section label ────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1.2rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Input card ───────────────────────────────────────── */
.input-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1.5rem;
    position: relative;
}
.input-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg,
        rgba(14,165,233,0.04) 0%,
        transparent 60%);
    pointer-events: none;
}

/* ── Streamlit widget overrides — nuclear specificity ─── */

/* Labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] p,
div[data-testid="stNumberInput"] > label > div {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* The actual input — target ALL possible emotion-cache wrappers */
div[data-testid="stNumberInput"] input,
div[data-testid="stNumberInput"] input[type="number"],
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
.stNumberInput input,
input[type="number"] {
    background: #0d1526 !important;
    background-color: #0d1526 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    caret-color: #0ea5e9 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}

/* BaseWeb Input wrapper (the div around the input) */
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background: #0d1526 !important;
    background-color: #0d1526 !important;
    border-radius: 12px !important;
}

/* Focus ring */
div[data-testid="stNumberInput"] input:focus,
div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.2) !important;
    outline: none !important;
}

/* +/- stepper buttons */
div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInput"] button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #64748b !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
}

/* autocomplete / browser autofill override */
input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 100px #0d1526 inset !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    caret-color: #e2e8f0 !important;
}

/* ── Predict button ───────────────────────────────────── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 60%, #8b5cf6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 8px 32px rgba(14,165,233,0.25), 0 0 0 1px rgba(255,255,255,0.05) !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Expander ─────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin-top: 0.8rem !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Dataframe ────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    background: transparent !important;
}

/* ── Tooltip ──────────────────────────────────────────── */
div[data-testid="stTooltipIcon"] { color: #334155 !important; }

/* ── Result cards ─────────────────────────────────────── */
.results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
}
.result-card {
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}
.result-card::after {
    content: '';
    position: absolute;
    top: -40%;
    right: -20%;
    width: 200px; height: 200px;
    border-radius: 50%;
    opacity: 0.15;
    filter: blur(40px);
}
.result-tag {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.75rem;
}
.result-score {
    font-size: 4.5rem;
    font-weight: 900;
    letter-spacing: -3px;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
}
.result-quality {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid;
}

.delta-card {
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(15,23,42,0.6);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}
.delta-value {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -2px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin: 0.5rem 0;
}
.delta-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
}
.delta-desc {
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.5rem;
    padding: 0.25rem 0.8rem;
    border-radius: 99px;
    background: rgba(255,255,255,0.06);
}

/* ── WQI sub-scores ───────────────────────────────────── */
.scores-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}
.score-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 0.6rem;
    text-align: center;
}
.score-chip .chip-val {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
    line-height: 1;
}
.score-chip .chip-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    margin-top: 0.35rem;
}
.score-chip .chip-weight {
    font-size: 0.6rem;
    color: #334155;
    margin-top: 0.15rem;
}

/* ── Gauge ────────────────────────────────────────────── */
.gauge-section { margin: 1.5rem 0; }
.gauge-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
}
.gauge-outer {
    height: 12px;
    border-radius: 99px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
    position: relative;
}
.gauge-inner {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg,
        #ef4444 0%, #f97316 25%, #f59e0b 45%,
        #22c55e 70%, #10b981 100%);
    position: relative;
    transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
}
.gauge-inner::after {
    content: '';
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 4px;
    background: rgba(255,255,255,0.6);
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(255,255,255,0.4);
}
.gauge-ticks {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
}
.gauge-tick {
    font-size: 0.62rem;
    color: #334155;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.15rem;
}
.gauge-tick-line {
    width: 1px; height: 6px;
    background: rgba(255,255,255,0.12);
}

/* ── Quality legend ───────────────────────────────────── */
.quality-legend {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.6rem;
    margin-top: 0.8rem;
}
.legend-item {
    border-radius: 10px;
    padding: 0.7rem 0.5rem;
    text-align: center;
    border: 1px solid;
}
.legend-item .leg-range {
    font-size: 0.62rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.2rem;
    opacity: 0.7;
}
.legend-item .leg-label {
    font-size: 0.72rem;
    font-weight: 700;
}

/* ── Feature table ────────────────────────────────────── */
.feat-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.feat-table th {
    text-align: left;
    padding: 0.5rem 0.8rem;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #334155;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.feat-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
.feat-table tr:last-child td { border-bottom: none; }
.feat-table tr:hover td { background: rgba(255,255,255,0.02); }
.feat-computed { color: #6366f1 !important; }

/* ── Footer ───────────────────────────────────────────── */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    color: #1e293b;
    font-size: 0.75rem;
}
.footer code {
    background: rgba(255,255,255,0.05);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.05) !important; }
</style>
""", unsafe_allow_html=True)

# JS-powered late override — fires after Streamlit emotion-cache styles mount
# Uses MutationObserver so it re-applies whenever Streamlit hot-reloads the DOM
st.markdown("""
<script>
(function() {
  const STYLE_ID = 'wqi-input-override';
  const CSS = `
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {
      background: #0d1526 !important;
      background-color: #0d1526 !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    input[type="number"] {
      background: #0d1526 !important;
      background-color: #0d1526 !important;
      color: #e2e8f0 !important;
      -webkit-text-fill-color: #e2e8f0 !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-weight: 600 !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {
      border-color: #0ea5e9 !important;
      box-shadow: 0 0 0 3px rgba(14,165,233,0.2) !important;
    }
  `;
  function inject() {
    if (!document.getElementById(STYLE_ID)) {
      const s = document.createElement('style');
      s.id = STYLE_ID;
      s.textContent = CSS;
      document.head.appendChild(s);
    }
  }
  inject();
  new MutationObserver(inject).observe(document.head, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <div class="hero-glow"></div>
  <span class="hero-icon">💧</span>
  <h1>WQI <span>Forecaster</span></h1>
  <p class="hero-sub">Enter this station's six readings and get a one-year-ahead
     Water Quality Index prediction powered by a Random Forest model.</p>
  <div class="hero-badges">
    <span class="badge-pill">🌊 Random Forest</span>
    <span class="badge-pill">📐 11 Engineered Features</span>
    <span class="badge-pill">📅 One-Year-Ahead</span>
    <span class="badge-pill">🔬 5-Parameter WQI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    rf, preprocessor, load_error = load_artifacts()

if load_error:
    st.error(load_error)
    st.stop()

# ── Inputs ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="input-card">
  <div class="section-label">📊 Station Readings — Current Year</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    pH      = st.number_input("pH",              min_value=0.0,  max_value=14.0,   value=7.4,   step=0.1,  format="%.2f", help="pH (0-14)")
with col2:
    DO      = st.number_input("DO mg/L",         min_value=0.0,  max_value=20.0,   value=8.0,   step=0.1,  format="%.2f", help="Dissolved Oxygen")
with col3:
    BOD     = st.number_input("BOD mg/L",        min_value=0.0,  max_value=100.0,  value=2.5,   step=0.1,  format="%.2f", help="Biochemical Oxygen Demand")
with col4:
    TC      = st.number_input("TC MPN/100mL",    min_value=0.0,  max_value=1e6,    value=500.0, step=10.0, format="%.0f", help="Total Coliform")
with col5:
    FC      = st.number_input("FC MPN/100mL",    min_value=0.0,  max_value=1e6,    value=100.0, step=10.0, format="%.0f", help="Fecal Coliform")
with col6:
    Nitrate = st.number_input("Nitrate mg/L",    min_value=0.0,  max_value=100.0,  value=1.5,   step=0.1,  format="%.2f", help="Nitrate concentration")

st.markdown("</div>", unsafe_allow_html=True)

# Live current WQI (always visible, updates with sliders)
live_wqi = compute_wqi(pH, DO, BOD, FC, Nitrate)
live_label, live_color, live_bg = quality_label(live_wqi)
live_emoji = QUALITY_EMOJI[live_label]

st.markdown(f"""
<div style="display:flex;align-items:center;gap:1.2rem;
            background:rgba(14,165,233,0.07);
            border:1px solid rgba(14,165,233,0.18);
            border-radius:16px;padding:1.1rem 1.5rem;margin-bottom:1.5rem;">
  <div style="font-size:2.8rem;font-weight:900;color:#0ea5e9;
              font-family:'JetBrains Mono',monospace;letter-spacing:-2px;line-height:1;">
    {live_wqi:.1f}
  </div>
  <div style="flex:1;">
    <div style="font-weight:700;color:#e2e8f0;font-size:0.95rem;">
      Current-Year WQI
      <span style="color:{live_color};margin-left:0.6rem;
                   font-size:0.8rem;background:{live_bg};
                   padding:0.15rem 0.6rem;border-radius:99px;
                   border:1px solid {live_color}55;">
        {live_emoji} {live_label}
      </span>
    </div>
    <div style="font-size:0.8rem;color:#475569;margin-top:0.25rem;">
      Persistence forecast: next-year WQI ≈ <strong style="color:#0ea5e9;">{live_wqi:.1f}</strong>
      if trend holds &mdash; changes live as you adjust inputs
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

predict_clicked = st.button("⚡ Predict Next-Year WQI with Random Forest")

# ── Results ───────────────────────────────────────────────────────────────
if predict_clicked:
    feature_row   = build_feature_row(pH, DO, BOD, TC, FC, Nitrate)
    X_scaled      = preprocessor.transform(feature_row)
    predicted_wqi = float(np.clip(rf.predict(X_scaled)[0], 0, 100))
    current_wqi   = compute_wqi(pH, DO, BOD, FC, Nitrate)

    pred_label, pred_color, pred_bg = quality_label(predicted_wqi)
    curr_label, curr_color, curr_bg = quality_label(current_wqi)
    pred_emoji = QUALITY_EMOJI[pred_label]
    curr_emoji = QUALITY_EMOJI[curr_label]

    delta       = predicted_wqi - current_wqi
    delta_str   = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    delta_color = "#10b981" if delta >= 0 else "#ef4444"
    delta_bg    = "rgba(16,185,129,0.08)" if delta >= 0 else "rgba(239,68,68,0.08)"
    delta_border= "rgba(16,185,129,0.25)" if delta >= 0 else "rgba(239,68,68,0.25)"
    delta_icon  = "📈" if delta >= 0 else "📉"
    delta_desc  = "Quality improving" if delta >= 0 else "Quality declining"

    st.markdown('<div class="section-label" style="margin-top:0.5rem;">🎯 Forecast Results</div>', unsafe_allow_html=True)

    # Main result cards
    st.markdown(f"""
    <div class="results-grid">
      <!-- RF Prediction -->
      <div class="result-card" style="background:linear-gradient(135deg,{pred_bg} 0%,rgba(15,23,42,0.9) 100%);
           border-color:{pred_color}33;">
        <div class="result-tag" style="color:{pred_color};">🤖 Random Forest · Next Year</div>
        <div class="result-score" style="color:{pred_color};">{predicted_wqi:.1f}</div>
        <div class="result-quality" style="color:{pred_color};background:{pred_bg};border-color:{pred_color}55;">
          {pred_emoji} {pred_label}
        </div>
        <div style="margin-top:1rem;font-size:0.78rem;color:#334155;line-height:1.5;">
          Model prediction using 11 engineered features<br>
          preprocessed through the fitted pipeline
        </div>
      </div>

      <!-- Persistence -->
      <div class="result-card" style="background:linear-gradient(135deg,{curr_bg} 0%,rgba(15,23,42,0.9) 100%);
           border-color:{curr_color}33;">
        <div class="result-tag" style="color:{curr_color};">📊 Persistence Baseline · Next Year</div>
        <div class="result-score" style="color:{curr_color};">{current_wqi:.1f}</div>
        <div class="result-quality" style="color:{curr_color};background:{curr_bg};border-color:{curr_color}55;">
          {curr_emoji} {curr_label}
        </div>
        <div style="margin-top:1rem;font-size:0.78rem;color:#334155;line-height:1.5;">
          Naive forecast: assumes next year's WQI<br>
          equals this year's measured WQI
        </div>
      </div>
    </div>

    <!-- Delta card -->
    <div class="delta-card" style="background:{delta_bg};border-color:{delta_border};">
      <div class="delta-label">{delta_icon} Model vs Persistence Delta</div>
      <div class="delta-value" style="color:{delta_color};">{delta_str}</div>
      <div class="delta-desc" style="color:{delta_color};">{delta_desc}</div>
      <div style="font-size:0.73rem;color:#334155;margin-top:0.6rem;">
        The RF model predicts WQI will be {abs(delta):.1f} points
        {'higher' if delta >= 0 else 'lower'} than the persistence baseline
      </div>
    </div>
    """, unsafe_allow_html=True)

    # WQI Gauge
    st.markdown(f"""
    <div class="gauge-section">
      <div class="gauge-title">
        <span>Predicted WQI Gauge</span>
        <span style="color:{pred_color};font-family:'JetBrains Mono',monospace;">{predicted_wqi:.1f} / 100</span>
      </div>
      <div class="gauge-outer">
        <div class="gauge-inner" style="width:{predicted_wqi:.1f}%"></div>
      </div>
      <div class="gauge-ticks">
        <div class="gauge-tick"><div class="gauge-tick-line"></div><span>0</span></div>
        <div class="gauge-tick"><div class="gauge-tick-line"></div><span>25</span></div>
        <div class="gauge-tick"><div class="gauge-tick-line"></div><span>50</span></div>
        <div class="gauge-tick"><div class="gauge-tick-line"></div><span>70</span></div>
        <div class="gauge-tick"><div class="gauge-tick-line"></div><span>90</span></div>
        <div class="gauge-tick"><div class="gauge-tick-line"></div><span>100</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-scores breakdown
    q_do      = float(np.clip(100 - (10 - DO) * 10,                        0, 100))
    q_ph      = float(np.clip(100 - abs(pH - 7.0) * 20,                    0, 100))
    q_bod     = float(np.clip(100 - BOD * 12,                               0, 100))
    q_fc      = float(np.clip(100 - np.log1p(FC) / np.log1p(10_000) * 100, 0, 100))
    q_nitrate = float(np.clip(100 - Nitrate * 15,                           0, 100))

    def chip_color(v):
        if v >= 70: return "#10b981"
        if v >= 50: return "#f59e0b"
        return "#ef4444"

    st.markdown(f"""
    <div class="section-label" style="margin-top:1.5rem;">🧪 WQI Sub-Score Breakdown</div>
    <div class="scores-grid">
      <div class="score-chip" style="border-color:{chip_color(q_do)}33;">
        <div class="chip-val" style="color:{chip_color(q_do)};">{q_do:.1f}</div>
        <div class="chip-label">DO</div>
        <div class="chip-weight">weight 0.31</div>
      </div>
      <div class="score-chip" style="border-color:{chip_color(q_ph)}33;">
        <div class="chip-val" style="color:{chip_color(q_ph)};">{q_ph:.1f}</div>
        <div class="chip-label">pH</div>
        <div class="chip-weight">weight 0.11</div>
      </div>
      <div class="score-chip" style="border-color:{chip_color(q_bod)}33;">
        <div class="chip-val" style="color:{chip_color(q_bod)};">{q_bod:.1f}</div>
        <div class="chip-label">BOD</div>
        <div class="chip-weight">weight 0.19</div>
      </div>
      <div class="score-chip" style="border-color:{chip_color(q_fc)}33;">
        <div class="chip-val" style="color:{chip_color(q_fc)};">{q_fc:.1f}</div>
        <div class="chip-label">FC</div>
        <div class="chip-weight">weight 0.28</div>
      </div>
      <div class="score-chip" style="border-color:{chip_color(q_nitrate)}33;">
        <div class="chip-val" style="color:{chip_color(q_nitrate)};">{q_nitrate:.1f}</div>
        <div class="chip-label">Nitrate</div>
        <div class="chip-weight">weight 0.11</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Engineered features expander
    with st.expander("🔧 Engineered Features (what the model sees)"):
        raw_feats = {
            "WQI (current year)": (f"{live_wqi:.4f}", False),
            "pH": (f"{pH:.4f}", False),
            "DO": (f"{DO:.4f}", False),
            "BOD": (f"{BOD:.4f}", False),
            "TC": (f"{TC:.4f}", False),
            "FC": (f"{FC:.4f}", False),
            "Nitrate": (f"{Nitrate:.4f}", False),
            "DO_BOD_ratio": (f"{DO/(BOD+1e-5):.4f}", True),
            "pH_deviation": (f"{abs(pH-7.0):.4f}", True),
            "log_fecal_coliform": (f"{np.log1p(max(FC,0)):.4f}", True),
            "log_total_coliform": (f"{np.log1p(max(TC,0)):.4f}", True),
        }
        rows_html = "".join(
            f'<tr><td>{"🔵 " if computed else "⚪ "}{name}</td>'
            f'<td class="{"feat-computed" if computed else ""}">{val}</td></tr>'
            for name, (val, computed) in raw_feats.items()
        )
        st.markdown(f"""
        <table class="feat-table">
          <thead><tr><th>Feature</th><th>Value</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        <div style="font-size:0.7rem;color:#334155;margin-top:0.8rem;">
          🔵 Computed &nbsp;|&nbsp; ⚪ Raw input &nbsp;|&nbsp;
          <span style="color:#6366f1;">Purple = engineered column</span>
        </div>
        """, unsafe_allow_html=True)

    # Quality scale
    with st.expander("📖 WQI Quality Scale Reference"):
        st.markdown("""
        <div class="quality-legend">
          <div class="legend-item" style="color:#10b981;border-color:#10b98133;background:rgba(16,185,129,0.06);">
            <div class="leg-range">90–100</div><div class="leg-label">🟢 Excellent</div>
            <div style="font-size:0.6rem;color:#064e3b;margin-top:0.3rem;">Pristine — all uses</div>
          </div>
          <div class="legend-item" style="color:#3b82f6;border-color:#3b82f633;background:rgba(59,130,246,0.06);">
            <div class="leg-range">70–89</div><div class="leg-label">🔵 Good</div>
            <div style="font-size:0.6rem;color:#1e3a5f;margin-top:0.3rem;">Minor treatment needed</div>
          </div>
          <div class="legend-item" style="color:#f59e0b;border-color:#f59e0b33;background:rgba(245,158,11,0.06);">
            <div class="leg-range">50–69</div><div class="leg-label">🟡 Medium</div>
            <div style="font-size:0.6rem;color:#451a03;margin-top:0.3rem;">Irrigation & contact</div>
          </div>
          <div class="legend-item" style="color:#f97316;border-color:#f9731633;background:rgba(249,115,22,0.06);">
            <div class="leg-range">25–49</div><div class="leg-label">🟠 Poor</div>
            <div style="font-size:0.6rem;color:#431407;margin-top:0.3rem;">Treatment required</div>
          </div>
          <div class="legend-item" style="color:#ef4444;border-color:#ef444433;background:rgba(239,68,68,0.06);">
            <div class="leg-range">0–24</div><div class="leg-label">🔴 Very Poor</div>
            <div style="font-size:0.6rem;color:#450a0a;margin-top:0.3rem;">Heavily polluted</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  WQI formula · Random Forest · feature engineering from <code>src/model.py</code>
  &nbsp;·&nbsp; models from <code>models/rf.joblib</code>
</div>
""", unsafe_allow_html=True)
