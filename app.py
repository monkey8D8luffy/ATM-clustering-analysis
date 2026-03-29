# =============================================================================
# ATM INTELLIGENCE DEMAND FORECASTING — Glass OS Edition
# FA-2: Building Actionable Insights and an Interactive Python Script
# Author : Saurav Kamble (IBCP - Artificial Intelligence)
# =============================================================================
# Dependencies:
# pip install streamlit pandas numpy scikit-learn plotly scipy
# Run:
# streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import warnings
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATM Intelligence · Glass OS",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="collapsed", # Sidebar removed
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CUSTOM CSS  — Glass OS / Layout Overrides / Water Drop Morph
# ─────────────────────────────────────────────────────────────────────────────
GLASS_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:       #03050f;
    --bg-mid:        #080c24;
    --accent-blue:   #00d4ff;
    --accent-violet: #7b2fff;
    --accent-rose:   #ff2d78;
    --accent-green:  #00ffb2;
    --glass-bg:      rgba(255, 255, 255, 0.06);
    --glass-border:  rgba(255, 255, 255, 0.12);
    --glass-blur:    blur(14px);
    --text-primary:  #e8eeff;
    --text-muted:    rgba(200, 210, 255, 0.55);
    --radius:        16px;
    --radius-sm:     10px;
}

/* ── Global Reset & Background ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #0d1a4a 0%, #03050f 55%),
                radial-gradient(ellipse at 80% 80%, #1a0535 0%, #03050f 55%);
    background-blend-mode: screen;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome & Sidebar completely ── */
#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Main content wrapper (Padding added for Fixed Header Layout) ── */
.block-container {
    padding: 7rem 2rem 3rem 2rem !important;
    max-width: 100% !important;
}

/* ── Fixed Layout: Top Title ── */
.top-title-img {
    position: fixed;
    top: 1.8rem;
    left: 2.5rem;
    z-index: 1000;
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: 1px;
    text-shadow: 0 0 10px rgba(0,212,255,0.4);
}

/* ── Fixed Layout: Center Pill Tabs ── */
[data-testid="stTabs"] > div:first-child {
    position: fixed;
    top: 1.5rem;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background: rgba(8, 12, 36, 0.6) !important;
    backdrop-filter: blur(15px) !important;
    -webkit-backdrop-filter: blur(15px) !important;
    border-radius: 40px !important;
    padding: 6px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    width: max-content;
    display: flex;
    gap: 8px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    border-radius: 30px !important;
    padding: 0.6rem 1.4rem !important;
    border: none !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(123,47,255,0.5), rgba(0,212,255,0.4)) !important;
    color: #fff !important;
    box-shadow: 0 0 18px rgba(123,47,255,0.3) !important;
}

/* ── Fixed Layout: Top Right Setting Button ── */
[data-testid="stButton"]:first-of-type {
    position: fixed;
    top: 1.5rem;
    right: 2.5rem;
    z-index: 1000;
}
[data-testid="stButton"]:first-of-type button {
    border-radius: 30px !important;
    background: rgba(8, 12, 36, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(15px) !important;
    padding: 0.6rem 2.2rem !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
[data-testid="stButton"]:first-of-type button:hover {
    background: rgba(123,47,255,0.4) !important;
    box-shadow: 0 0 20px rgba(123,47,255,0.4) !important;
}

/* ── Dialog / Pop-up Glassmorphism Animation ── */
[data-testid="stDialog"] > div {
    background: rgba(8, 12, 36, 0.65) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.25) !important;
    animation: glassMorph 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}
[data-testid="stModal"] {
    background: rgba(3, 5, 15, 0.7) !important;
    backdrop-filter: blur(8px) !important;
}
@keyframes glassMorph {
    0% { opacity: 0; transform: scale(0.92) translateY(30px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(32px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* ── WATER DROP MORPH ANIMATION ── */
@keyframes waterDropMorph {
    0%   { border-radius: 20px 20px 20px 20px; box-shadow: 0 4px 15px rgba(0,212,255,0.10); }
    25%  { border-radius: 28px 15px 22px 18px; box-shadow: 0 6px 20px rgba(123,47,255,0.15); }
    50%  { border-radius: 18px 28px 15px 22px; box-shadow: 0 4px 15px rgba(0,212,255,0.10); }
    75%  { border-radius: 22px 18px 28px 15px; box-shadow: 0 6px 20px rgba(123,47,255,0.15); }
    100% { border-radius: 20px 20px 20px 20px; box-shadow: 0 4px 15px rgba(0,212,255,0.10); }
}

/* ── KPI Metric Cards (With Water Drop Morph) ── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.12) 0%, rgba(123,47,255,0.12) 100%);
    border: 1px solid rgba(0,212,255,0.22);
    padding: 1.4rem 1.3rem;
    text-align: center;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    /* Morphing Animation injected here */
    animation: waterDropMorph 6s infinite alternate ease-in-out, slideUpFade 0.6s cubic-bezier(0.16,1,0.3,1) both !important;
    transition: transform 0.25s, box-shadow 0.25s;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,212,255,0.5);
}
.metric-value {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent-blue) !important;
}
.metric-label {
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.metric-delta {
    font-size: 0.82rem;
    color: var(--accent-green) !important;
    margin-top: 0.15rem;
}

/* Standard Glass Card styling with morph */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    animation: waterDropMorph 8s infinite alternate ease-in-out, slideUpFade 0.55s cubic-bezier(0.16,1,0.3,1) both;
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.35);
}

.section-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-blue) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-left: 3px solid var(--accent-violet);
    padding-left: 0.7rem;
    margin-bottom: 0.9rem;
}

/* ── Dropdown / Expander Checkbox Lists ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    margin-bottom: 1rem !important;
}

/* ── DataFrames ── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    backdrop-filter: var(--glass-blur) !important;
}

/* ── General Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

/* ── Settings Labels & Dividers ── */
.sidebar-section-header {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--accent-violet) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 1rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(123,47,255,0.3);
}

/* ── Plotly transparent backgrounds ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER — Apply consistent Plotly dark-glass theme
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Rajdhani, sans-serif", color="#e8eeff", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    colorway=["#00d4ff", "#7b2fff", "#ff2d78", "#00ffb2", "#ffb300", "#ff6b6b"],
    legend=dict(
        bgcolor="rgba(255,255,255,0.06)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
    ),
)

def apply_glass(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(family="Orbitron, monospace", size=14, color="#00d4ff")),
        height=height,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "atm_cash_management_dataset.csv"

@st.cache_data(show_spinner=False)
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"]      = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day
    df["Week"]       = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"]       = df["Date"].dt.year
    df["DayOfYear"]  = df["Date"].dt.dayofyear

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = df.select_dtypes(include="object").columns.difference(["ATM_ID", "Date"])
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    df["Location_Type_Enc"] = le.fit_transform(df["Location_Type"].astype(str))
    df["Weather_Enc"]       = le.fit_transform(df["Weather_Condition"].astype(str))
    df["TimeOfDay_Enc"]     = le.fit_transform(df["Time_of_Day"].astype(str))
    df["DayOfWeek_Enc"]     = le.fit_transform(df["Day_of_Week"].astype(str))

    df["Net_Flow"]          = df["Total_Deposits"] - df["Total_Withdrawals"]
    df["Cash_Utilisation"]  = (
        df["Total_Withdrawals"] /
        (df["Previous_Day_Cash_Level"].replace(0, np.nan))
    ).fillna(0).clip(0, 5)

    return df.dropna(subset=["Date"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_cluster_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df[feature_cols].fillna(0))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ML FUNCTIONS 
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_kmeans(X: np.ndarray, k: int, random_state: int) -> tuple:
    inertias, sil_scores = [], []
    k_range = range(2, min(12, len(X)))
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, lbl))

    final_km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = final_km.fit_predict(X)
    return labels, inertias, sil_scores, list(k_range)

def run_isolation_forest(df: pd.DataFrame, contamination: float, feature_cols: list) -> pd.DataFrame:
    X = df[feature_cols].fillna(0).values
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso.fit(X)
    
    df_out = df.copy()
    df_out["Anomaly_Score"]  = iso.decision_function(X) * -1
    df_out["Is_Anomaly"]     = iso.predict(X)
    df_out["Anomaly_Label"]  = df_out["Is_Anomaly"].map({-1: "⚠ Anomaly", 1: "✔ Normal"})
    return df_out

@st.cache_data(show_spinner=False)
def compute_forecast(df: pd.DataFrame, days_ahead: int = 7) -> tuple:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    ts = (
        df.groupby("Date")["Cash_Demand_Next_Day"]
        .mean()
        .sort_index()
        .reset_index()
    )
    ts.columns = ["Date", "Avg_Cash_Demand"]

    if len(ts) == 0: return pd.DataFrame(), pd.DataFrame()

    alpha = 0.3
    smoothed = [ts["Avg_Cash_Demand"].iloc[0]]
    for val in ts["Avg_Cash_Demand"].iloc[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    ts["Smoothed"] = smoothed

    last_val   = ts["Smoothed"].iloc[-1]
    last_date  = ts["Date"].iloc[-1]
    
    if len(ts) > 1:
        slope = ts["Smoothed"].diff().tail(30).mean()
        if pd.isna(slope): slope = 0
    else:
        slope = 0
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    future_vals  = [last_val + slope * (i + 1) for i in range(days_ahead)]

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_vals})
    return ts, forecast_df

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOAD DATA & INITIALISE STATE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🏧  Initialising Glass OS · Loading ATM data..."):
    try:
        df = load_and_preprocess(DATA_PATH)
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False

if not data_loaded:
    st.error(
        f"**Dataset not found:** `{DATA_PATH}`  \n"
        "Please place `atm_cash_management_dataset.csv` in the same folder as `app.py`."
    )
    st.stop()

all_locations = sorted(df["Location_Type"].unique())
all_weather = sorted(df["Weather_Condition"].unique())
date_min, date_max = df["Date"].min().date(), df["Date"].max().date()

if "loc_filter" not in st.session_state: st.session_state.loc_filter = all_locations
if "wx_filter" not in st.session_state: st.session_state.wx_filter = all_weather
if "date_range" not in st.session_state: st.session_state.date_range = (date_min, date_max)
if "k_val" not in st.session_state: st.session_state.k_val = 4
if "km_rs" not in st.session_state: st.session_state.km_rs = 42
if "cl_feat" not in st.session_state: 
    st.session_state.cl_feat = ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs", "Cash_Utilisation"]
if "iso_cont" not in st.session_state: st.session_state.iso_cont = 0.05
if "ano_feat" not in st.session_state: 
    st.session_state.ano_feat = ["Total_Withdrawals", "Cash_Demand_Next_Day"]
if "hol_only" not in st.session_state: st.session_state.hol_only = False
if "fc_days" not in st.session_state: st.session_state.fc_days = 7

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SETTINGS DIALOG (POP-UP GLASS OS)
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("⚙️ ATM OS Settings")
def settings_dialog():
    st.markdown(
        '<div style="text-align:center; padding: 0 0 1rem;">'
        '<span style="font-family:Orbitron,monospace; font-size:1.1rem; '
        'background:linear-gradient(135deg,#00d4ff,#7b2fff); '
        '-webkit-background-clip:text; -webkit-text-fill-color:transparent; '
        'font-weight:800; letter-spacing:2px;">INTELLIGENCE CONTROL PANEL</span><br>'
        '<span style="font-size:0.68rem; color:rgba(200,210,255,0.45); '
        'letter-spacing:1.5px;">SYSTEM PREFERENCES</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-header">Global Filters</div>', unsafe_allow_html=True)
    with st.expander("📍 Location Type"):
        temp_locs = []
        for loc in all_locations:
            if st.checkbox(loc, value=(loc in st.session_state.loc_filter), key=f"loc_chk_{loc}"):
                temp_locs.append(loc)
        st.session_state.loc_filter = temp_locs

    with st.expander("⛅ Weather Condition"):
        temp_wx = []
        for wx in all_weather:
            if st.checkbox(wx, value=(wx in st.session_state.wx_filter), key=f"wx_chk_{wx}"):
                temp_wx.append(wx)
        st.session_state.wx_filter = temp_wx
        
    st.date_input("Date Range", min_value=date_min, max_value=date_max, key="date_range")

    st.markdown('<div class="sidebar-section-header">Clustering (K-Means)</div>', unsafe_allow_html=True)
    st.slider("Number of Clusters (K)", 2, 10, key="k_val")
    st.slider("Random State", 0, 100, key="km_rs")
    
    cl_opts = ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs",
               "Cash_Utilisation", "Net_Flow", "Previous_Day_Cash_Level"]
    with st.expander("🔵 Cluster Features"):
        temp_cl = []
        for feat in cl_opts:
            if st.checkbox(feat, value=(feat in st.session_state.cl_feat), key=f"cl_chk_{feat}"):
                temp_cl.append(feat)
        st.session_state.cl_feat = temp_cl

    st.markdown('<div class="sidebar-section-header">Anomaly Detection</div>', unsafe_allow_html=True)
    st.slider("Contamination Rate", 0.01, 0.30, step=0.01, key="iso_cont")
    
    ano_opts = ["Total_Withdrawals", "Cash_Demand_Next_Day", "Previous_Day_Cash_Level", "Net_Flow"]
    with st.expander("⚠ Anomaly Features"):
        temp_ano = []
        for feat in ano_opts:
            if st.checkbox(feat, value=(feat in st.session_state.ano_feat), key=f"ano_chk_{feat}"):
                temp_ano.append(feat)
        st.session_state.ano_feat = temp_ano
        
    st.checkbox("Holiday / Event Days Only", key="hol_only")

    st.markdown('<div class="sidebar-section-header">Forecasting</div>', unsafe_allow_html=True)
    st.slider("Days to Forecast", 3, 30, key="fc_days")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Apply Parameters", use_container_width=True):
        st.rerun()

sel_locations = st.session_state.loc_filter
sel_weather = st.session_state.wx_filter
date_range = st.session_state.date_range
k_value = st.session_state.k_val
km_random_state = st.session_state.km_rs
cluster_features = st.session_state.cl_feat
iso_contamination = st.session_state.iso_cont
anomaly_features = st.session_state.ano_feat
holiday_only = st.session_state.hol_only
forecast_days = st.session_state.fc_days

# ─────────────────────────────────────────────────────────────────────────────
# 7.  APPLY GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d_start, d_end = df["Date"].min(), df["Date"].max()

mask = (
    df["Location_Type"].isin(sel_locations) &
    df["Weather_Condition"].isin(sel_weather) &
    df["Date"].between(d_start, d_end)
)
dff = df[mask].copy()

if dff.empty:
    st.warning("⚠ No data matches your current filters. Please widen the selection to view data.")
    if st.button("⚙️ Open Settings"):
        settings_dialog()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HEADER & LAYOUT INJECTION
# ─────────────────────────────────────────────────────────────────────────────

# Inject Fixed Header Title (Top Left)
st.markdown('<div class="top-title-img">ATM intelligence</div>', unsafe_allow_html=True)

# Inject Fixed Setting Button (Top Right)
if st.button("Setting"):
    settings_dialog()


# ── KPI Row ─────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

def kpi_card(col, value, label, delta=""):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-delta">{delta}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

total_w  = f"${dff['Total_Withdrawals'].sum()/1e6:,.1f}M"
avg_dem  = f"${dff['Cash_Demand_Next_Day'].mean():,.0f}"
n_atms   = str(dff["ATM_ID"].nunique())
hol_rows = str(dff[dff["Holiday_Flag"] == 1].shape[0])
util_pct = f"{dff['Cash_Utilisation'].mean()*100:.1f}%"

kpi_card(kpi1, total_w,  "Total Withdrawals",   "↑ filtered period")
kpi_card(kpi2, avg_dem,  "Avg. Cash Demand",    "next-day projection")
kpi_card(kpi3, n_atms,   "Active ATMs",         "in selection")
kpi_card(kpi4, hol_rows, "Holiday Records",     "high-risk rows")
kpi_card(kpi5, util_pct, "Cash Utilisation",    "mean across ATMs")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  TABS (Visually mapped to the Top Center Menu Bar)
# ─────────────────────────────────────────────────────────────────────────────
# Text spelling mapped strictly to the provided image visual
tab_eda, tab_cluster, tab_anomaly, tab_forecast, tab_export = st.tabs([
    "EDA patterns",
    "Clusttring",
    "Anomaly Detection",
    "Forecasting",
    "Export"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB A — EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_wdraw = px.histogram(
            dff, x="Total_Withdrawals", nbins=60,
            color_discrete_sequence=["#00d4ff"],
            marginal="box",
        )
        apply_glass(fig_wdraw, "Distribution — Total Withdrawals", 350)
        fig_wdraw.update_traces(opacity=0.78)
        st.plotly_chart(fig_wdraw, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_h2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_dep = px.histogram(
            dff, x="Total_Deposits", nbins=60,
            color_discrete_sequence=["#7b2fff"],
            marginal="box",
        )
        apply_glass(fig_dep, "Distribution — Total Deposits", 350)
        fig_dep.update_traces(opacity=0.78)
        st.plotly_chart(fig_dep, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Time-Based Trends</div>', unsafe_allow_html=True)

    daily_agg = (
        dff.groupby("Date")
           .agg(Avg_Withdrawals=("Total_Withdrawals", "mean"),
                Avg_Deposits=("Total_Deposits", "mean"))
           .reset_index()
    )

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=daily_agg["Date"], y=daily_agg["Avg_Withdrawals"],
        name="Withdrawals", line=dict(color="#00d4ff", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
    ))
    fig_ts.add_trace(go.Scatter(
        x=daily_agg["Date"], y=daily_agg["Avg_Deposits"],
        name="Deposits", line=dict(color="#7b2fff", width=1.8),
        fill="tozeroy", fillcolor="rgba(123,47,255,0.07)",
    ))
    apply_glass(fig_ts, "Daily Average Withdrawal & Deposit Trends", 380)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    col_dow, col_tod = st.columns(2)
    with col_dow:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_agg = dff.groupby("Day_of_Week")["Total_Withdrawals"].mean().reindex(dow_order).reset_index()
        fig_dow = px.bar(
            dow_agg, x="Day_of_Week", y="Total_Withdrawals",
            color="Total_Withdrawals",
            color_continuous_scale=["#0d1a4a","#00d4ff","#7b2fff"],
        )
        apply_glass(fig_dow, "Avg Withdrawals by Day of Week", 340)
        fig_dow.update_coloraxes(showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_tod:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tod_order = ["Morning","Afternoon","Evening","Night"]
        tod_agg = dff.groupby("Time_of_Day")["Total_Withdrawals"].mean().reindex(tod_order).reset_index()
        fig_tod = px.bar(
            tod_agg, x="Time_of_Day", y="Total_Withdrawals",
            color="Total_Withdrawals",
            color_continuous_scale=["#1a0535","#ff2d78","#ffb300"],
        )
        apply_glass(fig_tod, "Avg Withdrawals by Time of Day", 340)
        fig_tod.update_coloraxes(showscale=False)
        st.plotly_chart(fig_tod, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">3D Relationship Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_3d = px.scatter_3d(
        dff.sample(min(4000, len(dff)), random_state=1),
        x="Total_Withdrawals",
        y="Previous_Day_Cash_Level",
        z="Nearby_Competitor_ATMs",
        color="Weather_Condition",
        symbol="Location_Type",
        opacity=0.65,
        size_max=5,
        color_discrete_sequence=["#00d4ff","#7b2fff","#ff2d78","#00ffb2","#ffb300"],
    )
    apply_glass(fig_3d, "3D Explorer: Withdrawals × Cash Level × Competitors", 550)
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
            bgcolor="rgba(0,0,0,0)",
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    numeric_heat = dff[[
        "Total_Withdrawals","Total_Deposits","Previous_Day_Cash_Level",
        "Cash_Demand_Next_Day","Nearby_Competitor_ATMs",
        "Holiday_Flag","Special_Event_Flag","Cash_Utilisation","Net_Flow",
    ]]
    corr = numeric_heat.corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=["#0d1a4a","#7b2fff","#00d4ff","#00ffb2"],
        aspect="auto",
    )
    apply_glass(fig_heat, "Correlation Heatmap — Numeric Features", 480)
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB B — CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown('<div class="section-title">K-Means ATM Clustering</div>', unsafe_allow_html=True)

    if len(cluster_features) < 2:
        st.warning("⚠ Please select at least 2 cluster features in the settings panel.")
    else:
        X_cl = get_cluster_features(dff, cluster_features)
        labels, inertias, sil_scores, k_range = run_kmeans(X_cl, k_value, km_random_state)

        dff_cl = dff.copy()
        dff_cl["Cluster"] = labels.astype(str)

        cluster_agg = dff_cl.groupby("Cluster")["Total_Withdrawals"].mean().sort_values()
        rank_labels = {
            str(c): name for c, name in zip(
                cluster_agg.index,
                ["🟢 Low Demand","🔵 Moderate","🟡 High Demand","🔴 Very High"][:len(cluster_agg)]
                + [f"Cluster {i}" for i in range(len(cluster_agg), k_value)],
            )
        }
        dff_cl["Cluster_Label"] = dff_cl["Cluster"].map(rank_labels)

        col_el, col_sil = st.columns(2)
        with col_el:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_el = go.Figure()
            fig_el.add_trace(go.Scatter(
                x=list(k_range), y=inertias,
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=8, color="#7b2fff"),
                name="Inertia",
            ))
            fig_el.add_vline(
                x=k_value, line_dash="dash",
                line_color="rgba(255,45,120,0.7)", line_width=1.5,
            )
            apply_glass(fig_el, "Elbow Method — Inertia vs K", 320)
            st.plotly_chart(fig_el, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_sil:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=list(k_range), y=sil_scores,
                mode="lines+markers",
                line=dict(color="#00ffb2", width=2),
                marker=dict(size=8, color="#ffb300"),
                name="Silhouette",
            ))
            fig_sil.add_vline(
                x=k_value, line_dash="dash",
                line_color="rgba(255,45,120,0.7)", line_width=1.5,
            )
            apply_glass(fig_sil, "Silhouette Score vs K", 320)
            st.plotly_chart(fig_sil, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">3D Cluster Visualisation</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        sample_cl = dff_cl.sample(min(5000, len(dff_cl)), random_state=7)
        fig_3dcl = px.scatter_3d(
            sample_cl,
            x="Total_Withdrawals",
            y="Cash_Demand_Next_Day",
            z="Previous_Day_Cash_Level",
            color="Cluster_Label",
            symbol="Location_Type",
            opacity=0.70,
            color_discrete_sequence=["#00d4ff","#7b2fff","#ff2d78","#00ffb2","#ffb300","#ff6b6b"],
        )
        apply_glass(fig_3dcl, f"3D Cluster Space — K={k_value}", 580)
        fig_3dcl.update_traces(marker=dict(size=3.5))
        fig_3dcl.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
                bgcolor="rgba(0,0,0,0)",
            )
        )
        st.plotly_chart(fig_3dcl, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        cl_loc = dff_cl.groupby(["Cluster_Label","Location_Type"]).size().reset_index(name="Count")
        fig_clbar = px.bar(
            cl_loc, x="Cluster_Label", y="Count", color="Location_Type",
            color_discrete_sequence=["#00d4ff","#7b2fff","#ff2d78","#00ffb2","#ffb300","#ff6b6b"],
            barmode="stack",
        )
        apply_glass(fig_clbar, "Cluster Composition by Location Type", 380)
        st.plotly_chart(fig_clbar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB C — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<div class="section-title">Isolation Forest · Anomaly Detection</div>', unsafe_allow_html=True)

    if len(anomaly_features) < 1:
        st.warning("⚠ Select at least 1 anomaly feature in the settings panel.")
    else:
        df_ano_base = dff.copy()
        if holiday_only:
            df_ano_base = df_ano_base[(df_ano_base["Holiday_Flag"] == 1) |
                                       (df_ano_base["Special_Event_Flag"] == 1)]

        if df_ano_base.empty:
            st.warning("⚠ No holiday/event rows in the current filter. Uncheck the option in the settings panel.")
        else:
            df_ano = run_isolation_forest(df_ano_base, iso_contamination, anomaly_features)

            n_anomalies = (df_ano["Is_Anomaly"] == -1).sum()
            n_normal    = (df_ano["Is_Anomaly"] ==  1).sum()

            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#ff2d78;">{n_anomalies}</div>'
                    f'<div class="metric-label">Anomalies Detected</div>'
                    f'<div class="metric-delta">contamination={iso_contamination:.2f}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            with col_a2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{n_normal}</div>'
                    f'<div class="metric-label">Normal Records</div>'
                    f'<div class="metric-delta">✔ healthy</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            with col_a3:
                pct = f"{n_anomalies / len(df_ano) * 100:.1f}%"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#ffb300;">{pct}</div>'
                    f'<div class="metric-label">Anomaly Rate</div>'
                    f'<div class="metric-delta">of filtered records</div>'
                    f'</div>', unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            df_normal  = df_ano[df_ano["Is_Anomaly"] ==  1]
            df_spikes  = df_ano[df_ano["Is_Anomaly"] == -1]

            fig_ano = go.Figure()
            fig_ano.add_trace(go.Scatter(
                x=df_normal["Date"], y=df_normal["Total_Withdrawals"],
                mode="markers",
                marker=dict(color="rgba(0,212,255,0.4)", size=4),
                name="Normal",
            ))
            fig_ano.add_trace(go.Scatter(
                x=df_spikes["Date"], y=df_spikes["Total_Withdrawals"],
                mode="markers",
                marker=dict(
                    color="#ff2d78", size=9, symbol="x",
                    line=dict(width=1.5, color="#ffb300"),
                ),
                name="⚠ Anomaly",
            ))
            apply_glass(fig_ano, "Withdrawal Anomaly Overlay — Isolation Forest", 460)
            st.plotly_chart(fig_ano, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Anomaly Records — Detail View</div>', unsafe_allow_html=True)
            display_cols = ["ATM_ID","Date","Location_Type","Total_Withdrawals",
                            "Cash_Demand_Next_Day","Holiday_Flag","Special_Event_Flag",
                            "Anomaly_Score","Anomaly_Label"]

            anomaly_rows = df_spikes[display_cols].sort_values(
                "Anomaly_Score", ascending=False
            ).head(100)

            st.dataframe(
                anomaly_rows.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB D — FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="section-title">Cash Demand Forecasting · Exponential Smoothing</div>', unsafe_allow_html=True)

    ts_df, fc_df = compute_forecast(dff, forecast_days)

    if not ts_df.empty and not fc_df.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_fc = go.Figure()

        fig_fc.add_trace(go.Scatter(
            x=ts_df["Date"], y=ts_df["Avg_Cash_Demand"],
            name="Actual (avg)", mode="lines",
            line=dict(color="rgba(0,212,255,0.35)", width=1.2),
        ))

        fig_fc.add_trace(go.Scatter(
            x=ts_df["Date"], y=ts_df["Smoothed"],
            name="Smoothed (α=0.3)", mode="lines",
            line=dict(color="#00d4ff", width=2.2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        ))

        fig_fc.add_trace(go.Scatter(
            x=fc_df["Date"], y=fc_df["Forecast"],
            name=f"Forecast (+{forecast_days}d)", mode="lines+markers",
            line=dict(color="#ff2d78", width=2.4, dash="dot"),
            marker=dict(size=7, color="#ff2d78"),
        ))

        x_ci = fc_df["Date"].tolist() + fc_df["Date"].tolist()[::-1]
        y_ci = (fc_df["Forecast"] * 1.10).tolist() + (fc_df["Forecast"] * 0.90).tolist()[::-1]
        
        fig_fc.add_trace(go.Scatter(
            x=x_ci, y=y_ci,
            fill="toself",
            fillcolor="rgba(255,45,120,0.09)",
            line=dict(color="rgba(255,45,120,0)"),
            name="±10% CI",
            hoverinfo="skip",
        ))

        last_hist = ts_df["Date"].max()
        fig_fc.add_vline(
            x=last_hist, line_dash="dash",
            line_color="rgba(255,179,0,0.6)", line_width=1.5,
        )
        
        apply_glass(fig_fc, f"Cash Demand Trend + {forecast_days}-Day Forecast", 500)
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Forecast Values</div>', unsafe_allow_html=True)
        fc_display = fc_df.copy()
        fc_display["Lower_CI"] = (fc_display["Forecast"] * 0.90).round(0)
        fc_display["Upper_CI"] = (fc_display["Forecast"] * 1.10).round(0)
        fc_display["Forecast"] = fc_display["Forecast"].round(0)
        fc_display["Date"] = fc_display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(fc_display.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠ Not enough data available to generate a forecast. Please expand your date range.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB E — EXPORT ENGINE
# ═════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-title">Intelligence Report Export Engine</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="glass-card">'
        '<p style="color:rgba(200,210,255,0.7); font-size:0.88rem; line-height:1.7;">'
        'Export the filtered and enriched dataset — including cluster assignments, '
        'anomaly flags, and derived KPIs — as a CSV for downstream analysis.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    export_df = dff.copy()

    if len(cluster_features) >= 2:
        X_exp = get_cluster_features(dff, cluster_features)
        labels_exp, _, _, _ = run_kmeans(X_exp, k_value, km_random_state)
        export_df["Cluster_ID"] = labels_exp

    if len(anomaly_features) >= 1:
        df_exp_ano = run_isolation_forest(export_df, iso_contamination, anomaly_features)
        export_df["Anomaly_Score"]  = df_exp_ano["Anomaly_Score"].values
        export_df["Is_Anomaly"]     = df_exp_ano["Is_Anomaly"].values
        export_df["Anomaly_Label"]  = df_exp_ano["Anomaly_Label"].values

    col_prev, col_stats = st.columns([3, 1])
    with col_prev:
        st.markdown('<div class="section-title">Preview (first 50 rows)</div>', unsafe_allow_html=True)
        st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)

    with col_stats:
        st.markdown(
            '<div class="glass-card">'
            f'<div class="metric-value">{len(export_df):,}</div>'
            '<div class="metric-label">Total Rows</div><br>'
            f'<div class="metric-value" style="font-size:1.3rem;">{len(export_df.columns)}</div>'
            '<div class="metric-label">Columns</div><br>'
            f'<div class="metric-value" style="font-size:1.1rem;">'
            f'{export_df["ATM_ID"].nunique()}</div>'
            '<div class="metric-label">Unique ATMs</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    st.download_button(
        label="💾  Download Intelligence Report (CSV)",
        data=csv_bytes,
        file_name="atm_intelligence_report.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10.  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; margin-top:3rem; padding: 1rem; '
    'border-top:1px solid rgba(255,255,255,0.07);">'
    '<span style="font-size:0.65rem; color:rgba(200,210,255,0.25); '
    'letter-spacing:2px; font-family:Orbitron,monospace;">'
    'SAURAV KAMBLE · IBCP ARTIFICIAL INTELLIGENCE · DATA MINING FA-2'
    '</span>'
    '</div>',
    unsafe_allow_html=True,
)
