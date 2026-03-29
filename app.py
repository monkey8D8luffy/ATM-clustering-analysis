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
    initial_sidebar_state="collapsed", 
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CUSTOM CSS  — Layout matching image + Glass OS Water Morph
# ─────────────────────────────────────────────────────────────────────────────
GLASS_CSS = """
<style>
/* ── Global Reset & Background (Matching the vibrant purple from image) ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #8b5cf6 !important; /* Vibrant purple background */
    min-height: 100vh;
    overflow-x: hidden;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Main content wrapper ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 100% !important;
}

/* ── Global Water Drop Morph Animation Background ── */
.water-drop-bg {
    position: fixed;
    bottom: -10%; left: -10%;
    width: 60vw; height: 60vw;
    background: radial-gradient(circle, rgba(11, 18, 117, 0.15) 0%, transparent 65%);
    border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
    animation: morphBlob 12s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
}
@keyframes morphBlob {
    0% { border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%; transform: translate(0,0) scale(1) rotate(0deg); }
    100% { border-radius: 70% 30% 50% 70% / 30% 70% 70% 30%; transform: translate(5%, -5%) scale(1.1) rotate(15deg); }
}

/* ── Top Header Elements ── */
.custom-title {
    position: absolute;
    top: 35px;
    left: 45px;
    font-size: 28px;
    font-weight: 800;
    color: #000;
    z-index: 100;
}

/* ── Tab Bar (Centered dark pill container like the image) ── */
.stTabs {
    margin-top: -55px; /* Pulls the tabs up onto the header line */
    position: relative;
    z-index: 50;
}
.stTabs [data-baseweb="tab-list"] {
    margin: 0 auto !important;
    background-color: #0c1273 !important; /* Deep navy blue */
    border-radius: 40px !important;
    padding: 6px 12px !important;
    gap: 5px !important;
    border: none !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 30px !important;
    padding: 8px 20px !important;
    color: #aeb4e6 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #4852b5 !important; /* Lighter blue/purple active state */
    color: #fff !important;
}

/* ── Settings Button Styling (Pill shape right aligned) ── */
.stButton > button {
    background-color: #0c1273 !important;
    color: #fff !important;
    border-radius: 30px !important;
    padding: 8px 30px !important;
    font-weight: 600 !important;
    border: none !important;
    float: right;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    transition: transform 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    background-color: #121b9e !important;
}

/* ── KPI Metric Cards (The 5 dark blue rounded rectangles) ── */
.metric-card {
    background-color: #0c1273 !important;
    border-radius: 30px;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
    margin-top: 15px;
    margin-bottom: 25px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
/* Internal water drop morph for cards */
.metric-card::before {
    content: '';
    position: absolute;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 60%);
    border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
    animation: morphCard 6s linear infinite alternate;
    top: -40px; left: -40px;
    z-index: 0;
}
@keyframes morphCard {
    0% { border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%; transform: rotate(0deg); }
    100% { border-radius: 70% 30% 50% 70% / 30% 70% 70% 30%; transform: rotate(25deg); }
}
.metric-value, .metric-label, .metric-delta {
    position: relative;
    z-index: 1;
    text-align: center;
}
.metric-value { font-size: 28px; color: #fff !important; font-weight: 800; }
.metric-label { font-size: 13px; color: #aeb4e6 !important; margin-top: 6px; font-weight: 600;}
.metric-delta { font-size: 12px; color: #00ffb2 !important; margin-top: 4px; }

/* ── Content Glass Cards ── */
.glass-card {
    background: rgba(12, 18, 115, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    color: white;
}

/* ── Section Titles ── */
.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #fff !important;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}

/* ── Dialog Glassmorphism ── */
[data-testid="stDialog"] > div {
    background: rgba(12, 18, 115, 0.85) !important;
    backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 24px !important;
    color: white !important;
}
[data-testid="stModal"] {
    background: rgba(139, 92, 246, 0.5) !important;
    backdrop-filter: blur(8px) !important;
}
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)
st.markdown('<div class="water-drop-bg"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER — Apply transparent Plotly theme for purple background
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#fff", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
    colorway=["#00d4ff", "#a78bfa", "#ff2d78", "#00ffb2", "#ffb300"],
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)

def apply_glass(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(family="Inter", size=16, color="#fff", weight="bold")),
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
        
    ts = df.groupby("Date")["Cash_Demand_Next_Day"].mean().sort_index().reset_index()
    ts.columns = ["Date", "Avg_Cash_Demand"]

    if len(ts) == 0:
        return pd.DataFrame(), pd.DataFrame()

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
with st.spinner("ATM OS is initializing..."):
    try:
        df = load_and_preprocess(DATA_PATH)
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False

if not data_loaded:
    st.error(f"**Dataset not found:** `{DATA_PATH}`")
    st.stop()

all_locations = sorted(df["Location_Type"].unique())
all_weather = sorted(df["Weather_Condition"].unique())
date_min, date_max = df["Date"].min().date(), df["Date"].max().date()

if "loc_filter" not in st.session_state: st.session_state.loc_filter = all_locations
if "wx_filter" not in st.session_state: st.session_state.wx_filter = all_weather
if "date_range" not in st.session_state: st.session_state.date_range = (date_min, date_max)
if "k_val" not in st.session_state: st.session_state.k_val = 4
if "km_rs" not in st.session_state: st.session_state.km_rs = 42
if "cl_feat" not in st.session_state: st.session_state.cl_feat = ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs", "Cash_Utilisation"]
if "iso_cont" not in st.session_state: st.session_state.iso_cont = 0.05
if "ano_feat" not in st.session_state: st.session_state.ano_feat = ["Total_Withdrawals", "Cash_Demand_Next_Day"]
if "hol_only" not in st.session_state: st.session_state.hol_only = False
if "fc_days" not in st.session_state: st.session_state.fc_days = 7

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SETTINGS DIALOG
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("⚙️ Settings")
def settings_dialog():
    st.markdown('<div class="section-title">Global Filters</div>', unsafe_allow_html=True)
    with st.expander("📍 Location Type"):
        temp_locs = []
        for loc in all_locations:
            if st.checkbox(loc, value=(loc in st.session_state.loc_filter), key=f"loc_{loc}"): temp_locs.append(loc)
        st.session_state.loc_filter = temp_locs

    with st.expander("⛅ Weather Condition"):
        temp_wx = []
        for wx in all_weather:
            if st.checkbox(wx, value=(wx in st.session_state.wx_filter), key=f"wx_{wx}"): temp_wx.append(wx)
        st.session_state.wx_filter = temp_wx
        
    st.date_input("Date Range", min_value=date_min, max_value=date_max, key="date_range")

    st.markdown('<div class="section-title">Clustering</div>', unsafe_allow_html=True)
    st.slider("Clusters (K)", 2, 10, key="k_val")
    st.slider("Random State", 0, 100, key="km_rs")
    cl_opts = ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs", "Cash_Utilisation", "Net_Flow", "Previous_Day_Cash_Level"]
    with st.expander("🔵 Features"):
        temp_cl = []
        for feat in cl_opts:
            if st.checkbox(feat, value=(feat in st.session_state.cl_feat), key=f"cl_{feat}"): temp_cl.append(feat)
        st.session_state.cl_feat = temp_cl

    st.markdown('<div class="section-title">Anomalies</div>', unsafe_allow_html=True)
    st.slider("Contamination", 0.01, 0.30, step=0.01, key="iso_cont")
    ano_opts = ["Total_Withdrawals", "Cash_Demand_Next_Day", "Previous_Day_Cash_Level", "Net_Flow"]
    with st.expander("⚠ Features"):
        temp_ano = []
        for feat in ano_opts:
            if st.checkbox(feat, value=(feat in st.session_state.ano_feat), key=f"ano_{feat}"): temp_ano.append(feat)
        st.session_state.ano_feat = temp_ano
    st.checkbox("Holiday Events Only", key="hol_only")

    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)
    st.slider("Days to Forecast", 3, 30, key="fc_days")
    if st.button("Apply", use_container_width=True): st.rerun()

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

if len(date_range) == 2:
    d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d_start, d_end = df["Date"].min(), df["Date"].max()

dff = df[(df["Location_Type"].isin(sel_locations)) & (df["Weather_Condition"].isin(sel_weather)) & (df["Date"].between(d_start, d_end))].copy()

if dff.empty:
    st.warning("⚠ No data matches filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 8.  HEADER & KPI ROW (Rebuilt layout to match the image perfectly)
# ─────────────────────────────────────────────────────────────────────────────

# Inject absolute Title
st.markdown('<div class="custom-title">ATM intelligence</div>', unsafe_allow_html=True)

# Grid layout for Setting button on the right
top_col1, top_col2, top_col_btn = st.columns([2, 5, 2])
with top_col_btn:
    st.markdown('<div style="padding-top: 15px;"></div>', unsafe_allow_html=True) # push down slightly
    if st.button("Setting", key="header_setting"):
        settings_dialog()

# Tabs (CSS will pull this up horizontally into the center)
tab_eda, tab_cluster, tab_anomaly, tab_forecast, tab_export = st.tabs([
    "EDA patterns",
    "Clusttring",
    "Anomaly Detection",
    "Forecasting",
    "Export",
])

# ── 5 Solid Dark Blue KPI Cards (Matching Image)
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

kpi_card(kpi1, total_w,  "TOTAL WITHDRAWALS", "↑ Last Period")
kpi_card(kpi2, avg_dem,  "AVG CASH DEMAND",   "Next-Day Proj")
kpi_card(kpi3, n_atms,   "ACTIVE ATMS",       "In Network")
kpi_card(kpi4, hol_rows, "EVENT DAYS",        "Risk Factor")
kpi_card(kpi5, util_pct, "CASH UTILIZATION",  "Mean Overall")

# ═════════════════════════════════════════════════════════════════════════════
# TAB CONTENT
# ═════════════════════════════════════════════════════════════════════════════
with tab_eda:
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_wdraw = px.histogram(dff, x="Total_Withdrawals", nbins=60, color_discrete_sequence=["#00d4ff"], marginal="box")
        apply_glass(fig_wdraw, "Distribution — Withdrawals", 350)
        st.plotly_chart(fig_wdraw, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_h2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_dep = px.histogram(dff, x="Total_Deposits", nbins=60, color_discrete_sequence=["#a78bfa"], marginal="box")
        apply_glass(fig_dep, "Distribution — Deposits", 350)
        st.plotly_chart(fig_dep, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    daily_agg = dff.groupby("Date").agg(Avg_Withdrawals=("Total_Withdrawals", "mean"), Avg_Deposits=("Total_Deposits", "mean")).reset_index()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=daily_agg["Date"], y=daily_agg["Avg_Withdrawals"], name="Withdrawals", line=dict(color="#00d4ff", width=2)))
    fig_ts.add_trace(go.Scatter(x=daily_agg["Date"], y=daily_agg["Avg_Deposits"], name="Deposits", line=dict(color="#a78bfa", width=2)))
    apply_glass(fig_ts, "Daily Trend Comparison", 380)
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with tab_cluster:
    if len(cluster_features) >= 2:
        X_cl = get_cluster_features(dff, cluster_features)
        labels, inertias, sil_scores, k_range = run_kmeans(X_cl, k_value, km_random_state)
        dff_cl = dff.copy()
        dff_cl["Cluster"] = labels.astype(str)
        
        col_el, col_sil = st.columns(2)
        with col_el:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_el = px.line(x=k_range, y=inertias, markers=True, color_discrete_sequence=["#00d4ff"])
            fig_el.add_vline(x=k_value, line_dash="dash", line_color="#ff2d78")
            apply_glass(fig_el, "Elbow Method (Inertia)", 320)
            st.plotly_chart(fig_el, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_sil:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_sil = px.line(x=k_range, y=sil_scores, markers=True, color_discrete_sequence=["#00ffb2"])
            fig_sil.add_vline(x=k_value, line_dash="dash", line_color="#ff2d78")
            apply_glass(fig_sil, "Silhouette Score", 320)
            st.plotly_chart(fig_sil, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_3dcl = px.scatter_3d(dff_cl.sample(min(3000, len(dff_cl))), x="Total_Withdrawals", y="Cash_Demand_Next_Day", z="Previous_Day_Cash_Level", color="Cluster", opacity=0.8)
        apply_glass(fig_3dcl, f"3D Cluster Distribution (K={k_value})", 550)
        st.plotly_chart(fig_3dcl, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_anomaly:
    df_ano_base = dff.copy()
    if holiday_only: df_ano_base = df_ano_base[(df_ano_base["Holiday_Flag"] == 1) | (df_ano_base["Special_Event_Flag"] == 1)]
    
    if len(anomaly_features) >= 1 and not df_ano_base.empty:
        df_ano = run_isolation_forest(df_ano_base, iso_contamination, anomaly_features)
        df_spikes = df_ano[df_ano["Is_Anomaly"] == -1]
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_ano = px.scatter(df_ano, x="Date", y="Total_Withdrawals", color="Anomaly_Label", color_discrete_map={"✔ Normal": "#00d4ff", "⚠ Anomaly": "#ff2d78"})
        apply_glass(fig_ano, "Detected Anomalies", 450)
        st.plotly_chart(fig_ano, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card"><div class="section-title">Anomaly Log</div>', unsafe_allow_html=True)
        st.dataframe(df_spikes[["Date","Total_Withdrawals","Cash_Demand_Next_Day","Anomaly_Score"]].head(50), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_forecast:
    ts_df, fc_df = compute_forecast(dff, forecast_days)
    if not ts_df.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=ts_df["Date"], y=ts_df["Smoothed"], name="Smoothed Trend", line=dict(color="#00d4ff", width=2)))
        fig_fc.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], name="Forecast", line=dict(color="#00ffb2", width=3, dash="dot")))
        apply_glass(fig_fc, f"Demand Forecast ({forecast_days} Days)", 500)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_export:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Export Ready</div>', unsafe_allow_html=True)
    export_df = dff.copy()
    
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    st.download_button("💾 Download CSV", data=buf.getvalue().encode("utf-8"), file_name="export.csv", mime="text/csv")
    st.dataframe(export_df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
