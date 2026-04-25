"""
Streamlit Dashboard — Optical Fiber Predictive Maintenance
Run: streamlit run dashboard/app.py

Features:
- Attenuation / PMD time-series plots
- Forecast curves with confidence intervals
- Risk level visualization & alerts
- Per-link health overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_fiber_dataset
from utils.preprocessing import (
    clean_series, add_features, compute_health_index, THETA_1, THETA_2
)
from models.rf_model import FiberRFModel

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fiber Link Health Monitor",
    page_icon="📡",
    layout="wide",
)

# ── Color palette ─────────────────────────────────────────────────────────────
RISK_COLORS = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
RISK_BG     = {"Low": "#E8F5E9", "Medium": "#FFF3E0", "High": "#FFEBEE"}

# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating fiber dataset...")
def load_data(T=400, n_links=5):
    raw = generate_fiber_dataset(T=T, n_links=n_links)
    clean = clean_series(raw)
    feat = add_features(clean)
    final = compute_health_index(feat)
    return final


@st.cache_resource(show_spinner="Training RF forecast model...")
def train_model(df):
    exclude = {"timestamp", "link_id", "attenuation_dB_km", "pmd_ps_sqkm",
               "health_index", "risk_class", "risk_numeric"}
    feature_cols = [c for c in df.columns if c not in exclude]
    target_cols = ["attenuation_dB_km", "pmd_ps_sqkm"]

    horizon = 6
    df_s = df.sort_values(["link_id", "timestamp"])
    shifted = df_s.groupby("link_id")[target_cols].shift(-horizon)
    df_s = pd.concat([df_s, shifted.add_suffix("_target")], axis=1).dropna()
    X = df_s[feature_cols].values
    y = df_s[[f"{c}_target" for c in target_cols]].values

    model = FiberRFModel(n_estimators=150)
    model.fit(X, y, feature_cols=feature_cols)
    return model, feature_cols, df_s


def get_forecast(model, link_df, feature_cols, steps=48):
    """Generate rolling forecast for a single link."""
    link_df = link_df.sort_values("timestamp").reset_index(drop=True)
    last_features = link_df[feature_cols].values

    forecasts_A, forecasts_D = [], []
    ci_A_lo, ci_A_hi = [], []
    ci_D_lo, ci_D_hi = [], []

    for i in range(len(last_features) - steps, len(last_features)):
        row = last_features[i:i+1]
        intervals = model.predict_with_intervals(row)
        forecasts_A.append(intervals["attenuation_dB_km"]["mean"][0])
        forecasts_D.append(intervals["pmd_ps_sqkm"]["mean"][0])
        ci_A_lo.append(intervals["attenuation_dB_km"]["lower"][0])
        ci_A_hi.append(intervals["attenuation_dB_km"]["upper"][0])
        ci_D_lo.append(intervals["pmd_ps_sqkm"]["lower"][0])
        ci_D_hi.append(intervals["pmd_ps_sqkm"]["upper"][0])

    ts = link_df["timestamp"].values[-steps:]
    return {
        "timestamp": ts,
        "A_pred": np.array(forecasts_A),
        "D_pred": np.array(forecasts_D),
        "A_lo": np.array(ci_A_lo), "A_hi": np.array(ci_A_hi),
        "D_lo": np.array(ci_D_lo), "D_hi": np.array(ci_D_hi),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/fiber-optic.png", width=64)
st.sidebar.title("⚙️ Settings")

T_steps = st.sidebar.slider("Dataset size (time steps / link)", 200, 600, 400, step=50)
n_links = st.sidebar.slider("Number of fiber links", 2, 8, 5)

df = load_data(T=T_steps, n_links=n_links)
model, feature_cols, df_shifted = train_model(df)

link_ids = sorted(df["link_id"].unique())
selected_link = st.sidebar.selectbox("🔗 Select Link", link_ids)
show_ci = st.sidebar.checkbox("Show confidence intervals", value=True)
forecast_window = st.sidebar.slider("Forecast display window (steps)", 12, 60, 30, step=6)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 Optical Fiber Predictive Maintenance Dashboard")
st.caption("AI-Based Fault Detection using Attenuation & PMD Time-Series Forecasting")

# ── KPI Row ────────────────────────────────────────────────────────────────────
latest = df.sort_values("timestamp").groupby("link_id").last().reset_index()

cols = st.columns(len(link_ids))
for i, (col, lid) in enumerate(zip(cols, link_ids)):
    row = latest[latest["link_id"] == lid].iloc[0]
    risk = str(row["risk_class"])
    color = RISK_COLORS.get(risk, "#9E9E9E")
    col.markdown(
        f"""
        <div style="background:{RISK_BG.get(risk,'#fafafa')};border-left:4px solid {color};
        padding:10px 12px;border-radius:6px;">
        <b>{lid}</b><br>
        <span style="color:{color};font-weight:700;font-size:1.1em">{risk} Risk</span><br>
        <small>A: {row['attenuation_dB_km']:.4f} dB/km<br>
        PMD: {row['pmd_ps_sqkm']:.4f} ps/√km<br>
        H: {row['health_index']:.3f}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Selected Link Detail ───────────────────────────────────────────────────────
link_df = df[df["link_id"] == selected_link].sort_values("timestamp")
forecast = get_forecast(model, df_shifted[df_shifted["link_id"] == selected_link],
                        feature_cols, steps=forecast_window)

st.subheader(f"🔍 {selected_link} — Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["📈 Time Series & Forecast", "⚠️ Risk Analysis", "📊 Feature Insights"])

# ── Tab 1: Time Series ─────────────────────────────────────────────────────────
with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Attenuation (dB/km)", "PMD (ps/√km)"),
                         vertical_spacing=0.08)

    # Actual data
    display_n = min(300, len(link_df))
    ts_hist = link_df["timestamp"].values[-display_n:]

    for row_i, (col_name, y_key, lo_key, hi_key, color) in enumerate([
        ("attenuation_dB_km", "A_pred", "A_lo", "A_hi", "#1976D2"),
        ("pmd_ps_sqkm",       "D_pred", "D_lo", "D_hi", "#7B1FA2"),
    ], start=1):
        actual_vals = link_df[col_name].values[-display_n:]

        fig.add_trace(go.Scatter(
            x=ts_hist, y=actual_vals,
            name=f"Actual ({col_name.split('_')[0].capitalize()})",
            line=dict(color=color, width=1.8),
        ), row=row_i, col=1)

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast["timestamp"], y=forecast[y_key],
            name=f"Forecast", line=dict(color="#E53935", width=2, dash="dash"),
            showlegend=(row_i == 1),
        ), row=row_i, col=1)

        if show_ci:
            fig.add_trace(go.Scatter(
                x=np.concatenate([forecast["timestamp"], forecast["timestamp"][::-1]]),
                y=np.concatenate([forecast[hi_key], forecast[lo_key][::-1]]),
                fill="toself", fillcolor="rgba(229,57,53,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="90% CI", showlegend=(row_i == 1),
            ), row=row_i, col=1)

    fig.update_layout(height=500, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Risk Analysis ───────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns([1, 2])

    with c1:
        # Current gauge
        h_val = float(link_df["health_index"].values[-1])
        risk_now = str(link_df["risk_class"].values[-1])
        color_now = RISK_COLORS.get(risk_now, "#9E9E9E")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(h_val, 3),
            title={"text": f"Health Index<br><span style='color:{color_now}'>{risk_now} Risk</span>"},
            gauge={
                "axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0, THETA_1], "color": "#C8E6C9"},
                    {"range": [THETA_1, THETA_2], "color": "#FFE0B2"},
                    {"range": [THETA_2, 1], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": color_now, "width": 4},
                    "thickness": 0.75, "value": h_val
                },
                "bar": {"color": color_now},
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        # Alert box
        if risk_now == "High":
            st.error("🚨 **HIGH RISK ALERT** — Schedule immediate inspection!")
        elif risk_now == "Medium":
            st.warning("⚠️ **MEDIUM RISK** — Monitor closely. Plan maintenance.")
        else:
            st.success("✅ **LOW RISK** — Link operating normally.")

    with c2:
        # Risk class over time
        risk_ts = link_df[["timestamp", "health_index", "risk_class"]].copy()
        risk_ts["risk_numeric"] = risk_ts["risk_class"].map({"Low": 0, "Medium": 1, "High": 2})

        fig2 = go.Figure()
        for risk_level, color in RISK_COLORS.items():
            mask = risk_ts["risk_class"] == risk_level
            fig2.add_trace(go.Scatter(
                x=risk_ts[mask]["timestamp"],
                y=risk_ts[mask]["health_index"],
                mode="markers",
                name=risk_level,
                marker=dict(color=color, size=4),
            ))

        fig2.add_hline(y=THETA_1, line_dash="dot", line_color="#FF9800",
                       annotation_text="θ₁ (Low→Medium)")
        fig2.add_hline(y=THETA_2, line_dash="dot", line_color="#F44336",
                       annotation_text="θ₂ (Medium→High)")

        fig2.update_layout(title="Health Index H(t) Over Time",
                            yaxis_title="H(t)", height=300,
                            legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    # Risk distribution pie
    risk_counts = link_df["risk_class"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    fig3 = px.pie(risk_counts, names="Risk", values="Count",
                  color="Risk", color_discrete_map=RISK_COLORS,
                  title="Risk Class Distribution")
    fig3.update_layout(height=280)
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Feature Insights ────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Top Predictive Features (Random Forest)")
    fi = model.feature_importance()

    for tgt, label in [("attenuation_dB_km", "Attenuation"), ("pmd_ps_sqkm", "PMD")]:
        sub = fi[fi["target"] == tgt].head(12)
        fig_fi = px.bar(sub, x="importance", y="feature", orientation="h",
                         title=f"Feature Importance — {label}",
                         color="importance", color_continuous_scale="Blues")
        fig_fi.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Final Year Project | AI-Based Predictive Fault Detection in Optical Fiber Networks")
