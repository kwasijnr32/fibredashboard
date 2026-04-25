"""
Optical Fiber Predictive Maintenance Dashboard
AI-Based Fault Detection using Attenuation & PMD Time-Series Forecasting
Final Year Project | Self-contained Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
THETA_1 = 0.40        # Low → Medium risk boundary
THETA_2 = 0.70        # Medium → High risk boundary
W_A     = 0.6         # Attenuation weight in health index
W_D     = 0.4         # PMD weight in health index

RISK_COLORS = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
RISK_BG     = {"Low": "#E8F5E9", "Medium": "#FFF3E0", "High": "#FFEBEE"}


# ═══════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════
def _stress_events(T, n_events):
    """Generate exponentially decaying stress spikes."""
    events = np.zeros(T)
    times  = np.random.choice(T, size=n_events, replace=False)
    for t in times:
        dur = np.random.randint(3, 15)
        mag = np.random.uniform(0.05, 0.25)
        end = min(t + dur, T)
        decay = np.exp(-np.arange(end - t) / max(dur / 3, 1))
        events[t:end] += mag * decay
    return events


def generate_dataset(T=400, n_links=5):
    """
    Synthetic fiber degradation dataset.
    Each link: slow aging drift + Gaussian noise + random stress bursts.
    Returns DataFrame: timestamp | link_id | attenuation_dB_km | pmd_ps_sqkm
    """
    np.random.seed(42)
    start = datetime(2023, 1, 1)
    timestamps = [start + timedelta(hours=6 * i) for i in range(T)]
    rows = []
    for lid in range(1, n_links + 1):
        A0    = np.random.uniform(0.18, 0.22)
        D0    = np.random.uniform(0.05, 0.12)
        alpha = np.random.uniform(1e-5, 5e-5)
        beta  = np.random.uniform(5e-6, 2e-5)
        t     = np.arange(T)
        A = (A0 + alpha * t
             + np.random.normal(0, 0.003, T)
             + _stress_events(T, np.random.randint(3, 8)))
        D = (D0 + beta  * t
             + np.random.normal(0, 0.002, T)
             + _stress_events(T, np.random.randint(3, 8)) * 0.5)
        A = np.clip(A, 0.15, 0.60)
        D = np.clip(D, 0.02, 0.50)
        for i in range(T):
            rows.append({
                "timestamp":         timestamps[i],
                "link_id":           f"Link_{lid:02d}",
                "attenuation_dB_km": round(float(A[i]), 5),
                "pmd_ps_sqkm":       round(float(D[i]), 5),
            })
    df = pd.DataFrame(rows).sort_values(["link_id", "timestamp"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════
# PREPROCESSING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def preprocess(df):
    """
    Clean → engineer features → compute health index → assign risk class.
    Returns enriched DataFrame ready for modelling.
    """
    df = df.copy().sort_values(["link_id", "timestamp"])

    # ── Clean ──────────────────────────────────────────────
    for col in ["attenuation_dB_km", "pmd_ps_sqkm"]:
        df[col] = df[col].ffill().bfill()
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(Q1 - 3 * (Q3 - Q1), Q3 + 3 * (Q3 - Q1))

    # ── Feature engineering ────────────────────────────────
    extras = {}
    for col in ["attenuation_dB_km", "pmd_ps_sqkm"]:
        for w in [6, 12, 24]:
            extras[f"{col}_roll_mean_{w}"] = (
                df.groupby("link_id")[col]
                  .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            extras[f"{col}_roll_std_{w}"] = (
                df.groupby("link_id")[col]
                  .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            )
            def _make_slope(w=w):
                def _slope_transform(x):
                    def _slope(v):
                        if len(v) < 2: return 0.0
                        return float(np.polyfit(np.arange(len(v)), v, 1)[0])
                    return x.rolling(w, min_periods=2).apply(_slope, raw=True).fillna(0)
                return _slope_transform
            extras[f"{col}_slope_{w}"] = (
                df.groupby("link_id")[col].transform(_make_slope(w))
            )
        for lag in [1, 3, 6]:
            extras[f"{col}_lag_{lag}"] = df.groupby("link_id")[col].shift(lag)
        extras[f"{col}_change"] = df.groupby("link_id")[col].diff().fillna(0)

    df = pd.concat([df, pd.DataFrame(extras, index=df.index)], axis=1)
    df = df.dropna().reset_index(drop=True)

    # ── Health index H(t) ──────────────────────────────────
    sc = MinMaxScaler()
    A_n = sc.fit_transform(df[["attenuation_dB_km"]]).flatten()
    D_n = sc.fit_transform(df[["pmd_ps_sqkm"]]).flatten()
    df["health_index"] = W_A * A_n + W_D * D_n

    df["risk_class"] = pd.cut(
        df["health_index"],
        bins=[-np.inf, THETA_1, THETA_2, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return df


# ═══════════════════════════════════════════════════════════
# RANDOM FOREST MODEL
# ═══════════════════════════════════════════════════════════
class RFModel:
    """Random Forest regressor for attenuation + PMD forecasting."""

    TARGET_COLS = ["attenuation_dB_km", "pmd_ps_sqkm"]

    def __init__(self, n_estimators=150):
        self.scaler       = StandardScaler()
        self.models       = {}
        self.feature_cols = None
        self.n_estimators = n_estimators

    def fit(self, X, y, feature_cols):
        self.feature_cols = feature_cols
        Xs = self.scaler.fit_transform(X)
        for i, col in enumerate(self.TARGET_COLS):
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42, n_jobs=-1
            )
            rf.fit(Xs, y[:, i])
            self.models[col] = rf

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return np.column_stack(
            [self.models[c].predict(Xs) for c in self.TARGET_COLS]
        )

    def predict_with_intervals(self, X, alpha=0.10):
        Xs = self.scaler.transform(X)
        out = {}
        for col in self.TARGET_COLS:
            tree_preds = np.array(
                [t.predict(Xs) for t in self.models[col].estimators_]
            )
            out[col] = {
                "mean":  tree_preds.mean(axis=0),
                "lower": np.percentile(tree_preds, 100 * alpha / 2,       axis=0),
                "upper": np.percentile(tree_preds, 100 * (1 - alpha / 2), axis=0),
            }
        return out

    def feature_importance(self):
        rows = []
        for col in self.TARGET_COLS:
            for feat, imp in zip(
                self.feature_cols, self.models[col].feature_importances_
            ):
                rows.append({"target": col, "feature": feat, "importance": imp})
        return (
            pd.DataFrame(rows)
            .sort_values(["target", "importance"], ascending=[True, False])
        )


# ═══════════════════════════════════════════════════════════
# CACHED DATA & MODEL LOADERS
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="📡 Generating fiber dataset…")
def load_data(T, n_links):
    raw = generate_dataset(T=T, n_links=n_links)
    return preprocess(raw)


@st.cache_resource(show_spinner="🤖 Training forecast model…")
def load_model(T, n_links):
    df = load_data(T, n_links)

    EXCLUDE = {
        "timestamp", "link_id",
        "attenuation_dB_km", "pmd_ps_sqkm",
        "health_index", "risk_class",
    }
    fc = [c for c in df.columns if c not in EXCLUDE]
    tc = ["attenuation_dB_km", "pmd_ps_sqkm"]

    horizon = 6
    dfs = df.sort_values(["link_id", "timestamp"]).copy()
    shifted = dfs.groupby("link_id")[tc].shift(-horizon)
    dfs = pd.concat([dfs, shifted.add_suffix("_target")], axis=1).dropna()

    X = dfs[fc].values
    y = dfs[[f"{c}_target" for c in tc]].values

    model = RFModel(n_estimators=150)
    model.fit(X, y, fc)
    return model, fc, dfs


def get_forecast(model, link_shifted_df, feature_cols, steps):
    """Rolling window forecast with confidence intervals for one link."""
    ldf  = link_shifted_df.sort_values("timestamp").reset_index(drop=True)
    feat = ldf[feature_cols].values
    n    = len(feat)
    start = max(0, n - steps)

    fA, fD = [], []
    aLo, aHi, dLo, dHi = [], [], [], []

    for i in range(start, n):
        iv = model.predict_with_intervals(feat[i : i + 1])
        fA.append(iv["attenuation_dB_km"]["mean"][0])
        fD.append(iv["pmd_ps_sqkm"]["mean"][0])
        aLo.append(iv["attenuation_dB_km"]["lower"][0])
        aHi.append(iv["attenuation_dB_km"]["upper"][0])
        dLo.append(iv["pmd_ps_sqkm"]["lower"][0])
        dHi.append(iv["pmd_ps_sqkm"]["upper"][0])

    return {
        "timestamp": ldf["timestamp"].values[start:],
        "A_pred": np.array(fA),  "D_pred": np.array(fD),
        "A_lo":   np.array(aLo), "A_hi":   np.array(aHi),
        "D_lo":   np.array(dLo), "D_hi":   np.array(dHi),
    }


# ═══════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fiber Link Health Monitor",
    page_icon="📡",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
T_steps  = st.sidebar.slider("Time steps per link", 200, 600, 400, step=50)
n_links  = st.sidebar.slider("Number of fiber links", 2, 8, 5)

model, feature_cols, df_shifted = load_model(T_steps, n_links)
df = load_data(T_steps, n_links)

link_ids      = sorted(df["link_id"].unique())
selected_link = st.sidebar.selectbox("🔗 Select Link", link_ids)
show_ci       = st.sidebar.checkbox("Show 90% confidence intervals", value=True)
fc_steps      = st.sidebar.slider("Forecast window (steps)", 12, 60, 30, step=6)

st.sidebar.markdown("---")
st.sidebar.caption("Each step = 6 hours")

# ── Header ─────────────────────────────────────────────────
st.title("📡 Optical Fiber Predictive Maintenance Dashboard")
st.caption(
    "AI-Based Fault Detection · Attenuation & PMD Time-Series Forecasting · Final Year Project"
)

# ── KPI cards ──────────────────────────────────────────────
latest = (
    df.sort_values("timestamp")
    .groupby("link_id")
    .last()
    .reset_index()
)
kpi_cols = st.columns(len(link_ids))
for col_ui, lid in zip(kpi_cols, link_ids):
    row   = latest[latest["link_id"] == lid].iloc[0]
    risk  = str(row["risk_class"])
    color = RISK_COLORS.get(risk, "#9E9E9E")
    bg    = RISK_BG.get(risk, "#fafafa")
    col_ui.markdown(
        f"""
        <div style="background:{bg};border-left:5px solid {color};
                    padding:10px 14px;border-radius:7px;margin-bottom:4px;">
          <b style="font-size:0.95em">{lid}</b><br>
          <span style="color:{color};font-weight:700;font-size:1.05em">
            {risk} Risk
          </span><br>
          <small>
            A: {row['attenuation_dB_km']:.4f} dB/km<br>
            PMD: {row['pmd_ps_sqkm']:.4f} ps/√km<br>
            H(t): {row['health_index']:.3f}
          </small>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Selected link ──────────────────────────────────────────
link_df      = df[df["link_id"] == selected_link].sort_values("timestamp")
ldf_shifted  = df_shifted[df_shifted["link_id"] == selected_link]
forecast     = get_forecast(model, ldf_shifted, feature_cols, steps=fc_steps)

st.subheader(f"🔍 {selected_link} — Detailed Analysis")
tab1, tab2, tab3 = st.tabs(
    ["📈 Time Series & Forecast", "⚠️ Risk Analysis", "📊 Feature Insights"]
)

# ── Tab 1 : Time Series & Forecast ────────────────────────
with tab1:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Attenuation (dB/km)", "PMD (ps/√km)"),
        vertical_spacing=0.10,
    )
    disp_n  = min(300, len(link_df))
    ts_hist = link_df["timestamp"].values[-disp_n:]

    traces = [
        ("attenuation_dB_km", "A_pred", "A_lo", "A_hi", "#1976D2", 1),
        ("pmd_ps_sqkm",       "D_pred", "D_lo", "D_hi", "#7B1FA2", 2),
    ]
    for col_name, y_key, lo_key, hi_key, color, row_i in traces:
        # Actual
        fig.add_trace(
            go.Scatter(
                x=ts_hist,
                y=link_df[col_name].values[-disp_n:],
                name=f"Actual ({col_name.split('_')[0].capitalize()})",
                line=dict(color=color, width=1.8),
            ),
            row=row_i, col=1,
        )
        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"], y=forecast[y_key],
                name="Forecast",
                line=dict(color="#E53935", width=2, dash="dash"),
                showlegend=(row_i == 1),
            ),
            row=row_i, col=1,
        )
        # Confidence band
        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(
                        [forecast["timestamp"], forecast["timestamp"][::-1]]
                    ),
                    y=np.concatenate(
                        [forecast[hi_key], forecast[lo_key][::-1]]
                    ),
                    fill="toself",
                    fillcolor="rgba(229,57,53,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="90% CI",
                    showlegend=(row_i == 1),
                ),
                row=row_i, col=1,
            )

    fig.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 : Risk Analysis ──────────────────────────────────
with tab2:
    c1, c2 = st.columns([1, 2])

    with c1:
        h_val     = float(link_df["health_index"].values[-1])
        risk_now  = str(link_df["risk_class"].values[-1])
        color_now = RISK_COLORS.get(risk_now, "#9E9E9E")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(h_val, 3),
            title={
                "text": (
                    f"Health Index H(t)<br>"
                    f"<span style='color:{color_now}'>{risk_now} Risk</span>"
                )
            },
            gauge={
                "axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0,       THETA_1], "color": "#C8E6C9"},
                    {"range": [THETA_1, THETA_2], "color": "#FFE0B2"},
                    {"range": [THETA_2, 1],       "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line":      {"color": color_now, "width": 4},
                    "thickness": 0.75,
                    "value":     h_val,
                },
                "bar": {"color": color_now},
            },
        ))
        gauge.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        if risk_now == "High":
            st.error("🚨 **HIGH RISK** — Schedule immediate inspection!")
        elif risk_now == "Medium":
            st.warning("⚠️ **MEDIUM RISK** — Monitor closely. Plan maintenance.")
        else:
            st.success("✅ **LOW RISK** — Link operating normally.")

    with c2:
        risk_ts = link_df[["timestamp", "health_index", "risk_class"]].copy()
        fig2    = go.Figure()
        for rl, clr in RISK_COLORS.items():
            mask = risk_ts["risk_class"] == rl
            fig2.add_trace(go.Scatter(
                x=risk_ts[mask]["timestamp"],
                y=risk_ts[mask]["health_index"],
                mode="markers",
                name=rl,
                marker=dict(color=clr, size=4),
            ))
        fig2.add_hline(y=THETA_1, line_dash="dot", line_color="#FF9800",
                       annotation_text="θ₁ (Low→Medium)")
        fig2.add_hline(y=THETA_2, line_dash="dot", line_color="#F44336",
                       annotation_text="θ₂ (Medium→High)")
        fig2.update_layout(
            title="Health Index H(t) Over Time",
            yaxis_title="H(t)",
            height=300,
            legend=dict(orientation="h", y=-0.28),
        )
        st.plotly_chart(fig2, use_container_width=True)

    risk_counts = (
        link_df["risk_class"]
        .value_counts()
        .reset_index()
        .rename(columns={"risk_class": "Risk", "count": "Count"})
    )
    fig3 = px.pie(
        risk_counts, names="Risk", values="Count",
        color="Risk", color_discrete_map=RISK_COLORS,
        title="Risk Class Distribution",
    )
    fig3.update_layout(height=280)
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3 : Feature Insights ───────────────────────────────
with tab3:
    st.markdown("#### Top Predictive Features (Random Forest)")
    fi = model.feature_importance()
    for tgt, label in [
        ("attenuation_dB_km", "Attenuation"),
        ("pmd_ps_sqkm",       "PMD"),
    ]:
        sub    = fi[fi["target"] == tgt].head(12)
        fig_fi = px.bar(
            sub, x="importance", y="feature", orientation="h",
            title=f"Feature Importance — {label}",
            color="importance", color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "Final Year Project · AI-Based Predictive Fault Detection in Optical Fiber Networks"
)
