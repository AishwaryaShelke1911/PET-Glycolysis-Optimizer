"""
app.py — PET Glycolysis ML Optimizer
A clean 5-tab Streamlit dashboard for predicting and optimizing
catalytic PET glycolysis conditions using a literature-mined ML model.
"""

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from predict import predict_yield
from preprocessing import get_catalyst_list
from optimize import recommend_conditions
from feature_engineering import KNOWN_EA_KJ_MOL

BASE_DIR    = Path(__file__).resolve().parent.parent
META_PATH   = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH   = BASE_DIR / "data" / "processed" / "glycolysis_clean.csv"
MODEL_PATH  = BASE_DIR / "models" / "best_model.pkl"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PET Glycolysis ML Optimizer",
    page_icon="♻️",
    layout="wide",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  h1 { color: #1E3A5F; }
  h3 { color: #2E86AB; margin-bottom: 0.2rem; }
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
  .info-box {
    background: #EBF4FA; border-left: 4px solid #2E86AB;
    border-radius: 6px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.88rem; color: #1E3A5F;
  }
  .physics-box {
    background: #F0FDF4; border: 1px solid #86EFAC;
    border-radius: 6px; padding: 12px 16px; margin: 8px 0;
    font-family: monospace; font-size: 0.88rem; color: #166534;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("♻️ PET Glycolysis ML Optimizer")
st.caption("Physics-informed ML for catalytic PET recycling · Literature dataset from 11 peer-reviewed papers (2021–2026)")
st.divider()

# ── Shared data (cached so it loads once) ───────────────────────────────────
@st.cache_resource
def get_catalysts():
    try:
        return get_catalyst_list()
    except Exception:
        return []

@st.cache_data
def get_dataset():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

@st.cache_data
def get_metadata():
    return json.load(open(META_PATH)) if META_PATH.exists() else {}

catalysts   = get_catalysts()
df          = get_dataset()
meta        = get_metadata()
model_ready = MODEL_PATH.exists() and len(catalysts) > 0

if not model_ready:
    st.error("Model not trained. Run: `python src/data_loader.py` then `python src/train_model.py`")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Yield",
    "🎯 Recommend Conditions",
    "📈 Model Performance",
    "🔬 Dataset Explorer",
    "⚖️ Catalyst Comparison",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT YIELD
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict BHET Yield for Given Reaction Conditions")
    st.caption("Select a real catalyst and reaction conditions. The model returns a predicted yield with a ±uncertainty band.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        catalyst = st.selectbox("Catalyst", catalysts)
        temp     = st.slider("Temperature (°C)", 100, 260, 195, step=5)
        time_min = st.slider("Reaction time (min)", 5, 360, 120, step=5)
        dosage   = st.slider("Catalyst dosage (wt%)", 0.1, 20.0, 1.0, step=0.1)
        eg_ratio = st.slider("EG : PET ratio (mass)", 1.0, 15.0, 5.0, step=0.5)
        clicked  = st.button("Predict Yield", type="primary", use_container_width=True)

    with right:
        if clicked:
            with st.spinner("Predicting..."):
                result = predict_yield(catalyst, dosage, temp, time_min, eg_ratio)

            pred = result["predicted_yield"]
            lo   = result["lower"]
            hi   = result["upper"]

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={"text": "Predicted BHET Yield (%)", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1E3A5F"},
                    "steps": [
                        {"range": [0,  40], "color": "#FEE2E2"},
                        {"range": [40, 70], "color": "#FEF3C7"},
                        {"range": [70,100], "color": "#D1FAE5"},
                    ],
                }
            ))
            fig_g.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Yield",     f"{pred:.1f}%")
            c2.metric("Lower bound (80% CI)", f"{lo:.1f}%")
            c3.metric("Upper bound (80% CI)", f"{hi:.1f}%")

            # Arrhenius explanation
            ea_val = None
            for key, val in KNOWN_EA_KJ_MOL.items():
                if key in catalyst.lower():
                    ea_val = val
                    break

            st.markdown("#### ⚗️ Physics Feature Used by Model")
            if ea_val:
                log_k = -ea_val * 1000 / (8.314 * (temp + 273.15))
                st.markdown(f"""<div class="physics-box">
Catalyst Ea (from literature) = {ea_val} kJ/mol<br>
log(k) = −Ea / (R × T) = −{ea_val*1000:.0f} / (8.314 × {temp+273.15:.1f}) = <b>{log_k:.4f}</b><br><br>
This Arrhenius feature captures how temperature and catalyst activation energy combine
to drive the reaction rate — more informative than raw temperature alone.
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="info-box">
No literature activation energy (Ea) found for <b>{catalyst}</b>.<br>
The model uses the <b>dataset median Ea</b> as a fallback for this catalyst.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">👈 Set conditions on the left and click <b>Predict Yield</b></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RECOMMEND CONDITIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Find Optimal Conditions for a Catalyst")
    st.caption("Searches temperature / time / dosage / EG ratio combinations within the range actually tested for this catalyst in the literature.")

    left2, right2 = st.columns([1, 2], gap="large")

    with left2:
        opt_cat  = st.selectbox("Catalyst", catalysts, key="opt_cat")
        top_n    = st.slider("Number of results", 3, 10, 5)
        n_pts    = st.slider("Grid resolution", 5, 12, 8,
                              help="Controls how finely the search grid is sampled. Higher = slower but more precise.")
        go_btn   = st.button("Find Best Conditions", type="primary", use_container_width=True)

    with right2:
        if go_btn:
            with st.spinner("Searching..."):
                res = recommend_conditions(opt_cat, n_points=n_pts, top_n=top_n)

            display = res[["Temp_C","Time_min","Catalyst_Dosage_wt%","EG_PET_Ratio","Predicted_BHET_Yield_pct"]].copy()
            display.columns = ["Temp (°C)","Time (min)","Dosage (wt%)","EG:PET Ratio","Predicted Yield (%)"]
            display["Predicted Yield (%)"] = display["Predicted Yield (%)"].round(1)
            display["Temp (°C)"]           = display["Temp (°C)"].round(0).astype(int)
            display["Time (min)"]          = display["Time (min)"].round(0).astype(int)
            display[["Dosage (wt%)","EG:PET Ratio"]] = display[["Dosage (wt%)","EG:PET Ratio"]].round(2)

            st.dataframe(display, use_container_width=True, hide_index=True)

            fig = px.bar(display, x=display.index.astype(str), y="Predicted Yield (%)",
                          color="Predicted Yield (%)", color_continuous_scale="Blues",
                          text="Predicted Yield (%)",
                          labels={"x": "Rank"},
                          title=f"Top {top_n} Conditions — {opt_cat}")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, showlegend=False,
                               coloraxis_showscale=False, margin=dict(t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="info-box">👈 Select a catalyst and click <b>Find Best Conditions</b></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance — 5-Fold Cross-Validation")
    st.caption("5-fold CV reports mean ± std across folds, which is the correct way to evaluate a small 71-row dataset. A single train/test split would give an unreliable R² that can swing by ±0.3 just from random chance.")

    if meta and "cv_results" in meta:
        cv   = meta["cv_results"]
        best = meta.get("best_model", "")

        rows = [{"Model": m,
                 "R² mean": round(cv[m]["r2_mean"],3),
                 "R² ±std": round(cv[m]["r2_std"],3),
                 "RMSE mean": round(cv[m]["rmse_mean"],2),
                 "RMSE ±std": round(cv[m]["rmse_std"],2),
                 "MAE mean":  round(cv[m]["mae_mean"],2)}
                for m in cv]
        cv_df = pd.DataFrame(rows)
        st.dataframe(cv_df.style.highlight_max(subset=["R² mean"], color="#D1FAE5")
                              .highlight_min(subset=["RMSE mean"], color="#D1FAE5"),
                     use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_r2 = go.Figure([
                go.Bar(name=m, x=[m], y=[cv[m]["r2_mean"]],
                       error_y=dict(type="data", array=[cv[m]["r2_std"]], visible=True),
                       marker_color="#2E86AB" if m == best else "#94A3B8")
                for m in cv])
            fig_r2.update_layout(title="R² across models (mean ± std)",
                                  yaxis_title="R²", height=320,
                                  showlegend=False, margin=dict(t=50,b=20))
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_b:
            try:
                model = joblib.load(MODEL_PATH)
                if hasattr(model, "feature_importances_"):
                    feats = meta["features"]
                    fi_df = pd.DataFrame({"Feature": feats,
                                          "Importance": model.feature_importances_})
                    fi_df = fi_df.sort_values("Importance", ascending=True)
                    fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                                    orientation="h", color="Importance",
                                    color_continuous_scale="Blues",
                                    title=f"Feature Importance — {best}")
                    fig_fi.update_layout(height=320, margin=dict(t=50,b=20),
                                          coloraxis_showscale=False)
                    st.plotly_chart(fig_fi, use_container_width=True)
            except Exception:
                pass

        st.markdown(f"""<div class="info-box">
<b>Best model: {best}</b> (R² = {cv[best]['r2_mean']:.3f} ± {cv[best]['r2_std']:.3f})<br><br>
R² ≈ 0.23–0.32 is expected for a 71-row multi-source literature dataset — each paper used different
equipment, PET sources, and measurement methods. The model captures broad trends useful for
narrowing down experimental search spaces, not for making precise guarantees.
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Dataset Explorer")

    if not df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows",              len(df))
        m2.metric("Unique catalysts",  df["Catalyst"].nunique())
        m3.metric("Yield range",       f"{df['BHET_Yield_pct'].min():.0f}–{df['BHET_Yield_pct'].max():.0f}%")
        m4.metric("Temp range",        f"{df['Temp_C'].min():.0f}–{df['Temp_C'].max():.0f}°C")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            fig1 = px.histogram(df, x="BHET_Yield_pct", nbins=20,
                                 color_discrete_sequence=["#2E86AB"],
                                 title="BHET Yield Distribution",
                                 labels={"BHET_Yield_pct": "BHET Yield (%)"})
            fig1.update_layout(height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig1, use_container_width=True)

            top_cats = df.groupby("Catalyst")["BHET_Yield_pct"].mean().nlargest(12).reset_index()
            fig2 = px.bar(top_cats, x="BHET_Yield_pct", y="Catalyst",
                           orientation="h", color="BHET_Yield_pct",
                           color_continuous_scale="Blues",
                           title="Average BHET Yield by Catalyst (top 12)",
                           labels={"BHET_Yield_pct": "Avg Yield (%)", "Catalyst": ""})
            fig2.update_layout(height=380, margin=dict(t=40,b=20),
                                coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            fig3 = px.scatter(df.dropna(subset=["Temp_C","BHET_Yield_pct"]),
                               x="Temp_C", y="BHET_Yield_pct", color="Catalyst",
                               opacity=0.75,
                               title="Yield vs Temperature",
                               labels={"Temp_C":"Temperature (°C)","BHET_Yield_pct":"BHET Yield (%)"},
                               hover_data=["Catalyst","Catalyst_Dosage_wt%","Time_min"])
            fig3.update_layout(height=300, margin=dict(t=40,b=20),
                                legend=dict(font=dict(size=9)))
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.scatter(df.dropna(subset=["Time_min","BHET_Yield_pct"]),
                               x="Time_min", y="BHET_Yield_pct", color="Catalyst",
                               opacity=0.75,
                               title="Yield vs Reaction Time",
                               labels={"Time_min":"Time (min)","BHET_Yield_pct":"BHET Yield (%)"})
            fig4.update_layout(height=300, margin=dict(t=40,b=20),
                                legend=dict(font=dict(size=9)))
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("View raw dataset"):
            st.dataframe(df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — CATALYST COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Head-to-Head Catalyst Comparison")
    st.caption("Compare up to 4 catalysts using the literature data collected for this project.")

    if not df.empty:
        available = sorted(df["Catalyst"].dropna().unique().tolist())
        selected  = st.multiselect("Select 2–4 catalysts", available,
                                    default=available[:min(4, len(available))],
                                    max_selections=4)

        if len(selected) >= 2:
            stats = df[df["Catalyst"].isin(selected)].groupby("Catalyst").agg(
                Avg_Yield    = ("BHET_Yield_pct","mean"),
                Max_Yield    = ("BHET_Yield_pct","max"),
                Avg_Temp     = ("Temp_C","mean"),
                Avg_Time_min = ("Time_min","mean"),
                N_datapoints = ("BHET_Yield_pct","count"),
            ).reset_index().round(1)

            st.dataframe(stats, use_container_width=True, hide_index=True)

            col_l, col_r = st.columns(2)
            with col_l:
                # Violin plot
                sub = df[df["Catalyst"].isin(selected)].dropna(subset=["BHET_Yield_pct"])
                fig_v = px.violin(sub, x="Catalyst", y="BHET_Yield_pct",
                                   box=True, points="all", color="Catalyst",
                                   title="Yield Distribution per Catalyst",
                                   labels={"BHET_Yield_pct":"BHET Yield (%)"})
                fig_v.update_layout(height=380, showlegend=False,
                                     margin=dict(t=40,b=20))
                fig_v.update_xaxes(tickangle=20)
                st.plotly_chart(fig_v, use_container_width=True)

            with col_r:
                # Radar chart
                dims  = ["Avg_Yield","Max_Yield","N_datapoints"]
                norm  = stats[dims].copy()
                for c in dims:
                    rng = norm[c].max() - norm[c].min()
                    norm[c] = ((norm[c] - norm[c].min()) / rng * 100) if rng > 0 else 50

                colors = ["#1E3A5F","#2E86AB","#059669","#DC2626"]
                fig_r  = go.Figure()
                for i, row in stats.iterrows():
                    vals = [norm.loc[i, d] for d in dims] + [norm.loc[i, dims[0]]]
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=["Avg Yield","Max Yield","Data Coverage","Avg Yield"],
                        fill="toself",
                        name=row["Catalyst"][:30],
                        line_color=colors[i % len(colors)],
                        opacity=0.65,
                    ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                    title="Normalised Comparison (0–100 scale)",
                    height=380, margin=dict(t=50,b=20),
                )
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Select at least 2 catalysts above to compare.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Sources: Zhang et al. ACS 2024 · Swapna et al. RSC 2024 · Amundarain et al. MDPI 2024 · Gabrič et al. MDPI 2025 · Mehjabeen et al. Elsevier 2026 · Ha et al. Elsevier 2024 · Pan et al. Elsevier 2024 · Hwang & Han Elsevier 2026 · Aer et al. Elsevier 2022 · Zhong-Johnson et al. Nature 2021 · Jo et al. RSC 2023")
