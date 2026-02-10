"""
CardioScreen Pro — Cardiovascular Disease Risk Prediction
Streamlit Clinical Dashboard  •  v3.1
Gradient Boosting Pipeline  •  scikit-learn
Author: Matthias
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ━━━━━━━━━━━━━  PAGE  ━━━━━━━━━━━━━
st.set_page_config(
    page_title="CardioScreen Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

# ━━━━━━━━━━━━━  CSS  ━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.main { background: #0d1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 20px !important;
}

/* Primary action button */
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.6em;
    background: linear-gradient(135deg, #d63031, #6c5ce7);
    color: #fff;
    font-weight: 700;
    font-size: 0.92rem;
    border: none;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    box-shadow: 0 6px 20px rgba(214,48,49,0.2);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(108,92,231,0.3);
}

/* Risk badge */
.badge-high, .badge-low {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 1px;
}
.badge-high { background: rgba(214,48,49,0.14); color: #ff6b6b; border: 1px solid rgba(214,48,49,0.3); }
.badge-low  { background: rgba(0,206,120,0.10); color: #00ce78; border: 1px solid rgba(0,206,120,0.25); }

/* Score */
.score-high {
    font-size: 4.2rem; font-weight: 900; line-height: 1; margin: 4px 0 0 0;
    background: linear-gradient(180deg, #fff 25%, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.score-low {
    font-size: 4.2rem; font-weight: 900; line-height: 1; margin: 4px 0 0 0;
    background: linear-gradient(180deg, #fff 25%, #00ce78);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* Protocol cards */
.proto {
    background: rgba(255,255,255,0.025);
    padding: 13px 16px;
    border-radius: 10px;
    border-left: 4px solid #6c5ce7;
    margin-bottom: 9px;
    font-size: 0.88rem;
}
.proto.urgent { border-left-color: #d63031; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 5px; }

hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━  MODEL  ━━━━━━━━━━━━━
@st.cache_resource
def load_assets():
    try:
        base = os.path.dirname(__file__)
        return (
            joblib.load(os.path.join(base, "cardio_pipeline.pkl")),
            joblib.load(os.path.join(base, "feature_names.pkl")),
        )
    except Exception:
        return None, None

pipeline, feature_names = load_assets()

# ━━━━━━━━━━━━━  CHARTS  ━━━━━━━━━━━━━
def make_gauge(value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 40, "color": "white", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "rgba(255,255,255,0.15)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)", "size": 10}},
            "bar": {"color": color, "thickness": 0.65},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "rgba(0,206,120,0.07)"},
                {"range": [35, 65], "color": "rgba(255,193,7,0.05)"},
                {"range": [65, 100],"color": "rgba(214,48,49,0.07)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Inter"},
        height=220, margin=dict(l=25, r=25, t=25, b=5),
    )
    return fig

def make_radar(bmi_n, bp_n, chol_n, gluc_n, smoke_n, inact_n):
    cats = ["BMI", "Blood Pressure", "Cholesterol", "Glucose", "Smoking", "Inactivity"]
    vals = [bmi_n, bp_n, chol_n, gluc_n, smoke_n, inact_n]
    vals += [vals[0]]; cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(108,92,231,0.12)",
        line=dict(color="#6c5ce7", width=2),
        marker=dict(size=5, color="#a29bfe"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.05)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                             tickfont=dict(color="rgba(255,255,255,0.6)", size=11)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=280, margin=dict(l=45, r=45, t=25, b=25),
        showlegend=False,
    )
    return fig

# ━━━━━━━━━━━━━  SIDEBAR  ━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:8px 0;'>"
        "<span style='font-size:2rem;'>🫀</span><br>"
        "<span style='font-weight:800; font-size:1.3rem; color:#ff6b6b;'>CARDIO</span>"
        "<span style='font-weight:800; font-size:1.3rem; color:#c9d1d9;'>SCREEN</span>"
        "<span style='font-weight:300; font-size:1.3rem; color:#6c5ce7;'> PRO</span><br>"
        "<span style='font-size:0.65rem; color:#484f58; letter-spacing:2px;'>CLINICAL DIAGNOSTIC SUITE</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.expander("👤  Patient", expanded=True):
        age = st.slider("Age (years)", 30, 80, 50)
        gender = st.selectbox("Sex", [1, 2],
                              format_func=lambda x: "♀ Female" if x == 1 else "♂ Male")

    with st.expander("📏  Biometrics", expanded=True):
        c1, c2 = st.columns(2)
        height = c1.number_input("Height (cm)", 120, 220, 165)
        weight = c2.number_input("Weight (kg)", 30, 200, 70)

    with st.expander("🩸  Vitals & Labs", expanded=True):
        ap_hi = st.slider("Systolic BP (mmHg)", 60, 250, 120)
        ap_lo = st.slider("Diastolic BP (mmHg)", 30, 150, 80)
        chol = st.select_slider("Cholesterol", [1, 2, 3], 1,
                                format_func=lambda x: ["Normal", "Elevated", "High"][x-1])
        gluc = st.select_slider("Glucose", [1, 2, 3], 1,
                                format_func=lambda x: ["Normal", "Elevated", "High"][x-1])

    with st.expander("🚬  Lifestyle", expanded=True):
        smoke  = st.toggle("Smoker")
        alco   = st.toggle("Alcohol")
        active = st.toggle("Physically Active", value=True)

# ━━━━━━━━━━━━━  DERIVED VALUES  ━━━━━━━━━━━━━
bmi = round(weight / ((height / 100) ** 2), 1)
pp  = ap_hi - ap_lo
mAP = round(ap_lo + pp / 3, 1)
hbp = int(ap_hi >= 140 or ap_lo >= 90)

# Normalised risk factors (0 = optimal, 1 = worst)
bmi_n   = min(max(bmi - 18.5, 0) / 20, 1.0)
bp_n    = min(max(ap_hi - 90, 0) / 160, 1.0)
chol_n  = (chol - 1) / 2
gluc_n  = (gluc - 1) / 2
smoke_n = float(smoke)
inact_n = 0.0 if active else 1.0

# ━━━━━━━━━━━━━  HEADER  ━━━━━━━━━━━━━
st.markdown(
    "<h2 style='margin-bottom:0; font-weight:800;'>Clinical Dashboard</h2>"
    "<p style='color:#8b949e; margin-top:2px; font-size:0.85rem;'>"
    "Cardiovascular risk stratification — enter patient data in the sidebar, then run the analysis.</p>",
    unsafe_allow_html=True,
)
st.markdown("")

# ━━━━━━━━━━━━━  KPI ROW  ━━━━━━━━━━━━━
k1, k2, k3, k4 = st.columns(4)
k1.metric("BMI", bmi, delta=f"{bmi-24.9:+.1f}" if bmi > 24.9 else "Healthy", delta_color="inverse")
k2.metric("Blood Pressure", f"{ap_hi}/{ap_lo}",
          delta="Hypertensive" if hbp else "Optimal", delta_color="inverse")
k3.metric("MAP", f"{mAP} mmHg")
k4.metric("Age", f"{age} yr")

st.markdown("")

# ━━━━━━━━━━━━━  PREDICTION  ━━━━━━━━━━━━━
if ap_lo >= ap_hi:
    st.error("⛔ **Diastolic BP ≥ Systolic BP** — correct the values in the sidebar before running.")
else:
    if st.button("🔬  RUN DIAGNOSTIC ANALYSIS"):
        if pipeline and feature_names:
            row = {
                "age": age, "gender": gender, "height": height, "weight": weight,
                "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                "smoke": int(smoke), "alco": int(alco), "active": int(active),
                "bmi": bmi, "pulse_pressure": pp, "high_bp_flag": hbp, "map": mAP,
            }
            prob = pipeline.predict_proba(pd.DataFrame([row])[feature_names])[0][1] * 100
            st.session_state.prediction_results = {
                "prob": prob,
                "risk": "HIGH" if prob > 50 else "LOW",
                "ts": datetime.now().strftime("%H:%M:%S"),
            }
        else:
            st.error("Model files missing — ensure `cardio_pipeline.pkl` and `feature_names.pkl` exist.")

# ━━━━━━━━━━━━━  RESULTS  ━━━━━━━━━━━━━
if st.session_state.prediction_results:
    res = st.session_state.prediction_results
    high = res["prob"] > 50
    color = "#ff6b6b" if high else "#00ce78"

    st.divider()

    # ── Score + Gauge side by side ──
    left, right = st.columns([1, 1])

    with left:
        badge = "badge-high" if high else "badge-low"
        score = "score-high" if high else "score-low"
        st.markdown(f'<span class="{badge}">{res["risk"]} RISK</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="{score}">{res["prob"]:.1f}%</p>', unsafe_allow_html=True)
        st.caption(f'Analysis at {res["ts"]}')
        st.markdown("")
        if high:
            st.markdown(
                "**Interpretation:** Multiple risk indicators exceed clinical thresholds. "
                "A comprehensive cardiovascular workup and specialist referral are recommended."
            )
        else:
            st.markdown(
                "**Interpretation:** Current profile falls within acceptable limits. "
                "Continue preventive care and schedule routine follow-up."
            )

    with right:
        st.plotly_chart(make_gauge(res["prob"], color), use_container_width=True)

    st.divider()

    # ── Tabs: Protocols • Risk Profile • Hemodynamics ──
    tab_proto, tab_radar, tab_hemo = st.tabs([
        "📋  Protocols", "🕸️  Risk Profile", "📊  Hemodynamics",
    ])

    with tab_proto:
        st.markdown("#### Clinical Action Items")
        p1, p2 = st.columns(2)
        has_flag = False

        with p1:
            if bmi > 25:
                has_flag = True
                st.markdown(
                    '<div class="proto urgent">⚖️ <b>Weight Management</b> — '
                    'Target BMI &lt; 25 via structured dietary plan and exercise.</div>',
                    unsafe_allow_html=True)
            if hbp:
                has_flag = True
                st.markdown(
                    '<div class="proto urgent">🩸 <b>Hypertension</b> — '
                    'Start 24-hr ambulatory BP monitoring. Evaluate antihypertensive therapy.</div>',
                    unsafe_allow_html=True)
            if chol > 1:
                has_flag = True
                st.markdown(
                    '<div class="proto">🧪 <b>Lipid Management</b> — '
                    'Order fasting lipid panel. Consider statins if LDL &gt; 130.</div>',
                    unsafe_allow_html=True)
        with p2:
            if smoke:
                has_flag = True
                st.markdown(
                    '<div class="proto urgent">🚬 <b>Smoking Cessation</b> — '
                    'Initiate NRT and behavioural counseling.</div>',
                    unsafe_allow_html=True)
            if not active:
                has_flag = True
                st.markdown(
                    '<div class="proto">🏃 <b>Exercise Rx</b> — '
                    '≥ 150 min/wk moderate aerobic activity; gradual 4-week ramp.</div>',
                    unsafe_allow_html=True)
            if gluc > 1:
                has_flag = True
                st.markdown(
                    '<div class="proto">🔬 <b>Glycemic Control</b> — '
                    'Order HbA1c. Consider metformin if prediabetic range.</div>',
                    unsafe_allow_html=True)

        if not has_flag:
            st.success("All indicators within optimal ranges — maintain current lifestyle.")

    with tab_radar:
        st.markdown("#### Risk Factor Fingerprint")
        st.plotly_chart(make_radar(bmi_n, bp_n, chol_n, gluc_n, smoke_n, inact_n),
                        use_container_width=True)
        st.caption(
            "Each axis is normalised 0 → 1 (optimal → highest risk). "
            "A smaller polygon = healthier profile."
        )

    with tab_hemo:
        st.markdown("#### Hemodynamic Reference")
        ok = lambda v: "✅" if v else "⚠️"
        st.dataframe(
            pd.DataFrame({
                "Indicator": ["BMI", "Systolic BP", "Diastolic BP", "MAP", "Pulse Pressure"],
                "Value": [bmi, ap_hi, ap_lo, mAP, pp],
                "Target": ["18.5 – 24.9", "< 130", "< 85", "70 – 100", "30 – 50"],
                "Status": [
                    ok(18.5 <= bmi <= 24.9), ok(ap_hi < 130),
                    ok(ap_lo < 85), ok(70 <= mAP <= 100), ok(30 <= pp <= 50),
                ],
            }),
            hide_index=True, use_container_width=True,
        )

else:
    st.divider()
    st.info("👋 **Ready** — configure patient data in the sidebar, then click **Run Diagnostic Analysis**.")

# ━━━━━━━━━━━━━  ABOUT & DISCLAIMER  ━━━━━━━━━━━━━
with st.expander("ℹ️  About this tool"):
    st.markdown(
        "**CardioScreen Pro** uses a Gradient Boosting classifier (scikit-learn) "
        "trained on 70 000+ patient records from the "
        "[Kaggle Cardiovascular Disease dataset]"
        "(https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). "
        "Model accuracy ≈ 73.5 %."
    )
    st.markdown(
        "⚕️ **Disclaimer:** This tool is for educational screening purposes only. "
        "It does **not** replace professional medical advice, diagnosis, or treatment."
    )

# ━━━━━━━━━━━━━  FOOTER  ━━━━━━━━━━━━━
st.markdown(
    "<p style='text-align:center; color:#484f58; font-size:0.72rem; margin-top:30px;'>"
    "© 2026 CardioScreen Pro v3.1  •  Streamlit & scikit-learn</p>",
    unsafe_allow_html=True,
)
