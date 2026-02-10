"""
CardioScreen Pro — Cardiovascular Disease Risk Prediction
Clinical Diagnostic Dashboard  •  Version 3.0
Built with Streamlit  •  scikit-learn Gradient Boosting Pipeline
Author: Matthias
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ━━━━━━━━━━━━━━━━━━━━  PAGE CONFIG  ━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="CardioScreen Pro | CVD Diagnostic Suite",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━  SESSION STATE  ━━━━━━━━━━━━━━━━━━━━
for key, default in [
    ("prediction_results", None),
    ("run_count", 0),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ━━━━━━━━━━━━━━━━━━━━  PREMIUM CSS  ━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(160deg, #0d1117 0%, #101820 50%, #0d1117 100%); }

/* ── Sidebar ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117, #131a24) !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label { color: #c9d1d9 !important; }

/* ── Metric cards ──────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 22px !important;
    transition: border 0.3s;
}
[data-testid="stMetric"]:hover { border: 1px solid rgba(255,75,75,0.25); }

/* ── Primary button ────────────────────────── */
div.stButton > button[kind="primary"],
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.8em;
    background: linear-gradient(135deg, #d63031 0%, #6c5ce7 100%);
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
    border: none;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    box-shadow: 0 8px 24px rgba(214,48,49,0.25);
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 32px rgba(108,92,231,0.35);
    background: linear-gradient(135deg, #e84343 0%, #7e6eea 100%);
}

/* ── Glass card ────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.05);
    padding: 28px 30px;
    margin-bottom: 20px;
}
.glass:hover { border-color: rgba(255,75,75,0.18); }

/* ── Risk badge ────────────────────────────── */
.badge-high {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 1.2px;
    background: rgba(214,48,49,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(214,48,49,0.35);
}
.badge-low {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 1.2px;
    background: rgba(0,206,120,0.12);
    color: #00ce78;
    border: 1px solid rgba(0,206,120,0.3);
}

/* ── Score typography ──────────────────────── */
.score-high {
    font-size: 4.5rem; font-weight: 900; line-height: 1;
    background: linear-gradient(180deg, #ffffff 30%, #ff6b6b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.score-low {
    font-size: 4.5rem; font-weight: 900; line-height: 1;
    background: linear-gradient(180deg, #ffffff 30%, #00ce78 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Protocol card ─────────────────────────── */
.protocol {
    background: rgba(255,255,255,0.025);
    padding: 14px 18px;
    border-radius: 10px;
    border-left: 4px solid #6c5ce7;
    margin-bottom: 10px;
    font-size: 0.92rem;
    color: #c9d1d9;
}
.protocol.urgent { border-left-color: #d63031; }

/* ── Tabs ──────────────────────────────────── */
button[data-baseweb="tab"] { font-weight: 600 !important; letter-spacing: 0.5px; }

/* ── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 6px; }

/* ── Dividers ──────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━  MODEL LOADER  ━━━━━━━━━━━━━━━━━━━━
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


# ━━━━━━━━━━━━━━━━━━━━  CHART HELPERS  ━━━━━━━━━━━━━━━━━━━━
def gauge_chart(value: float, color: str):
    """Semicircular gauge for CVD probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": "%", "font": {"size": 38, "color": "white", "family": "Inter"}},
        delta={"reference": 50, "increasing": {"color": "#ff6b6b"}, "decreasing": {"color": "#00ce78"},
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)",
                     "tickfont": {"color": "rgba(255,255,255,0.5)", "size": 10}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(0,206,120,0.08)"},
                {"range": [30, 60], "color": "rgba(255,193,7,0.06)"},
                {"range": [60, 100], "color": "rgba(214,48,49,0.08)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Inter"},
        height=230, margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


def radar_chart(bmi_norm, bp_norm, chol_norm, gluc_norm, smoke_norm, activity_norm):
    """Spider / radar chart for risk factor profile."""
    cats = ["BMI", "Blood Pressure", "Cholesterol", "Glucose", "Smoking", "Inactivity"]
    vals = [bmi_norm, bp_norm, chol_norm, gluc_norm, smoke_norm, activity_norm]
    vals.append(vals[0])  # close polygon
    cats.append(cats[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(108,92,231,0.15)',
        line=dict(color='#6c5ce7', width=2),
        marker=dict(size=6, color='#a29bfe'),
        name='Patient',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(color="rgba(255,255,255,0.7)", size=11)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        height=300, margin=dict(l=50, r=50, t=30, b=30),
        showlegend=False,
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━  SIDEBAR  ━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 10px 0 0 0;'>"
        "<span style='font-size:2.2rem;'>🫀</span><br>"
        "<span style='font-size:1.4rem; font-weight:800; letter-spacing:-0.5px; color:#ff6b6b;'>"
        "CARDIO</span><span style='font-size:1.4rem; font-weight:800; color:#c9d1d9;'>SCREEN</span>"
        "<span style='font-size:1.4rem; font-weight:300; color:#6c5ce7;'> PRO</span>"
        "<br><span style='font-size:0.7rem; color:#484f58; letter-spacing:2px;'>"
        "CLINICAL DIAGNOSTIC SUITE</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Patient Identity ──
    with st.expander("👤  PATIENT IDENTITY", expanded=True):
        age = st.slider("Age (years)", 30, 80, 50)
        gender = st.selectbox(
            "Biological Sex",
            options=[1, 2],
            format_func=lambda x: "♀  Female" if x == 1 else "♂  Male",
        )

    # ── Biometrics ──
    with st.expander("📏  BIOMETRICS", expanded=True):
        bc1, bc2 = st.columns(2)
        height = bc1.number_input("Height (cm)", 120, 220, 165)
        weight = bc2.number_input("Weight (kg)", 30, 200, 70)

    # ── Vitals & Labs ──
    with st.expander("🩸  VITALS & LABS", expanded=True):
        ap_hi = st.slider("Systolic BP (mmHg)", 60, 250, 120)
        ap_lo = st.slider("Diastolic BP (mmHg)", 30, 150, 80)
        chol = st.select_slider(
            "Cholesterol",
            options=[1, 2, 3],
            value=1,
            format_func=lambda x: ["✅ Normal", "⚠️ Elevated", "🔴 Critical"][x - 1],
        )
        gluc = st.select_slider(
            "Glucose",
            options=[1, 2, 3],
            value=1,
            format_func=lambda x: ["✅ Normal", "⚠️ Elevated", "🔴 Critical"][x - 1],
        )

    # ── Lifestyle ──
    with st.expander("🚬  LIFESTYLE", expanded=True):
        smoke = st.toggle("Nicotine / Tobacco Use")
        alco = st.toggle("Regular Alcohol Intake")
        active = st.toggle("Physical Activity (≥150 min/wk)", value=True)

    st.divider()
    st.caption(f"Session started {datetime.now().strftime('%d %b %Y')}")


# ━━━━━━━━━━━━━━━━━━━━  DERIVED METRICS  ━━━━━━━━━━━━━━━━━━
bmi = round(weight / ((height / 100) ** 2), 1)
pulse_pressure = ap_hi - ap_lo
mean_ap = round(ap_lo + (pulse_pressure / 3), 1)
high_bp = int(ap_hi >= 140 or ap_lo >= 90)

# Normalised 0-1 risk factors for radar chart
bmi_n = min((max(bmi - 18.5, 0)) / 20, 1.0)
bp_n = min(max(ap_hi - 90, 0) / 160, 1.0)
chol_n = (chol - 1) / 2
gluc_n = (gluc - 1) / 2
smoke_n = 1.0 if smoke else 0.0
inactivity_n = 0.0 if active else 1.0


# ━━━━━━━━━━━━━━━━━━━━  MAIN LAYOUT  ━━━━━━━━━━━━━━━━━━━━━
# ── Header ──
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown(
        "<h2 style='margin-bottom:2px; font-weight:800; letter-spacing:-0.5px;'>"
        "Clinical Dashboard</h2>"
        "<p style='color:#8b949e; margin-top:0; font-size:0.88rem;'>"
        "Real-time cardiovascular risk stratification powered by machine learning</p>",
        unsafe_allow_html=True,
    )
with hdr_r:
    st.markdown(
        f"<p style='text-align:right; color:#484f58; font-size:0.78rem; padding-top:12px;'>"
        f"📅 {datetime.now().strftime('%d %B %Y  •  %H:%M')}<br>"
        f"Analyses this session: <b style='color:#c9d1d9;'>{st.session_state.run_count}</b></p>",
        unsafe_allow_html=True,
    )

# ── KPI Strip ──
k1, k2, k3, k4 = st.columns(4)
with k1:
    bmi_delta = f"{bmi - 24.9:+.1f}" if bmi > 24.9 else "Healthy"
    st.metric("Body Mass Index", f"{bmi}", delta=bmi_delta, delta_color="inverse")
with k2:
    st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}",
              delta="Hypertensive" if high_bp else "Optimal", delta_color="inverse")
with k3:
    st.metric("Mean Arterial Pressure", f"{mean_ap} mmHg")
with k4:
    st.metric("Patient Age", f"{age} yr")

st.markdown("")  # small spacer

# ── Two-column body ──
col_left, col_right = st.columns([5, 2])

# ─────────────── LEFT COLUMN: Diagnosis Engine ───────────────
with col_left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # Input validation
    if ap_lo >= ap_hi:
        st.error("⛔  **Input Error:** Diastolic pressure ≥ Systolic. Correct the vitals in the sidebar.")
    else:
        if st.button("🔬  EXECUTE DIAGNOSTIC ANALYSIS"):
            if pipeline is not None and feature_names is not None:
                row = {
                    "age": age, "gender": gender, "height": height, "weight": weight,
                    "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                    "smoke": int(smoke), "alco": int(alco), "active": int(active),
                    "bmi": bmi, "pulse_pressure": pulse_pressure,
                    "high_bp_flag": high_bp, "map": mean_ap,
                }
                input_df = pd.DataFrame([row])[feature_names]
                prob = pipeline.predict_proba(input_df)[0][1] * 100
                st.session_state.prediction_results = {
                    "prob": prob,
                    "risk": "HIGH" if prob > 50 else "LOW",
                    "ts": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.run_count += 1
                st.session_state.history.append(round(prob, 1))
                if prob < 30:
                    st.balloons()
            else:
                st.error("Model assets not found. Ensure `cardio_pipeline.pkl` and `feature_names.pkl` are present.")

    # ── Results Panel ──
    if st.session_state.prediction_results:
        res = st.session_state.prediction_results
        is_high = res["prob"] > 50
        accent = "#ff6b6b" if is_high else "#00ce78"

        st.markdown("---")
        rl, rr = st.columns([1, 1])

        with rl:
            badge_cls = "badge-high" if is_high else "badge-low"
            score_cls = "score-high" if is_high else "score-low"
            st.markdown(f'<span class="{badge_cls}">{res["risk"]} RISK</span>', unsafe_allow_html=True)
            st.markdown(f'<p class="{score_cls}">{res["prob"]:.1f}%</p>', unsafe_allow_html=True)
            st.caption(f'Completed {res["ts"]}  •  Run #{st.session_state.run_count}')

            # Interpretation
            if is_high:
                st.markdown(
                    "**Interpretation:** Elevated cardiovascular risk detected. "
                    "Multiple clinical indicators exceed optimal thresholds. "
                    "Recommend comprehensive cardiovascular workup and specialist referral."
                )
            else:
                st.markdown(
                    "**Interpretation:** Risk profile within acceptable range. "
                    "Continue current preventive measures and schedule routine follow-up screening."
                )

        with rr:
            st.plotly_chart(gauge_chart(res["prob"], accent), use_container_width=True)

    else:
        st.info(
            "👋  **Welcome** — Configure patient parameters in the sidebar and click the button above "
            "to generate a cardiovascular risk assessment."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tabbed Details ──
    tab_proto, tab_profile, tab_hemo = st.tabs([
        "📋  Clinical Protocols", "🕸️  Risk Profile", "📊  Hemodynamics"
    ])

    with tab_proto:
        if st.session_state.prediction_results:
            st.markdown("#### Recommended Clinical Protocols")
            p1, p2 = st.columns(2)

            with p1:
                if bmi > 25:
                    st.markdown(
                        '<div class="protocol urgent">⚖️ <b>Weight Management</b> — '
                        'Prescribe caloric deficit plan; target BMI &lt; 25. '
                        'Refer to dietitian.</div>',
                        unsafe_allow_html=True,
                    )
                if high_bp:
                    st.markdown(
                        '<div class="protocol urgent">🩸 <b>Hypertension Control</b> — '
                        'Initiate 24-hr ambulatory BP monitoring. '
                        'Evaluate ACE inhibitor / ARB therapy.</div>',
                        unsafe_allow_html=True,
                    )
                if chol > 1:
                    st.markdown(
                        '<div class="protocol">🧪 <b>Lipid Intervention</b> — '
                        'Order fasting lipid panel. '
                        'Consider statin therapy if LDL &gt; 130 mg/dL.</div>',
                        unsafe_allow_html=True,
                    )

            with p2:
                if smoke:
                    st.markdown(
                        '<div class="protocol urgent">🚬 <b>Smoking Cessation</b> — '
                        'Initiate nicotine replacement therapy. '
                        'Schedule behavioral counseling.</div>',
                        unsafe_allow_html=True,
                    )
                if not active:
                    st.markdown(
                        '<div class="protocol">🏃 <b>Exercise Prescription</b> — '
                        'Prescribe ≥ 150 min/week moderate-intensity aerobic activity. '
                        'Gradual ramp-up over 4 weeks.</div>',
                        unsafe_allow_html=True,
                    )
                if gluc > 1:
                    st.markdown(
                        '<div class="protocol">🔬 <b>Glycemic Monitoring</b> — '
                        'Order HbA1c panel. Consider metformin if prediabetic.</div>',
                        unsafe_allow_html=True,
                    )

            # If no flags at all
            if bmi <= 25 and not high_bp and chol == 1 and gluc == 1 and not smoke and active:
                st.success("All clinical indicators within optimal ranges. Maintain current lifestyle.")
        else:
            st.caption("Execute an analysis to generate personalised protocols.")

    with tab_profile:
        st.markdown("#### Risk Factor Fingerprint")
        st.plotly_chart(
            radar_chart(bmi_n, bp_n, chol_n, gluc_n, smoke_n, inactivity_n),
            use_container_width=True,
        )
        st.caption(
            "Each axis represents a normalised risk factor (0 = optimal, 1 = highest risk). "
            "A smaller polygon indicates a healthier overall profile."
        )

    with tab_hemo:
        st.markdown("#### Hemodynamic Summary")

        def status_icon(ok: bool):
            return "✅" if ok else "⚠️"

        hemo_df = pd.DataFrame({
            "Indicator": ["Body Mass Index", "Systolic BP", "Diastolic BP",
                          "Mean Arterial Pressure", "Pulse Pressure"],
            "Measured": [bmi, ap_hi, ap_lo, mean_ap, pulse_pressure],
            "Reference": ["18.5 – 24.9", "< 130 mmHg", "< 85 mmHg", "70 – 100 mmHg", "30 – 50 mmHg"],
            "Status": [
                status_icon(18.5 <= bmi <= 24.9),
                status_icon(ap_hi < 130),
                status_icon(ap_lo < 85),
                status_icon(70 <= mean_ap <= 100),
                status_icon(30 <= pulse_pressure <= 50),
            ],
        })
        st.dataframe(hemo_df, hide_index=True, use_container_width=True)


# ─────────────── RIGHT COLUMN: Intel & Notes ─────────────────
with col_right:
    # System Info Card
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 🧬  System Intelligence")
    st.markdown(
        "<span style='font-size:0.85rem; color:#8b949e;'>"
        "Gradient Boosting classifier trained on <b style='color:#c9d1d9;'>70 000+</b> patient "
        "records from the Kaggle Cardiovascular Disease dataset.</span>",
        unsafe_allow_html=True,
    )
    st.progress(0.735, text="Model Accuracy: 73.5 %")
    st.markdown("")

    # Quick stats
    st.markdown("**Pipeline Info**")
    info_data = {
        "Algorithm": "GradientBoostingClassifier",
        "Library": "scikit-learn 1.5.2",
        "Features": f"{len(feature_names) if feature_names else '—'}",
        "Preprocessing": "ColumnTransformer",
    }
    for label, val in info_data.items():
        st.markdown(
            f"<span style='font-size:0.8rem; color:#6e7681;'>{label}</span><br>"
            f"<span style='font-size:0.88rem; color:#c9d1d9;'>{val}</span>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis History Sparkline
    if st.session_state.history:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 📈  Session History")
        hist = st.session_state.history
        spark = go.Figure(go.Scatter(
            y=hist, mode="lines+markers",
            line=dict(color="#6c5ce7", width=2),
            marker=dict(size=6, color="#a29bfe"),
            fill="tozeroy", fillcolor="rgba(108,92,231,0.08)",
        ))
        spark.update_layout(
            height=160, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 100]),
        )
        st.plotly_chart(spark, use_container_width=True)
        st.caption(f"Last: **{hist[-1]}%** • Min: {min(hist)}% • Max: {max(hist)}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Notes Card
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📝  Clinical Notes")
    st.text_area(
        "Observations",
        placeholder="Document findings, differential considerations, follow-up plan…",
        height=120,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### ⚕️  Disclaimer")
    st.caption(
        "This tool is intended for clinical screening assistance only. "
        "It does **not** replace professional medical judgment, formal diagnosis, "
        "or treatment planning. Always consult a qualified healthcare provider."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━  FOOTER  ━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<hr style='border-color: rgba(255,255,255,0.05);'>"
    "<p style='text-align:center; color:#484f58; font-size:0.75rem;'>"
    "© 2026 CardioScreen Pro  •  Clinical Diagnostic Suite v3.0  •  "
    "Built with Streamlit & scikit-learn  •  Secure Environment</p>",
    unsafe_allow_html=True,
)
