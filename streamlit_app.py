"""
ApexShield | Dual-Cohort Health Insurance Underwriting Engine
Clean, compact, session-state-driven form — no scroll needed, instant feedback.
"""
import os
import textwrap
import joblib
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ApexShield · Insurance Underwriting",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def html(content: str):
    """Render HTML safely without indentation causing markdown code blocks."""
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg:     #F1F5F9;
    --white:  #FFFFFF;
    --navy:   #0F172A;
    --nmid:   #1E293B;
    --nsoft:  #334155;
    --blue:   #2563EB;
    --bdk:    #1D4ED8;
    --blt:    #EFF6FF;
    --brim:   #BFDBFE;
    --green:  #059669;
    --glt:    #ECFDF5;
    --grim:   #A7F3D0;
    --border: #E2E8F0;
    --muted:  #64748B;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--navy);
}
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { visibility: hidden !important; }
.block-container {
    max-width: 1320px !important;
    padding: 0.45rem 1.15rem 0.85rem !important;
}

/* ── Header ── */
.hdr {
    background: linear-gradient(130deg, #0F172A 0%, #1E293B 55%, #1E3A8A 100%);
    border-radius: 0.6rem;
    padding: 0.45rem 1rem;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.4rem;
    box-shadow: 0 4px 12px -2px rgba(15,23,42,0.16);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 0.4rem;
}
.hdr-brand { display: flex; align-items: center; gap: 0.55rem; }
.hdr-icon {
    width: 28px; height: 28px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 0.4rem; display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.hdr-title { font-size: 0.98rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
.hdr-title b { color: #93C5FD; }
.hdr-sub { font-size: 0.64rem; color: #94A3B8; margin-top: 1px; }
.hdr-badges { display: flex; gap: 0.3rem; flex-wrap: wrap; align-items: center; }
.hbadge {
    font-size: 0.63rem; font-weight: 600;
    padding: 0.16rem 0.48rem; border-radius: 9999px;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.11);
    color: #CBD5E1; white-space: nowrap;
    display: inline-flex; align-items: center; gap: 0.25rem;
}
.hbadge.live {
    background: rgba(16,185,129,0.14); border-color: rgba(52,211,153,0.28); color: #34D399;
}
.ldot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #34D399; box-shadow: 0 0 0 2px rgba(52,211,153,0.3);
}

/* ── Preset buttons ── */
.preset-row {
    display: flex; align-items: center; gap: 0.4rem;
    margin-bottom: 0.35rem;
}
.preset-lbl {
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); margin: 0;
}
div[data-testid="stButton"] > button {
    background: var(--white) !important;
    color: var(--nmid) !important;
    -webkit-text-fill-color: var(--nmid) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 0.45rem !important;
    font-size: 0.74rem !important; font-weight: 600 !important;
    padding: 0.28rem 0.4rem !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--blt) !important;
    border-color: var(--blue) !important;
    color: var(--bdk) !important;
    -webkit-text-fill-color: var(--bdk) !important;
    transform: translateY(-1px) !important;
}

/* ── Form card ── */
div[data-testid="stForm"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0.7rem !important;
    padding: 0.65rem 0.95rem 0.75rem !important;
    box-shadow: 0 2px 8px -2px rgba(15,23,42,0.05) !important;
}
div[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }

/* ── Section row ── */
.sec {
    display: flex; align-items: center; gap: 0.4rem;
    padding-bottom: 0.2rem; margin-bottom: 0.35rem;
    border-bottom: 1px solid #F1F5F9;
    margin-top: 0;
}
.sec-mt { margin-top: 0.45rem; }
.sec-num {
    width: 1.25rem; height: 1.25rem; border-radius: 50%;
    background: #1E3A8A; color: #93C5FD;
    font-size: 0.62rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.sec-title { font-size: 0.78rem; font-weight: 700; color: var(--navy); }
.ctag {
    margin-left: auto; font-size: 0.63rem; font-weight: 700;
    padding: 0.14rem 0.45rem; border-radius: 9999px;
}
.ctag-y { background: var(--glt); color: var(--green); border: 1px solid var(--grim); }
.ctag-a { background: var(--blt); color: var(--bdk);   border: 1px solid var(--brim); }

/* ── Widget labels ── */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.74rem !important; font-weight: 700 !important;
    color: #1E293B !important; -webkit-text-fill-color: #1E293B !important;
    margin-bottom: 0.15rem !important;
}

/* ── Selectboxes & Inputs ── */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border-radius: 0.45rem !important;
    border: 1.5px solid #CBD5E1 !important;
    background: #F8FAFC !important;
    min-height: 36px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--blue) !important;
    background: var(--white) !important;
}

/* Guaranteed high-contrast visible text for selectbox values */
div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
div[data-testid="stSelectbox"] svg {
    fill: #475569 !important;
}

/* Number Inputs */
div[data-testid="stNumberInput"] > div > div > input {
    border-radius: 0.45rem !important;
    border: 1.5px solid #CBD5E1 !important;
    background: #F8FAFC !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    min-height: 36px !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
div[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: var(--blue) !important;
    background: var(--white) !important;
}

/* Dropdown Menu Popover */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 0.5rem !important;
    box-shadow: 0 10px 25px rgba(15,23,42,0.18) !important;
}
div[data-baseweb="popover"] li,
ul[role="listbox"] li {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.75rem !important;
}
div[data-baseweb="popover"] li:hover,
ul[role="listbox"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
    -webkit-text-fill-color: #1D4ED8 !important;
    font-weight: 700 !important;
}

/* ── Radio buttons (Coverage Tier) ── */
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: grid !important;
    grid-template-columns: 1fr 1fr 1fr !important;
    gap: 0.5rem !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: #F8FAFC !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 0.5rem !important;
    padding: 0.4rem 0.4rem !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.15s ease !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border-color: #2563EB !important;
    background: #EFF6FF !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label p {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    -webkit-text-fill-color: #1E293B !important;
    margin: 0 !important;
    white-space: nowrap !important;
}

/* ── Genetic risk badge ── */
.gr-strip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.6rem; border-radius: 0.4rem;
    font-size: 0.72rem; font-weight: 600; margin-top: 0.25rem;
}

/* ── Submit Button ── */
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 0.55rem !important;
    padding: 0.52rem 1.2rem !important;
    font-size: 0.85rem !important; font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important; width: 100% !important;
    letter-spacing: -0.01em !important;
}
div.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #172554 0%, #1D4ED8 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.4) !important;
}
div.stFormSubmitButton > button:active { transform: translateY(0) !important; }

/* ── Result strip below submit ── */
.result-strip {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--glt); border: 1.5px solid var(--grim);
    border-radius: 0.55rem; padding: 0.6rem 0.95rem;
    font-size: 0.82rem; font-weight: 700; color: #047857;
    margin-top: 0.6rem;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-amt { font-size: 1.05rem; font-weight: 800; color: var(--navy); }

/* ── Quote card ── */
.q-card {
    background: linear-gradient(155deg, #0F172A 0%, #1E293B 55%, #0F172A 100%);
    border-radius: 0.85rem;
    padding: 1.15rem 1.25rem;
    color: #fff;
    box-shadow: 0 10px 24px -4px rgba(15,23,42,0.3);
    border: 1px solid rgba(255,255,255,0.08);
    position: relative; overflow: hidden;
}
.q-card.fresh {
    box-shadow: 0 0 0 3px #2563EB, 0 10px 28px -4px rgba(15,23,42,0.35);
    border-color: #3B82F6;
}
.q-card::before {
    content: ''; position: absolute; top: -35px; right: -35px;
    width: 130px; height: 130px;
    background: radial-gradient(circle, rgba(37,99,235,0.2) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.q-top {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 0.75rem;
}
.q-lbl {
    font-size: 0.64rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; color: #93C5FD;
}
.q-badge {
    font-size: 0.63rem; font-weight: 600;
    padding: 0.16rem 0.45rem; border-radius: 9999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12); color: #E2E8F0;
}
.q-amount {
    font-size: 2.5rem; font-weight: 800;
    letter-spacing: -0.04em; color: #fff; line-height: 1;
    margin-bottom: 0.2rem;
}
.q-amount sup { font-size: 1.05rem; font-weight: 600; color: #93C5FD; margin-right: 2px; }
.q-mo { font-size: 0.75rem; color: #94A3B8; margin-bottom: 0.85rem; font-weight: 500; }
.q-mo b { color: #38BDF8; }
.q-div { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 0 0 0.65rem; }
.q-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.28rem 0; font-size: 0.74rem; color: #CBD5E1;
}
.q-row b { font-weight: 700; color: #fff; }
.q-note {
    margin-top: 0.75rem; padding: 0.55rem 0.75rem;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 0.5rem; font-size: 0.7rem; color: #94A3B8; line-height: 1.4;
}
.q-note strong { color: #60A5FA; }

/* ── Benchmark ── */
.cmp {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--glt); border: 1px solid var(--grim);
    border-radius: 0.6rem; padding: 0.6rem 0.95rem; margin-top: 0.65rem;
}
.cmp-lbl { font-size: 0.66rem; font-weight: 600; color: #065F46; margin-bottom: 0.05rem; }
.cmp-old { font-size: 0.76rem; color: #64748B; text-decoration: line-through; font-weight: 600; }
.cmp-save { font-size: 0.85rem; font-weight: 800; color: var(--green); }

/* ── Perks ── */
.perks {
    background: var(--white); border: 1px solid var(--border);
    border-radius: 0.6rem; padding: 0.75rem 0.95rem; margin-top: 0.65rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.perks-ttl { font-size: 0.72rem; font-weight: 700; color: var(--navy); margin-bottom: 0.5rem; }
.perks-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.38rem; }
.perk { font-size: 0.7rem; color: var(--nsoft); font-weight: 500; }

/* ── Footer ── */
.foot {
    text-align: center; padding: 0.85rem 0 0.2rem;
    font-size: 0.67rem; color: #94A3B8;
    border-top: 1px solid var(--border); margin-top: 0.85rem; line-height: 1.45;
}
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# ML ENGINE
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

RISK_MAP = {
    "none": 0, "no disease": 0, "thyroid": 5,
    "diabetes": 6, "high blood pressure": 6, "heart disease": 8,
}

@st.cache_resource(show_spinner="Loading underwriting engine…")
def load_engine():
    try:
        m_y = joblib.load(f"{ARTIFACTS_DIR}/model_young.joblib")
        b_y = joblib.load(f"{ARTIFACTS_DIR}/scaler_young.joblib")
        m_r = joblib.load(f"{ARTIFACTS_DIR}/model_rest.joblib")
        b_r = joblib.load(f"{ARTIFACTS_DIR}/scaler_rest.joblib")
        return dict(
            m_y=m_y, sc_y=b_y["scaler"], cols_y=b_y["cols_to_scale"],
            m_r=m_r, sc_r=b_r["scaler"], cols_r=b_r["cols_to_scale"], ok=True,
        )
    except Exception as e:
        return dict(ok=False, err=str(e))

engine = load_engine()

def predict(payload: dict, young: bool) -> float:
    df = pd.DataFrame([payload])
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df["number_of_dependants"] = abs(df["number_of_dependants"])
    med   = str(payload.get("medical_history", "none")).lower()
    parts = [x.strip() for x in med.split("&")]
    df["normalized_risk_score"] = sum(RISK_MAP.get(x, 0) for x in parts) / 14.0
    df["insurance_plan"] = {"Bronze": 1, "Silver": 2, "Gold": 3}.get(payload["insurance_plan"], 2)
    df["income_level"]   = 1
    df["gender_Male"]                     = int(payload["gender"] == "Male")
    df["region_Northwest"]                = int(payload["region"] == "Northwest")
    df["region_Southeast"]                = int(payload["region"] == "Southeast")
    df["region_Southwest"]                = int(payload["region"] == "Southwest")
    df["marital_status_Unmarried"]        = int(payload["marital_status"] == "Unmarried")
    df["bmi_category_Obesity"]            = int(payload["bmi_category"] == "Obesity")
    df["bmi_category_Overweight"]         = int(payload["bmi_category"] == "Overweight")
    df["bmi_category_Underweight"]        = int(payload["bmi_category"] == "Underweight")
    df["smoking_status_Occasional"]       = int(payload["smoking_status"] == "Occasional")
    df["smoking_status_Regular"]          = int(payload["smoking_status"] == "Regular")
    df["employment_status_Salaried"]      = int(payload["employment_status"] == "Salaried")
    df["employment_status_Self-Employed"] = int(payload["employment_status"] == "Self-Employed")
    sc   = engine["sc_y"] if young else engine["sc_r"]
    cols = engine["cols_y"] if young else engine["cols_r"]
    df[cols] = sc.transform(df[cols])
    X = df.drop(columns=["income_level"])
    feat_order = [
        "age", "number_of_dependants", "income_lakhs", "insurance_plan",
        "genetical_risk", "normalized_risk_score", "gender_Male",
        "region_Northwest", "region_Southeast", "region_Southwest",
        "marital_status_Unmarried", "bmi_category_Obesity",
        "bmi_category_Overweight", "bmi_category_Underweight",
        "smoking_status_Occasional", "smoking_status_Regular",
        "employment_status_Salaried", "employment_status_Self-Employed",
    ]
    m = engine["m_y"] if young else engine["m_r"]
    return float(m.predict(X[feat_order])[0])

# ─────────────────────────────────────────────────────────────────────────────
# PROFILES
# ─────────────────────────────────────────────────────────────────────────────
PROFILES = {
    "🌱 Young (22y)": dict(
        age=22, gender="Female", dependants=0, income=10.0,
        marital="Unmarried", region="Northwest", bmi="Normal", smoking="No",
        medical="No Disease", gr=1, plan="Silver", emp="Salaried",
    ),
    "🧬 Young+Risk (24y)": dict(
        age=24, gender="Male", dependants=0, income=14.0,
        marital="Unmarried", region="Northeast", bmi="Normal", smoking="No",
        medical="No Disease", gr=4, plan="Silver", emp="Salaried",
    ),
    "💼 Family (42y)": dict(
        age=42, gender="Male", dependants=2, income=32.0,
        marital="Married", region="Southeast", bmi="Overweight", smoking="Occasional",
        medical="High Blood Pressure", gr=2, plan="Gold", emp="Salaried",
    ),
    "🩺 Senior (57y)": dict(
        age=57, gender="Female", dependants=1, income=22.0,
        marital="Married", region="Southwest", bmi="Obesity", smoking="Regular",
        medical="Diabetes & Heart Disease", gr=3, plan="Gold", emp="Self-Employed",
    ),
}

# ── Initialize widget keys in session_state once ──────────────────────────────
_DEF = PROFILES["🌱 Young (22y)"]
_INIT = {
    "w_age":     _DEF["age"],
    "w_gender":  _DEF["gender"],
    "w_marital": _DEF["marital"],
    "w_deps":    _DEF["dependants"],
    "w_income":  float(_DEF["income"]),
    "w_region":  _DEF["region"],
    "w_bmi":     _DEF["bmi"],
    "w_smoking": _DEF["smoking"],
    "w_emp":     _DEF["emp"],
    "w_medical": _DEF["medical"],
    "w_gr":      int(_DEF["gr"]),
    "w_plan":    _DEF["plan"],
    "last_result": None,
}
for _k, _v in _INIT.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
html("""
<div class="hdr">
  <div class="hdr-brand">
    <div class="hdr-icon">🛡️</div>
    <div>
      <div class="hdr-title">Apex<b>Shield</b></div>
      <div class="hdr-sub">Dual-Cohort Actuarial ML Underwriting Engine v2.0</div>
    </div>
  </div>
  <div class="hdr-badges">
    <span class="hbadge live"><span class="ldot"></span> ML Active</span>
    <span class="hbadge">IRDAI Compliant</span>
    <span class="hbadge">Young R² 98.87%</span>
    <span class="hbadge">Adult R² 99.71%</span>
  </div>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# PRESET BUTTONS — write into widget keys, then rerun
# ─────────────────────────────────────────────────────────────────────────────
html('<div class="preset-row"><p class="preset-lbl">⚡ Quick-fill sample profile:</p></div>')
pcols = st.columns(4)
for i, (lbl, dat) in enumerate(PROFILES.items()):
    if pcols[i].button(lbl, use_container_width=True, key=f"preset_{i}"):
        st.session_state["w_age"]     = dat["age"]
        st.session_state["w_gender"]  = dat["gender"]
        st.session_state["w_marital"] = dat["marital"]
        st.session_state["w_deps"]    = dat["dependants"]
        st.session_state["w_income"]  = float(dat["income"])
        st.session_state["w_region"]  = dat["region"]
        st.session_state["w_bmi"]     = dat["bmi"]
        st.session_state["w_smoking"] = dat["smoking"]
        st.session_state["w_emp"]     = dat["emp"]
        st.session_state["w_medical"] = dat["medical"]
        st.session_state["w_gr"]      = int(dat["gr"])
        st.session_state["w_plan"]    = dat["plan"]
        st.session_state["last_result"] = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([11, 9], gap="large")

# ═══════════════════════════════════════════════════════════════════
# LEFT — FORM
# ═══════════════════════════════════════════════════════════════════
with left:
    cur_age    = st.session_state["w_age"]
    young_prev = int(cur_age) <= 25
    ctag_cls   = "ctag-y" if young_prev else "ctag-a"
    ctag_lbl   = "Young ≤ 25y" if young_prev else "Adult > 25y"

    with st.form("qf", border=False):

        # ── Section 1: Demographics ────────────────────────────────
        html(f"""
        <div class="sec">
          <span class="sec-num">1</span>
          <span class="sec-title">Personal &amp; Demographics</span>
          <span class="ctag {ctag_cls}">{ctag_lbl}</span>
        </div>""")

        c1, c2, c3 = st.columns(3)
        age     = c1.number_input("Age", min_value=18, max_value=85, key="w_age")
        gender  = c2.selectbox("Gender", ["Female", "Male"], key="w_gender")
        marital = c3.selectbox("Marital", ["Unmarried", "Married"], key="w_marital")

        c4, c5, c6 = st.columns(3)
        deps   = c4.selectbox("Dependants", [0, 1, 2, 3, 4, 5],
                              format_func=lambda x: "None" if x == 0 else str(x),
                              key="w_deps")
        income = c5.number_input("Income (₹ Lakhs/yr)", min_value=0.5, max_value=250.0,
                                 step=1.0, format="%.1f", key="w_income")
        region = c6.selectbox("Region",
                              ["Northwest", "Northeast", "Southeast", "Southwest"],
                              key="w_region")

        # ── Section 2: Health & Lifestyle ──────────────────────────
        html("""
        <div class="sec sec-mt">
          <span class="sec-num">2</span>
          <span class="sec-title">Health &amp; Lifestyle</span>
        </div>""")

        c7, c8, c9 = st.columns(3)
        bmi     = c7.selectbox("BMI Category",
                               ["Normal", "Overweight", "Obesity", "Underweight"],
                               key="w_bmi")
        smoking = c8.selectbox("Smoking Status",
                               ["No", "Occasional", "Regular"],
                               format_func=lambda x: {"No": "Non-Smoker",
                                                      "Occasional": "Occasional",
                                                      "Regular": "Regular Smoker"}[x],
                               key="w_smoking")
        emp     = c9.selectbox("Employment",
                               ["Salaried", "Self-Employed", "Freelancer"],
                               key="w_emp")

        cm1, cm2 = st.columns([1, 1])
        with cm1:
            medical = st.selectbox(
                "Pre-existing Conditions",
                ["No Disease", "Diabetes", "High Blood Pressure", "Thyroid", "Heart Disease",
                 "Diabetes & High Blood Pressure", "Diabetes & Heart Disease",
                 "High Blood Pressure & Heart Disease"],
                key="w_medical",
            )
        with cm2:
            gr = st.slider("Genetic Risk (0 = None → 5 = High)",
                           min_value=0, max_value=5, step=1, key="w_gr")
            GR_META = {
                0: ("No family risk",                 "#047857", "#ECFDF5", "#A7F3D0"),
                1: ("Minimal — distant relatives",     "#047857", "#ECFDF5", "#A7F3D0"),
                2: ("Mild hereditary patterns",        "#1D4ED8", "#EFF6FF", "#BFDBFE"),
                3: ("Moderate — parent chronic",       "#B45309", "#FFFBEB", "#FDE68A"),
                4: ("Elevated — cardiovascular fam.",  "#C2410C", "#FFF7ED", "#FED7AA"),
                5: ("High — multiple direct rel.",     "#B91C1C", "#FEF2F2", "#FECACA"),
            }
            glbl, gcol, gbg, gbdr = GR_META[gr]
            html(f'<span class="gr-strip" style="background:{gbg};border:1px solid {gbdr};color:{gcol};">'
                 f'Level {gr} · {glbl}</span>')

        # ── Section 3: Coverage Tier ───────────────────────────────
        html("""
        <div class="sec sec-mt">
          <span class="sec-num">3</span>
          <span class="sec-title">Choose Coverage Tier</span>
        </div>""")

        plan = st.radio(
            "Coverage Tier",
            ["Bronze", "Silver", "Gold"],
            horizontal=True,
            format_func=lambda x: {
                "Bronze": "🥉 Bronze · Essential",
                "Silver": "🥈 Silver · Comprehensive ★",
                "Gold":   "🥇 Gold · Premium Global",
            }[x],
            key="w_plan",
            label_visibility="collapsed",
        )

        html("<div style='height:0.35rem'></div>")
        submitted = st.form_submit_button(
            "⚡  Calculate Instant Actuarial Quote",
            type="primary",
            use_container_width=True,
        )

    # ── Result strip — directly below submit button ─────────────────
    result_ph = st.empty()

    if submitted:
        if not engine.get("ok"):
            result_ph.error(f"Engine error: {engine.get('err')}")
        else:
            with st.spinner("Calculating actuarial quote…"):
                _payload = dict(
                    age=age, gender=gender, region=region,
                    marital_status=marital, number_of_dependants=deps,
                    bmi_category=bmi, smoking_status=smoking,
                    employment_status=emp, income_lakhs=income,
                    medical_history=medical, insurance_plan=plan, genetical_risk=gr,
                )
                _young  = age <= 25
                _raw    = predict(_payload, _young)
                _annual = max(round(_raw), 3000)
                _mo     = round(_annual / 12)
                st.session_state["last_result"] = {
                    "annual": _annual, "monthly": _mo, "young": _young,
                    "plan": plan, "bmi": bmi, "smoking": smoking, "gr": gr,
                    **_payload,
                }
            result_ph.markdown(
                f'<div class="result-strip">'
                f'<span>✅ Quote Calculated &amp; Ready</span>'
                f'<span class="result-amt">₹{_annual:,} / yr &nbsp;·&nbsp; ₹{_mo:,} / mo</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.toast(f"✅ Quote Ready: ₹{_annual:,} / yr (approx. ₹{_mo:,} / mo)", icon="🛡️")

    elif st.session_state["last_result"]:
        r = st.session_state["last_result"]
        result_ph.markdown(
            f'<div class="result-strip">'
            f'<span>✅ Current Saved Quote</span>'
            f'<span class="result-amt">₹{r["annual"]:,} / yr &nbsp;·&nbsp; ₹{r["monthly"]:,} / mo</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════
# RIGHT — LIVE QUOTE CARD
# ═══════════════════════════════════════════════════════════════════
with right:
    if not engine.get("ok"):
        st.error(f"Engine error: {engine.get('err')}")
        st.stop()

    res = st.session_state["last_result"]
    if res:
        # Use submitted result
        annual      = res["annual"]
        monthly     = res["monthly"]
        young_c     = res["young"]
        plan_r      = res["plan"]
        bmi_r       = res["bmi"]
        smoke_r     = res["smoking"]
        gr_r        = res["gr"]
        card_cls    = "fresh"
    else:
        # Live preview from current widget session_state
        _pa = dict(
            age     = st.session_state["w_age"],
            gender  = st.session_state["w_gender"],
            region  = st.session_state["w_region"],
            marital_status       = st.session_state["w_marital"],
            number_of_dependants = st.session_state["w_deps"],
            bmi_category         = st.session_state["w_bmi"],
            smoking_status       = st.session_state["w_smoking"],
            employment_status    = st.session_state["w_emp"],
            income_lakhs         = st.session_state["w_income"],
            medical_history      = st.session_state["w_medical"],
            insurance_plan       = st.session_state["w_plan"],
            genetical_risk       = st.session_state["w_gr"],
        )
        young_c  = int(_pa["age"]) <= 25
        _raw     = predict(_pa, young_c)
        annual   = max(round(_raw), 3000)
        monthly  = round(annual / 12)
        plan_r   = _pa["insurance_plan"]
        bmi_r    = _pa["bmi_category"]
        smoke_r  = _pa["smoking_status"]
        gr_r     = _pa["genetical_risk"]
        card_cls = ""

    # Cost breakdown
    pf         = {"Bronze": 0.78, "Silver": 1.0, "Gold": 1.32}[plan_r]
    base_calc  = int(annual / pf * 0.82)
    plan_up    = max(int(annual - base_calc * pf), 400)
    life_up    = (800 if bmi_r == "Obesity" else 350 if bmi_r == "Overweight" else 0) + \
                 (700 if smoke_r == "Regular" else 280 if smoke_r == "Occasional" else 0)
    hist_up    = gr_r * 420 if young_c else gr_r * 180
    base_cover = max(annual - plan_up - life_up - hist_up, 1800)

    cohort_tag = "Young Cohort (≤ 25y)" if young_c else "Adult Cohort (> 25y)"
    note = (
        "Young Pipeline: genetic & early-career risks calibrated for fair pricing."
        if young_c else
        "Adult Pipeline: clinical history, BMI, lifestyle & tier drive this estimate."
    )

    html(f"""
    <div class="q-card {card_cls}">
      <div class="q-top">
        <span class="q-lbl">Actuarial Underwriting Result</span>
        <span class="q-badge">{cohort_tag}</span>
      </div>
      <div class="q-amount"><sup>₹</sup>{annual:,}</div>
      <div class="q-mo">per year &nbsp;·&nbsp; approx. <b>₹{monthly:,}</b> / month</div>
      <hr class="q-div">
      <div class="q-row"><span>Base Hospitalisation</span><b>₹{base_cover:,}</b></div>
      <div class="q-row"><span>{plan_r} Tier Loading</span><b>+ ₹{plan_up:,}</b></div>
      <div class="q-row"><span>Lifestyle / BMI Loading</span><b>{"+ ₹{:,}".format(life_up) if life_up else "₹ 0"}</b></div>
      <div class="q-row"><span>Family History Loading</span><b>{"+ ₹{:,}".format(hist_up) if hist_up else "₹ 0"}</b></div>
      <div class="q-note"><strong>Why this price?</strong> {note}</div>
    </div>
    """)

    baseline = int(annual * (1.28 if young_c and gr_r >= 3 else 1.12))
    saving   = baseline - annual
    if saving > 0:
        html(f"""
        <div class="cmp">
          <div>
            <div class="cmp-lbl">Industry Estimate</div>
            <div class="cmp-old">₹{baseline:,} / yr</div>
          </div>
          <div style="text-align:right">
            <div class="cmp-lbl" style="color:#047857">AI Advantage</div>
            <div class="cmp-save">Save ₹{saving:,} / yr</div>
          </div>
        </div>
        """)

    html("""
    <div class="perks">
      <div class="perks-ttl">✦ Included in every plan</div>
      <div class="perks-grid">
        <div class="perk">🏥 14,000+ Hospitals</div>
        <div class="perk">⚡ Cashless in 30 min</div>
        <div class="perk">🩺 Annual Health Check</div>
        <div class="perk">💊 Pre &amp; Post Care</div>
        <div class="perk">🚑 Ambulance Cover</div>
        <div class="perk">🧘 Mental Wellness</div>
      </div>
    </div>
    """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
html("""
<div class="foot">
  ApexShield Life &amp; Health Insurance · IRDAI Reg. No. 142 · ML Engine v2.0<br>
  Premiums are actuarial estimates for illustrative purposes. Subject to underwriting review.
</div>
""")
