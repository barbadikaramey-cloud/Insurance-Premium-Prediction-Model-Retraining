"""
ApexShield | Premium Health Insurance Quotation Engine
Final production build — elegant, minimal, professional.
"""
import os
import joblib
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# 1.  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ApexShield · Instant Health Insurance Quote",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# 2.  GLOBAL CSS  (rich but restrained palette)
#     Deep navy + warm gold + soft champagne
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&family=DM+Serif+Display&display=swap');

/* ── Reset & Root ── */
:root {
    --navy:       #0B1829;
    --navy-mid:   #122039;
    --navy-soft:  #1A2E4A;
    --gold:       #C9973B;
    --gold-light: #E8BE6C;
    --champagne:  #F9F4EC;
    --slate:      #64748B;
    --slate-lt:   #94A3B8;
    --border:     #E2E8F0;
    --white:      #FFFFFF;
    --success:    #059669;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
    color: var(--navy);
}
.stApp { background: var(--champagne); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── Navbar ── */
.apex-nav {
    background: var(--navy);
    margin: -1rem -4rem 2.5rem -4rem;
    padding: 1rem 4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.apex-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none;
}
.apex-logo-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: var(--white);
    letter-spacing: -0.3px;
}
.apex-logo-text span { color: var(--gold-light); }
.apex-nav-links {
    display: flex;
    gap: 2rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: rgba(255,255,255,0.55);
    letter-spacing: 0.04em;
}
.nav-active { color: var(--gold-light) !important; }

/* ── Hero ── */
.apex-hero {
    text-align: center;
    padding: 1.5rem 0 2.5rem;
    max-width: 680px;
    margin: 0 auto;
}
.apex-hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    line-height: 1.15;
    color: var(--navy);
    letter-spacing: -1px;
    margin-bottom: 0.85rem;
}
.apex-hero h1 em {
    font-style: normal;
    color: var(--gold);
}
.apex-hero p {
    font-size: 1rem;
    color: var(--slate);
    line-height: 1.65;
    max-width: 520px;
    margin: 0 auto 1.5rem;
}
.trust-badges {
    display: flex;
    justify-content: center;
    gap: 0.65rem;
    flex-wrap: wrap;
}
.trust-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--slate);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.trust-badge .dot { color: var(--success); font-size: 0.6rem; }

/* ── Preset Profiles ── */
.preset-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--slate);
    margin-bottom: 0.6rem;
}

/* ── Form Panel ── */
.panel {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 1.25rem;
    padding: 2rem 2.25rem;
    box-shadow: 0 8px 30px -8px rgba(11, 24, 41, 0.07);
}
.form-section-title {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #F1F5F9;
}
.section-pill {
    background: var(--navy);
    color: var(--gold-light);
    font-size: 0.72rem;
    font-weight: 800;
    width: 1.55rem;
    height: 1.55rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.section-title-text {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--navy);
}

/* ── Streamlit widget polish ── */
label { font-weight: 600 !important; font-size: 0.83rem !important; color: #334155 !important; }
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 0.6rem !important;
    border-color: var(--border) !important;
    background: #FAFBFC !important;
    font-size: 0.88rem !important;
}
.stSlider > div[data-baseweb="slider"] > div { padding: 0 !important; }
.stRadio > div { gap: 0.5rem !important; }
.stRadio label { 
    background: #FAFBFC;
    border: 1px solid var(--border);
    border-radius: 0.6rem;
    padding: 0.55rem 1rem;
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 500 !important;
}

/* ── Primary submit button ── */
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-soft) 100%) !important;
    color: white !important;
    border: 1px solid rgba(201, 151, 59, 0.4) !important;
    border-radius: 0.85rem !important;
    padding: 0.9rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 6px 20px -4px rgba(11,24,41,0.35), inset 0 1px 0 rgba(255,255,255,0.07) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
div.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #162C4A 0%, #1E3A5F 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 28px -4px rgba(11,24,41,0.45), inset 0 1px 0 rgba(255,255,255,0.07) !important;
}
div.stFormSubmitButton > button:active { transform: translateY(0) !important; }

/* Preset buttons */
div.stButton > button {
    background: white !important;
    color: var(--navy) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 0.5rem !important;
    transition: all 0.15s !important;
}
div.stButton > button:hover {
    border-color: var(--gold) !important;
    box-shadow: 0 4px 12px -3px rgba(201,151,59,0.2) !important;
    color: var(--gold) !important;
}

/* ── Quote Card ── */
.quote-card {
    background: linear-gradient(160deg, var(--navy) 0%, var(--navy-soft) 100%);
    border-radius: 1.25rem;
    padding: 2rem;
    color: white;
    box-shadow: 0 20px 50px -12px rgba(11, 24, 41, 0.4);
    position: relative;
    overflow: hidden;
}
.quote-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(201,151,59,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.quote-card::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.quote-eyebrow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.quote-eyebrow-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--gold-light);
}
.quote-status-dot {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
}
.status-dot {
    width: 0.45rem; height: 0.45rem;
    border-radius: 50%;
    background: #34D399;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.25);
}
.quote-main-amount {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    line-height: 1;
    color: white;
    letter-spacing: -1px;
    margin-bottom: 0.35rem;
}
.quote-main-amount sup {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    vertical-align: super;
    color: rgba(255,255,255,0.8);
}
.quote-per {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.45);
    margin-bottom: 1.75rem;
    font-weight: 500;
}
.quote-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 0 0 1.2rem;
}
.quote-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    font-size: 0.83rem;
    color: rgba(255,255,255,0.6);
}
.quote-line-val {
    font-weight: 700;
    color: white;
    font-size: 0.83rem;
}
.quote-footer {
    margin-top: 1.5rem;
    padding: 0.9rem 1rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0.75rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
    line-height: 1.55;
}
.quote-footer strong { color: var(--gold-light); }

/* ── Perks Panel ── */
.perks-panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 4px 12px -4px rgba(11,24,41,0.06);
    margin-top: 1.25rem;
}
.perks-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.perks-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
}
.perk-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    color: #334155;
    font-weight: 500;
}
.perk-icon { font-size: 1rem; }

/* ── Comparison bar ── */
.compare-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #F8FFF8;
    border: 1px solid #BBF7D0;
    border-radius: 0.75rem;
    padding: 0.9rem 1.25rem;
    margin-top: 1.25rem;
}
.compare-label { font-size: 0.78rem; color: var(--slate); font-weight: 600; }
.compare-val-good { font-size: 0.88rem; font-weight: 800; color: var(--success); }
.compare-val-cross { font-size: 0.88rem; font-weight: 700; color: #94A3B8; text-decoration: line-through; }

/* ── Page footer ── */
.apex-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: 0.76rem;
    color: var(--slate-lt);
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3.  MODEL LOADING (cached)
# ─────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

RISK_MAP = {
    "none": 0, "no disease": 0, "thyroid": 5,
    "diabetes": 6, "high blood pressure": 6, "heart disease": 8,
}

@st.cache_resource(show_spinner="Loading underwriting engine…")
def load_engine():
    try:
        m_y  = joblib.load(f"{ARTIFACTS_DIR}/model_young.joblib")
        b_y  = joblib.load(f"{ARTIFACTS_DIR}/scaler_young.joblib")
        m_r  = joblib.load(f"{ARTIFACTS_DIR}/model_rest.joblib")
        b_r  = joblib.load(f"{ARTIFACTS_DIR}/scaler_rest.joblib")
        return dict(m_y=m_y, sc_y=b_y["scaler"], cols_y=b_y["cols_to_scale"],
                    m_r=m_r, sc_r=b_r["scaler"], cols_r=b_r["cols_to_scale"], ok=True)
    except Exception as e:
        return dict(ok=False, err=str(e))

engine = load_engine()

def predict(payload, young):
    df = pd.DataFrame([payload])
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df["number_of_dependants"] = abs(df["number_of_dependants"])

    med = str(payload.get("medical_history", "none")).lower()
    parts = [p.strip() for p in med.split("&")]
    df["normalized_risk_score"] = sum(RISK_MAP.get(p, 0) for p in parts) / 14.0

    df["insurance_plan"] = {"Bronze": 1, "Silver": 2, "Gold": 3}.get(payload["insurance_plan"], 2)
    df["income_level"] = 1

    df["gender_Male"]                    = int(payload["gender"] == "Male")
    df["region_Northwest"]               = int(payload["region"] == "Northwest")
    df["region_Southeast"]               = int(payload["region"] == "Southeast")
    df["region_Southwest"]               = int(payload["region"] == "Southwest")
    df["marital_status_Unmarried"]       = int(payload["marital_status"] == "Unmarried")
    df["bmi_category_Obesity"]           = int(payload["bmi_category"] == "Obesity")
    df["bmi_category_Overweight"]        = int(payload["bmi_category"] == "Overweight")
    df["bmi_category_Underweight"]       = int(payload["bmi_category"] == "Underweight")
    df["smoking_status_Occasional"]      = int(payload["smoking_status"] == "Occasional")
    df["smoking_status_Regular"]         = int(payload["smoking_status"] == "Regular")
    df["employment_status_Salaried"]     = int(payload["employment_status"] == "Salaried")
    df["employment_status_Self-Employed"]= int(payload["employment_status"] == "Self-Employed")

    sc   = engine["sc_y"]   if young else engine["sc_r"]
    cols = engine["cols_y"] if young else engine["cols_r"]
    df[cols] = sc.transform(df[cols])
    X = df.drop(columns=["income_level"])
    feat_order = [
        "age","number_of_dependants","income_lakhs","insurance_plan",
        "genetical_risk","normalized_risk_score","gender_Male",
        "region_Northwest","region_Southeast","region_Southwest",
        "marital_status_Unmarried","bmi_category_Obesity",
        "bmi_category_Overweight","bmi_category_Underweight",
        "smoking_status_Occasional","smoking_status_Regular",
        "employment_status_Salaried","employment_status_Self-Employed",
    ]
    m = engine["m_y"] if young else engine["m_r"]
    return float(m.predict(X[feat_order])[0])

# ─────────────────────────────────────────────
# 4.  NAV BAR (with inline SVG logo)
# ─────────────────────────────────────────────
SHIELD_SVG = """
<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M18 3L5 8.5V18C5 25.18 10.74 31.94 18 33C25.26 31.94 31 25.18 31 18V8.5L18 3Z"
        fill="#C9973B" fill-opacity="0.18"/>
  <path d="M18 3L5 8.5V18C5 25.18 10.74 31.94 18 33C25.26 31.94 31 25.18 31 18V8.5L18 3Z"
        stroke="#C9973B" stroke-width="1.75" stroke-linejoin="round"/>
  <path d="M13 18L16.5 21.5L23 15" stroke="#E8BE6C" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

st.markdown(f"""
<div class="apex-nav">
  <a class="apex-logo" href="#">
    {SHIELD_SVG}
    <span class="apex-logo-text">Apex<span>Shield</span></span>
  </a>
  <div class="apex-nav-links">
    <span class="nav-active">Get Quote</span>
    <span>Our Plans</span>
    <span>Claims</span>
    <span>About</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5.  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="apex-hero">
  <h1>Your personalised health cover<br>quote, <em>in seconds</em></h1>
  <p>Answer a few quick questions and our AI-powered underwriting
     engine computes a fair, transparent premium — no callbacks, no paperwork.</p>
  <div class="trust-badges">
    <span class="trust-badge"><span class="dot">●</span> IRDAI Compliant</span>
    <span class="trust-badge">🔒 256-bit Encrypted</span>
    <span class="trust-badge">⚡ 14,000+ Network Hospitals</span>
    <span class="trust-badge">🏆 99.7% Claim Settlement</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6.  PRESET QUICK PROFILES
# ─────────────────────────────────────────────
PROFILES = {
    "🌱 Young Individual": dict(age=22, gender="Female", dependants=0, income=10.0,
        marital="Unmarried", region="Northwest", bmi="Normal", smoking="No",
        medical="No Disease", gr=1, plan="Silver", emp="Salaried"),
    "⚠️ Young + Family History": dict(age=24, gender="Male", dependants=0, income=14.0,
        marital="Unmarried", region="Northeast", bmi="Normal", smoking="No",
        medical="No Disease", gr=4, plan="Silver", emp="Salaried"),
    "💼 Family Breadwinner": dict(age=42, gender="Male", dependants=2, income=32.0,
        marital="Married", region="Southeast", bmi="Overweight", smoking="Occasional",
        medical="High Blood Pressure", gr=2, plan="Gold", emp="Salaried"),
    "🩺 Senior Applicant": dict(age=57, gender="Female", dependants=1, income=22.0,
        marital="Married", region="Southwest", bmi="Obesity", smoking="Regular",
        medical="Diabetes & Heart Disease", gr=3, plan="Gold", emp="Self-Employed"),
}

if "prof" not in st.session_state:
    st.session_state.prof = PROFILES["🌱 Young Individual"]

st.markdown('<p class="preset-label">✦ Quick-fill from a sample profile</p>', unsafe_allow_html=True)
pc = st.columns(4)
for idx, (label, data) in enumerate(PROFILES.items()):
    if pc[idx].button(label, use_container_width=True):
        st.session_state.prof = data

p = st.session_state.prof

# ─────────────────────────────────────────────
# 7.  MAIN 2-COLUMN LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([11, 9], gap="large")

# ── LEFT: FORM ──────────────────────────────
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    with st.form("quote_form", border=False):

        # ── Section 1 ─────────────────────
        st.markdown("""<div class="form-section-title">
            <span class="section-pill">1</span>
            <span class="section-title-text">Personal Details</span>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        age     = c1.number_input("Age", 18, 85, int(p["age"]), help="Minimum age: 18 years")
        gender  = c2.selectbox("Gender", ["Female", "Male"],
                               index=0 if p["gender"] == "Female" else 1)

        c3, c4 = st.columns(2)
        marital = c3.selectbox("Marital Status", ["Unmarried", "Married"],
                               index=0 if p["marital"] == "Unmarried" else 1)
        deps    = c4.selectbox("Dependants to Cover", [0, 1, 2, 3, 4, 5],
                               index=min(p["dependants"], 5),
                               format_func=lambda x: "None – just me" if x == 0 else f"{x} dependant{'s' if x > 1 else ''}")

        c5, c6 = st.columns(2)
        income  = c5.slider("Annual Income (₹ Lakhs)", 1.0, 80.0,
                             float(p["income"]), step=1.0, format="₹%g L")
        region  = c6.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"],
                               index=["Northwest","Northeast","Southeast","Southwest"].index(p["region"]))

        # ── Section 2 ─────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="form-section-title">
            <span class="section-pill">2</span>
            <span class="section-title-text">Health & Lifestyle</span>
        </div>""", unsafe_allow_html=True)

        c7, c8 = st.columns(2)
        bmi_opts = ["Normal", "Overweight", "Obesity", "Underweight"]
        bmi     = c7.selectbox("Body Profile (BMI)", bmi_opts,
                               index=bmi_opts.index(p["bmi"]))
        smk_opts = ["No", "Occasional", "Regular"]
        smoking  = c8.selectbox("Tobacco Use", smk_opts,
                                index=smk_opts.index(p["smoking"]),
                                format_func=lambda x: {"No":"Non-Smoker","Occasional":"Occasional","Regular":"Regular Smoker"}[x])

        med_opts = [
            "No Disease","Diabetes","High Blood Pressure","Thyroid","Heart Disease",
            "Diabetes & High Blood Pressure","Diabetes & Heart Disease","High Blood Pressure & Heart Disease",
        ]
        medical = st.selectbox("Known Pre-existing Conditions", med_opts,
                               index=med_opts.index(p["medical"]) if p["medical"] in med_opts else 0)

        gr_labels = {
            0: "0  –  No family illness history",
            1: "1  –  Minor, distant relatives only",
            2: "2  –  Mild hereditary patterns",
            3: "3  –  Moderate  (parent with chronic illness)",
            4: "4  –  Elevated  (cardiovascular in close family)",
            5: "5  –  High  (multiple direct relatives affected)",
        }
        gr = st.select_slider("Family Health History", options=[0,1,2,3,4,5],
                              value=int(p["gr"]), format_func=lambda x: gr_labels[x])

        emp_opts = ["Salaried", "Self-Employed", "Freelancer"]
        emp = st.selectbox("Employment Type", emp_opts,
                           index=emp_opts.index(p["emp"]) if p["emp"] in emp_opts else 0)

        # ── Section 3 ─────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="form-section-title">
            <span class="section-pill">3</span>
            <span class="section-title-text">Choose Your Coverage Plan</span>
        </div>""", unsafe_allow_html=True)

        plan = st.radio("Coverage Tier", ["Bronze", "Silver", "Gold"],
                        index=["Bronze","Silver","Gold"].index(p["plan"]),
                        horizontal=True,
                        format_func=lambda x: {
                            "Bronze": "🥉  Bronze  ·  Essential",
                            "Silver": "🥈  Silver  ·  Comprehensive  ★",
                            "Gold":   "🥇  Gold    ·  Premium Global",
                        }[x])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "  Get My Instant Quote  →",
            type="primary",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)   # close .panel

# ── RIGHT: LIVE QUOTE CARD ───────────────────
with right:
    if not engine.get("ok"):
        st.error(f"Engine error: {engine.get('err')}")
        st.stop()

    # Always compute (form re-runs on submit automatically)
    payload = dict(
        age=age, gender=gender, region=region, marital_status=marital,
        number_of_dependants=deps, bmi_category=bmi, smoking_status=smoking,
        employment_status=emp, income_lakhs=income,
        medical_history=medical, insurance_plan=plan, genetical_risk=gr,
    )
    young = age <= 25
    raw   = predict(payload, young)
    annual = max(round(raw), 3000)
    monthly = round(annual / 12)

    # Price breakdown (simplified, user-facing)
    plan_factor = {"Bronze": 0.78, "Silver": 1.0, "Gold": 1.32}[plan]
    base    = int(annual / plan_factor * 0.82)
    plan_up = int(annual - base * plan_factor)
    life_up = (800 if bmi == "Obesity" else (350 if bmi == "Overweight" else 0)) + \
              (700 if smoking == "Regular" else (280 if smoking == "Occasional" else 0))
    hist_up = gr * 420 if young else gr * 180
    base    = max(annual - plan_up - life_up - hist_up, 1800)

    # Cohort routing note (user-friendly)
    note = ("Tailored pricing for applicants under 25 — family health history "
            "plays a key role in determining your premium." if young else
            "Your quote factors in age, clinical history, lifestyle, and "
            "selected coverage tier.")

    st.markdown(f"""
<div class="quote-card">
  <div class="quote-eyebrow">
    <span class="quote-eyebrow-label">Your Personalised Quote</span>
    <span class="quote-status-dot"><span class="status-dot"></span> Live Estimate</span>
  </div>

  <div class="quote-main-amount"><sup>₹</sup>{annual:,}</div>
  <div class="quote-per">per year &nbsp;·&nbsp; approx. <strong style="color:rgba(255,255,255,0.85)">₹{monthly:,}</strong> / month</div>

  <hr class="quote-divider">

  <div class="quote-line">
    <span>Base hospitalisation cover</span>
    <span class="quote-line-val">₹{base:,}</span>
  </div>
  <div class="quote-line">
    <span>{plan} plan upgrade</span>
    <span class="quote-line-val">+ ₹{plan_up:,}</span>
  </div>
  <div class="quote-line">
    <span>Lifestyle / BMI loading</span>
    <span class="quote-line-val">{"+ ₹{:,}".format(life_up) if life_up else "₹ 0"}</span>
  </div>
  <div class="quote-line">
    <span>Family health history</span>
    <span class="quote-line-val">{"+ ₹{:,}".format(hist_up) if hist_up else "₹ 0"}</span>
  </div>

  <div class="quote-footer">
    <strong>Why this price?</strong> &nbsp;{note}
  </div>
</div>
""", unsafe_allow_html=True)

    # Comparison saving pill
    baseline = int(annual * (1.28 if young and gr >= 3 else 1.06))
    saving   = baseline - annual
    if saving > 0:
        st.markdown(f"""
<div class="compare-row">
  <div>
    <div class="compare-label">Industry Standard Estimate</div>
    <div class="compare-val-cross">₹{baseline:,} / yr</div>
  </div>
  <div style="text-align:right">
    <div class="compare-label">You Save with ApexShield</div>
    <div class="compare-val-good">− ₹{saving:,} / yr</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # What's included
    st.markdown("""
<div class="perks-panel">
  <div class="perks-title">✦ &nbsp;What's included in every plan</div>
  <div class="perks-grid">
    <div class="perk-item"><span class="perk-icon">🏥</span> 14,000+ Network Hospitals</div>
    <div class="perk-item"><span class="perk-icon">⚡</span> Cashless Claim in 30 mins</div>
    <div class="perk-item"><span class="perk-icon">🩺</span> Annual Free Health Check-up</div>
    <div class="perk-item"><span class="perk-icon">💊</span> Pre & Post Hospitalisation</div>
    <div class="perk-item"><span class="perk-icon">🚑</span> Emergency Ambulance Cover</div>
    <div class="perk-item"><span class="perk-icon">🧘</span> Mental Wellness Sessions</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8.  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="apex-footer">
  ApexShield Life &amp; Health Insurance Co. · IRDAI Reg. No. 142 · CIN: U66010MH2024PLC000000<br>
  Premiums are indicative, subject to underwriting review. Quotes valid for 30 days.
</div>
""", unsafe_allow_html=True)
