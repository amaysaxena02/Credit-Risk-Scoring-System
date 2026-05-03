"""
app.py — Credit Risk Scoring System: Streamlit UI
Real-time loan default probability prediction with SHAP explainability.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

sys.path.insert(0, os.path.dirname(__file__))
from modules.feature_engineering import engineer_features
from modules.explainability import explain_single_prediction

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1428 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1428 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}

/* Cards / containers */
.risk-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* Section headings */
.section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(99, 179, 237, 0.2);
}

/* Metric cards */
.metric-block {
    background: rgba(99, 179, 237, 0.08);
    border: 1px solid rgba(99, 179, 237, 0.25);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-label {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #a0aec0;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -1px;
}

.risk-low    { color: #48bb78; }
.risk-medium { color: #ed8936; }
.risk-high   { color: #fc5c65; }

/* Gauge bar */
.gauge-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    margin: 10px 0;
}
.gauge-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 32px;
    font-size: 15px;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(49, 130, 206, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    box-shadow: 0 6px 20px rgba(49, 130, 206, 0.45);
    transform: translateY(-1px);
}

/* Input labels */
label { color: #cbd5e0 !important; font-size: 13px !important; }

/* Divider */
hr { border-color: rgba(99, 179, 237, 0.15); }

/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #63b3ed;
    margin-bottom: 4px;
}
.sidebar-sub {
    font-size: 12px;
    color: #718096;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts ...")
def load_artifacts():
    model_path = 'models/xgb_model.pkl'
    prep_path = 'models/preprocessor.pkl'
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        return None, None, None
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    explainer = shap.TreeExplainer(model)
    return model, preprocessor, explainer


model, preprocessor, explainer = load_artifacts()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">💳 CreditRisk AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">LendingClub Default Probability Engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-title">📋 Loan Details</div>', unsafe_allow_html=True)
    loan_amnt     = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    funded_amnt   = st.number_input("Funded Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    term          = st.selectbox("Term", [" 36 months", " 60 months"])
    int_rate      = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)
    installment   = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=2000.0, value=300.0, step=10.0)
    purpose       = st.selectbox("Loan Purpose", [
        'debt_consolidation', 'credit_card', 'home_improvement', 'other',
        'major_purchase', 'small_business', 'car', 'medical', 'moving',
        'vacation', 'house', 'wedding', 'renewable_energy', 'educational'
    ])
    initial_list_status = st.selectbox("Initial List Status", ['w', 'f'])

    st.markdown("---")
    st.markdown('<div class="section-title">👤 Borrower Profile</div>', unsafe_allow_html=True)
    annual_inc    = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=60000, step=1000)
    emp_length    = st.selectbox("Employment Length", [
        '< 1 year', '1 year', '2 years', '3 years', '4 years',
        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
    ])
    home_ownership= st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    verification_status = st.selectbox("Verification Status", ['Verified', 'Not Verified', 'Source Verified'])
    grade         = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    sub_grade     = st.selectbox("Sub Grade", [
        'A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
        'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
        'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5'
    ])
    addr_state    = st.selectbox("State", [
        'CA','NY','TX','FL','IL','PA','OH','GA','NJ','NC','MI','VA','WA','MA',
        'AZ','CO','TN','MO','MD','WI','MN','OR','IN','CT','LA','AL','SC','KY',
        'OK','UT','AR','NV','KS','MS','NM','NE','WV','ID','HI','ME','NH','RI',
        'MT','DE','SD','AK','ND','VT','DC','WY'
    ])

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Credit History</div>', unsafe_allow_html=True)
    dti             = st.slider("Debt-to-Income Ratio (%)", 0.0, 40.0, 15.0, step=0.1)
    delinq_2yrs     = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)
    mths_since_last_delinq = st.number_input("Months Since Last Delinquency", 0, 180, 0)
    inq_last_6mths  = st.number_input("Inquiries (last 6 months)", 0, 20, 1)
    open_acc        = st.number_input("Open Credit Lines", 0, 60, 10)
    pub_rec         = st.number_input("Public Records", 0, 20, 0)
    pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 10, 0)
    revol_bal       = st.number_input("Revolving Balance ($)", 0, 200000, 8000, step=500)
    revol_util      = st.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0, step=0.5)
    bc_util         = st.slider("Bank Card Utilization (%)", 0.0, 100.0, 50.0, step=0.5)
    total_acc       = st.number_input("Total Credit Lines", 0, 100, 22)
    mort_acc        = st.number_input("Mortgage Accounts", 0, 20, 1)
    avg_cur_bal     = st.number_input("Avg Current Balance ($)", 0, 200000, 5000, step=500)
    tot_cur_bal     = st.number_input("Total Current Balance ($)", 0, 2000000, 50000, step=1000)
    tot_hi_cred_lim = st.number_input("Total High Credit Limit ($)", 0, 2000000, 100000, step=5000)
    num_actv_rev_tl = st.number_input("Active Revolving Accounts", 0, 50, 5)
    num_bc_sats     = st.number_input("Satisfactory Bank Cards", 0, 30, 4)
    total_rev_hi_lim= st.number_input("Total Revolving Credit Limit ($)", 0, 500000, 30000, step=1000)
    pct_tl_nvr_dlq  = st.slider("% Accounts Never Delinquent", 0.0, 100.0, 90.0, step=0.5)
    earliest_cr_line= st.number_input("Credit History Start Year", 1960, 2023, 2005)
    collections_12_mths_ex_med = st.number_input("Collections (12 months)", 0, 20, 0)
    acc_now_delinq  = st.number_input("Accounts Now Delinquent", 0, 20, 0)

    predict_btn = st.button("🔍  Predict Default Risk", use_container_width=True)


# ── Main Page Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 20px;">
    <h1 style="font-size:40px; font-weight:800; background: linear-gradient(135deg,#63b3ed,#b794f4);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        Credit Risk Scoring System
    </h1>
    <p style="color:#718096; font-size:15px; margin-top:8px;">
        XGBoost · SHAP Explainability · LendingClub Dataset
    </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model artifacts not found. Please run `python main.py` first to train the model.")
    st.info("Expected files: `models/xgb_model.pkl` and `models/preprocessor.pkl`")
    st.stop()


# ── Prediction Logic ──────────────────────────────────────────────────────────
def build_input_df():
    """Build raw input DataFrame matching training column schema."""
    return pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'funded_amnt': funded_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grade,
        'sub_grade': sub_grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': verification_status,
        'purpose': purpose,
        'addr_state': addr_state,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'earliest_cr_line': f'Jan-{earliest_cr_line}',
        'inq_last_6mths': inq_last_6mths,
        'mths_since_last_delinq': mths_since_last_delinq,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'initial_list_status': initial_list_status,
        'collections_12_mths_ex_med': collections_12_mths_ex_med,
        'acc_now_delinq': acc_now_delinq,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'pct_tl_nvr_dlq': pct_tl_nvr_dlq,
        'bc_util': bc_util,
        'avg_cur_bal': avg_cur_bal,
        'tot_cur_bal': tot_cur_bal,
        'tot_hi_cred_lim': tot_hi_cred_lim,
        'num_actv_rev_tl': num_actv_rev_tl,
        'num_bc_sats': num_bc_sats,
        'total_rev_hi_lim': total_rev_hi_lim,
    }])


def get_risk_category(prob):
    if prob < 0.30:
        return "LOW RISK", "risk-low", "#48bb78"
    elif prob < 0.60:
        return "MEDIUM RISK", "risk-medium", "#ed8936"
    else:
        return "HIGH RISK", "risk-high", "#fc5c65"


def make_gauge_html(prob, color):
    pct = prob * 100
    return f"""
    <div class="gauge-bar-bg">
        <div class="gauge-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:11px; color:#718096; margin-top:2px;">
        <span>0%</span><span>50%</span><span>100%</span>
    </div>
    """


# ── Default landing state ─────────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "🎯", "Real-Time Prediction", "Instant probability of default using XGBoost"),
        (col2, "🔍", "SHAP Explainability", "Understand why a prediction was made"),
        (col3, "⚖️", "Imbalance-Aware", "SMOTE-balanced training for fair risk scoring"),
    ]:
        with col:
            st.markdown(f"""
            <div class="risk-card" style="text-align:center; padding:32px 20px;">
                <div style="font-size:36px; margin-bottom:14px;">{icon}</div>
                <div style="font-weight:600; font-size:15px; color:#e2e8f0; margin-bottom:8px;">{title}</div>
                <div style="font-size:13px; color:#718096;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#4a5568; font-size:13px; margin-top:10px;">
        ← Fill in borrower details in the sidebar, then click <strong style="color:#63b3ed;">Predict Default Risk</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Prediction Output ─────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Computing prediction ..."):
        # Build input
        input_df = build_input_df()
        input_df = engineer_features(input_df)
        X_input = preprocessor.transform(input_df)

        # Predict
        prob = float(model.predict_proba(X_input)[0, 1])
        risk_label, risk_class, risk_color = get_risk_category(prob)

        # SHAP for single prediction
        col_transformer = preprocessor.named_steps['col_transformer']
        feature_names = list(col_transformer.get_feature_names_out())
        shap_result = explain_single_prediction(model, X_input, feature_names, explainer)

    # ── Row 1: Key Metrics ────────────────────────────────────────
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1])

    with c1:
        st.markdown(f"""
        <div class="metric-block">
            <div class="metric-label">Probability of Default</div>
            <div class="metric-value {risk_class}">{prob*100:.1f}%</div>
            {make_gauge_html(prob, risk_color)}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-block">
            <div class="metric-label">Risk Category</div>
            <div class="metric-value {risk_class}" style="font-size:26px; margin-top:10px;">{risk_label}</div>
            <div style="font-size:12px; color:#718096; margin-top:8px;">
                Low: &lt;30% · Medium: 30–60% · High: &gt;60%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        recmd_color = '#48bb78' if prob < 0.30 else ('#ed8936' if prob < 0.60 else '#fc5c65')
        recmd_text  = 'Approve' if prob < 0.30 else ('Review' if prob < 0.60 else 'Decline')
        recmd_icon  = '✅' if prob < 0.30 else ('⚠️' if prob < 0.60 else '❌')
        st.markdown(f"""
        <div class="metric-block">
            <div class="metric-label">Recommendation</div>
            <div style="font-size:44px; margin: 6px 0;">{recmd_icon}</div>
            <div style="font-size:20px; font-weight:700; color:{recmd_color};">{recmd_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # ── Row 2: SHAP Explanation ───────────────────────────────────
    st.markdown('<div class="section-title">🔍 SHAP Feature Explanation (This Prediction)</div>', unsafe_allow_html=True)

    shap_vals = shap_result['shap_values']
    feat_names = shap_result['feature_names']
    feat_vals = shap_result['feature_values']

    # Build a sorted bar chart
    df_shap = pd.DataFrame({
        'feature': feat_names,
        'shap_value': shap_vals,
        'feature_value': feat_vals
    }).reindex(columns=['feature', 'shap_value', 'feature_value'])
    df_shap['abs_shap'] = df_shap['shap_value'].abs()
    df_shap = df_shap.sort_values('abs_shap', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    colors = ['#fc5c65' if v > 0 else '#48bb78' for v in df_shap['shap_value']]
    bars = ax.barh(df_shap['feature'], df_shap['shap_value'], color=colors, edgecolor='none', height=0.65)
    ax.axvline(0, color='#4a5568', linewidth=1)
    ax.set_title('Top 15 Feature Contributions (SHAP)', color='white', fontsize=13, pad=10)
    ax.set_xlabel('SHAP Value  →  Increases Default Risk', color='#a0aec0', fontsize=11)
    ax.tick_params(colors='#a0aec0', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Annotate bar values
    for bar, val in zip(bars, df_shap['shap_value']):
        x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.4f}', va='center', ha=ha, color='#a0aec0', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Legend ────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; gap:24px; padding:12px 0; font-size:13px; color:#718096;">
        <span><span style="color:#fc5c65; font-weight:700;">■</span>  Increases default risk</span>
        <span><span style="color:#48bb78; font-weight:700;">■</span>  Decreases default risk</span>
        <span style="margin-left:auto; color:#4a5568; font-size:11px;">SHAP base value: {:.4f}</span>
    </div>
    """.format(shap_result['base_value']), unsafe_allow_html=True)

    # ── Input Summary Table ───────────────────────────────────────
    with st.expander("📋 View Input Summary", expanded=False):
        summary = {
            'Loan Amount': f"${loan_amnt:,}",
            'Interest Rate': f"{int_rate:.1f}%",
            'Term': term.strip(),
            'Grade': grade,
            'Annual Income': f"${annual_inc:,}",
            'DTI': f"{dti:.1f}%",
            'Employment Length': emp_length,
            'Home Ownership': home_ownership,
            'Purpose': purpose,
            'Revolving Utilization': f"{revol_util:.1f}%",
            'Bank Card Utilization': f"{bc_util:.1f}%",
            'Mortgage Accounts': mort_acc,
            'Delinquencies (2yr)': delinq_2yrs,
            'Inquiries (6mo)': inq_last_6mths,
            'Bankruptcies': pub_rec_bankruptcies,
            '% Never Delinquent': f"{pct_tl_nvr_dlq:.1f}%",
        }
        st.table(pd.DataFrame(list(summary.items()), columns=['Feature', 'Value']))


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:30px 0 10px; color:#2d3748; font-size:12px;">
    Credit Risk Scoring System · XGBoost + SHAP · LendingClub Dataset<br>
    <span style="color:#1a2035;">For demonstration purposes only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)
