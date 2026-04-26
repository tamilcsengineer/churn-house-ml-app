import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Churn Intelligence", page_icon="📡", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 3rem; }

    .title-box {
        background: linear-gradient(135deg, #1f77b4, #ff7f0e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .title-box h1 { color: white; font-size: 2rem; margin: 0; }
    .title-box p  { color: #ffffffcc; margin: 0.3rem 0 0; }

    .metric-card {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { color: #aaa; font-size: 0.85rem; margin: 0; }
    .metric-card h2 { color: white; font-size: 1.8rem; margin: 0.3rem 0 0; }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid #2e3250;
        padding-bottom: 0.4rem;
    }

    .reason-box {
        background: #1e2130;
        border-left: 4px solid #f0a500;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 0.5rem;
    }
    .action-box {
        background: #1e2130;
        border-left: 4px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 0.5rem;
    }
    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(135deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        padding: 0.7rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & Prepare Data ──────────────────────────────────────
data = pd.read_csv("churn.csv")
data = data.drop("customerID", axis=1)
data = data.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
data = data.replace({"Yes": 1, "No": 0})
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna(subset=["Churn"])
data = data.fillna(data.median(numeric_only=True))
data = pd.get_dummies(data)

X = data.drop("Churn", axis=1)
y = data["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = round(model.score(X_test, y_test) * 100, 2)
total_customers = len(data)
churn_rate = round(y.mean() * 100, 2)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>📡 Customer Churn Intelligence System</h1>
    <p>AI-powered churn prediction with risk analysis and business recommendations</p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ──────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-card">
        <h3>MODEL ACCURACY</h3><h2>{accuracy}%</h2></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <h3>TOTAL CUSTOMERS</h3><h2>{total_customers:,}</h2></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <h3>DATASET CHURN RATE</h3><h2>{churn_rate}%</h2></div>""", unsafe_allow_html=True)

# ── Input Panel ──────────────────────────────────────────────
st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Info**")
    tenure        = st.slider("Tenure (months)", 0, 72, 12)
    contract      = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment       = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                     "Bank transfer", "Credit card"])

with col2:
    st.markdown("**Billing**")
    monthly       = st.slider("Monthly Charges ($)", 0, 150, 65)
    total         = st.slider("Total Charges ($)", 0, 10000, monthly * tenure)
    paperless     = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

with col3:
    st.markdown("**Services**")
    internet      = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support  = st.radio("Tech Support", ["Yes", "No"], horizontal=True)
    online_sec    = st.radio("Online Security", ["Yes", "No"], horizontal=True)
    senior        = st.radio("Senior Citizen", ["Yes", "No"], horizontal=True)

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Analyse Customer"):

    input_df = pd.DataFrame([[0] * len(X.columns)], columns=X.columns)

    # Map inputs to model features
    field_map = {
        "tenure":                tenure,
        "MonthlyCharges":        monthly,
        "TotalCharges":          total,
        "SeniorCitizen":         1 if senior == "Yes" else 0,
        "PaperlessBilling":      1 if paperless == "Yes" else 0,
        "TechSupport":           1 if tech_support == "Yes" else 0,
        "OnlineSecurity":        1 if online_sec == "Yes" else 0,
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year":       1 if contract == "One year" else 0,
        "Contract_Two year":       1 if contract == "Two year" else 0,
        "InternetService_DSL":         1 if internet == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No":          1 if internet == "No" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check":     1 if payment == "Mailed check" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if payment == "Bank transfer" else 0,
        "PaymentMethod_Credit card (automatic)":   1 if payment == "Credit card" else 0,
    }

    for col, val in field_map.items():
        if col in input_df.columns:
            input_df[col] = val

    prob       = model.predict_proba(input_df)[0][1]
    prob_pct   = round(prob * 100, 2)

    # ── Result ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 2])

    with r1:
        if prob > 0.7:
            st.error(f"🔴 HIGH RISK\n\n### {prob_pct}% churn probability")
        elif prob > 0.4:
            st.warning(f"🟡 MEDIUM RISK\n\n### {prob_pct}% churn probability")
        else:
            st.success(f"🟢 LOW RISK\n\n### {prob_pct}% churn probability")

    with r2:
        st.progress(int(prob * 100))
        st.caption(f"Churn probability score: {prob_pct}%")

    # ── Reasons ──────────────────────────────────────────────
    reasons = []
    if tenure < 10:
        reasons.append("📅 Low tenure — customer not yet loyal")
    if monthly > 70:
        reasons.append("💸 High monthly charges — possible cost sensitivity")
    if contract == "Month-to-month":
        reasons.append("📋 Month-to-month contract — easy to leave")
    if internet == "Fiber optic" and tech_support == "No":
        reasons.append("🛠️ Fiber user with no tech support — frustration risk")
    if payment == "Electronic check":
        reasons.append("💳 Electronic check payment — linked to higher churn historically")
    if not reasons:
        reasons.append("✅ No major risk factors detected")

    st.markdown('<div class="section-title">Risk Factors</div>', unsafe_allow_html=True)
    st.markdown('<div class="reason-box">' +
                "<br>".join(reasons) + "</div>", unsafe_allow_html=True)

    # ── Actions ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Suggested Actions</div>', unsafe_allow_html=True)

    if prob > 0.7:
        actions = [
            "🎁 Offer a 20–30% discount or loyalty reward",
            "📞 Priority call from retention team within 24 hrs",
            "📦 Propose annual or two-year contract upgrade",
        ]
    elif prob > 0.4:
        actions = [
            "📧 Send personalised re-engagement email",
            "🛡️ Offer free tech support or service upgrade",
            "📊 Monitor account activity over next 30 days",
        ]
    else:
        actions = [
            "✅ Customer is stable — no immediate action needed",
            "🔔 Continue standard engagement cadence",
        ]

    st.markdown('<div class="action-box">' +
                "<br>".join(actions) + "</div>", unsafe_allow_html=True)