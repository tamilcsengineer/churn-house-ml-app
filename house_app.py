import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding: 2rem 3rem; }

    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .title-box h1 { color: white; font-size: 2rem; margin: 0; }
    .title-box p  { color: #aaaacc; margin: 0.3rem 0 0; }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { color: #aaa; font-size: 0.8rem; margin: 0; text-transform: uppercase; }
    .metric-card h2 { color: white; font-size: 1.6rem; margin: 0.3rem 0 0; }

    .result-box {
        background: #1a1a2e;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.5rem;
    }
    .insight-box {
        background: #1a1a2e;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 0.5rem;
        color: #ccccee;
    }
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid #0f3460;
        padding-bottom: 0.4rem;
    }
    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(135deg, #e94560, #0f3460);
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

# ── Load & Train ─────────────────────────────────────────────
data = pd.read_csv("house.csv")

data = data[["GrLivArea", "BedroomAbvGr", "SalePrice"]].dropna()

X = data[["GrLivArea", "BedroomAbvGr"]]
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

score      = round(model.score(X_test, y_test) * 100, 2)
avg_price  = round(y.mean(), 2)
max_price  = round(y.max(), 2)
min_price  = round(y.min(), 2)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🏠 House Price Prediction System</h1>
    <p>AI-powered price estimation using area and bedroom data</p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["MODEL SCORE", "AVG PRICE ($)", "MAX PRICE ($)", "MIN PRICE ($)"],
    [f"{score}%", f"{avg_price:,.0f}", f"{max_price:,.0f}", f"{min_price:,.0f}"]
):
    with col:
        st.markdown(f"""<div class="metric-card">
            <h3>{label}</h3><h2>{value}</h2></div>""", unsafe_allow_html=True)

# ── Inputs ───────────────────────────────────────────────────
st.markdown('<div class="section-title">Enter House Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    area     = st.slider("Living Area (sq ft)", 500, 5000, 1500)

with col2:
    bedrooms = st.slider("Bedrooms", 1, 6, 3)

with col3:
    quality  = st.selectbox("Overall Quality", [
        "Excellent", "Good", "Average", "Below Average"
    ])

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Predict House Price"):

    prediction = model.predict([[area, bedrooms]])[0]
    pred_inr   = round(prediction * 83, 2)   # USD → INR conversion

    # ── Result ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True)

    r1, r2 = st.columns([1, 2])

    with r1:
        if prediction > 300000:
            st.error(f"💎 Luxury House\n\n### ${prediction:,.0f} USD")
        elif prediction > 175000:
            st.warning(f"🏡 Mid Range\n\n### ${prediction:,.0f} USD")
        else:
            st.success(f"🏠 Affordable\n\n### ${prediction:,.0f} USD")

        st.markdown(f"""<div class="result-box">
            <b>💵 USD:</b> ${prediction:,.0f}<br>
            <b>🇮🇳 INR:</b> ₹ {pred_inr:,.0f}
        </div>""", unsafe_allow_html=True)

    with r2:
        pct = min(int((prediction / max_price) * 100), 100)
        st.caption(f"Price percentile vs dataset max")
        st.progress(pct)
        st.caption(f"This house is priced at {pct}% of the highest recorded price")

    # ── Insights ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Property Insights</div>',
                unsafe_allow_html=True)

    insights = []
    if area > 3000:
        insights.append("📐 Large living area — premium pricing applies")
    elif area < 1000:
        insights.append("📐 Compact size — great for affordable segment")

    if bedrooms >= 4:
        insights.append("🛏️ Multi-bedroom home — suited for large families")
    elif bedrooms <= 2:
        insights.append("🛏️ Fewer bedrooms — ideal for couples or singles")

    if quality == "Excellent":
        insights.append("⭐ Excellent quality finish — expect higher buyer interest")
    elif quality == "Below Average":
        insights.append("🔧 Below average quality — renovation may increase value")

    if prediction > avg_price:
        insights.append(f"📈 Above average market price (avg: ${avg_price:,.0f})")
    else:
        insights.append(f"📉 Below average market price (avg: ${avg_price:,.0f})")

    st.markdown('<div class="insight-box">' +
                "<br>".join(insights) + "</div>", unsafe_allow_html=True)

    # ── Suggestions ──────────────────────────────────────────
    st.markdown('<div class="section-title">Recommendations</div>',
                unsafe_allow_html=True)

    if prediction > 300000:
        tips = [
            "💼 Target high-net-worth buyers or investors",
            "📸 Professional staging and photography recommended",
            "🏦 Offer flexible financing or installment plans",
        ]
    elif prediction > 175000:
        tips = [
            "📢 List on major real estate platforms",
            "🤝 Negotiate with first-time home buyer schemes",
            "🔨 Minor renovations can push to luxury tier",
        ]
    else:
        tips = [
            "🏠 Great option for first-time buyers",
            "📋 Apply for government housing loan schemes",
            "📈 Location upgrades can significantly boost value",
        ]

    st.markdown('<div class="insight-box">' +
                "<br>".join(tips) + "</div>", unsafe_allow_html=True)