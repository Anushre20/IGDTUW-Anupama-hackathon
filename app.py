import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Beacon",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    risk_data = pd.read_csv(
        os.path.join(OUTPUT_DIR, "risk_engine_predictions.csv")
    )

    weights = pd.read_csv(
        os.path.join(OUTPUT_DIR, "module_a_rebalancing_weights.csv")
    )

    performance = pd.read_csv(
        os.path.join(OUTPUT_DIR, "module_a_portfolio_performance.csv")
    )

    module_b_summary = pd.read_csv(
        os.path.join(OUTPUT_DIR, "module_b_summary.csv")
    )

    stress_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "module_b_stress_test.csv")
    )

    stock_prices = pd.read_csv(
        os.path.join(DATA_DIR, "stock_prices_clean.csv")
    )

    return (
        risk_data,
        weights,
        performance,
        module_b_summary,
        stress_test,
        stock_prices
    )


(
    risk_data,
    weights,
    performance,
    module_b_summary,
    stress_test,
    stock_prices
) = load_data()


# ============================================================
# LOAD EVENT MODEL
# ============================================================

@st.cache_resource
def load_event_model():

    event_model = joblib.load(
        os.path.join(MODEL_DIR, "event_model.pkl")
    )

    event_vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "event_vectorizer.pkl")
    )

    return event_model, event_vectorizer


event_model, event_vectorizer = load_event_model()


# ============================================================
# LOAD FINBERT
# ============================================================

@st.cache_resource
def load_finbert():

    tokenizer = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


finbert_tokenizer, finbert_model, device = load_finbert()


# ============================================================
# IMPACT ESTIMATION
# ============================================================

def estimate_impact(
    sentiment,
    sentiment_confidence,
    event_type,
    event_confidence
):
    """
    Transparent rule-based impact estimate.

    This is NOT presented as a trained ML model.
    It is an interpretable estimate based on:
    - sentiment strength
    - event type
    - model confidence
    """

    score = 1.0

    # Sentiment contribution
    if sentiment == "negative":
        score += 2.5 * sentiment_confidence

    elif sentiment == "positive":
        score += 1.5 * sentiment_confidence

    else:
        score += 0.5 * sentiment_confidence


    # Event severity contribution
    high_impact_events = [
        "Geopolitical",
        "M&A / Acquisition",
        "Macroeconomic",
        "Credit / Debt"
    ]

    medium_impact_events = [
        "Earnings / Financial Results",
        "Product / Technology"
    ]

    if event_type in high_impact_events:
        score += 3.0

    elif event_type in medium_impact_events:
        score += 1.8

    else:
        score += 0.8


    # Confidence contribution
    score += 2.0 * event_confidence

    # Convert to 1–10
    score = int(round(score))

    return max(1, min(10, score))


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(impact_score):

    if impact_score >= 8:
        return "High"

    elif impact_score >= 5:
        return "Medium"

    else:
        return "Low"


# ============================================================
# LIVE RISK ENGINE
# ============================================================

def analyze_text(text):

    # --------------------------------------------------------
    # FinBERT Sentiment
    # --------------------------------------------------------

    inputs = finbert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = finbert_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_class = torch.argmax(probabilities).item()

    sentiment_labels = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }

    sentiment = sentiment_labels[predicted_class]

    sentiment_confidence = float(
        probabilities[predicted_class].cpu()
    )


    # --------------------------------------------------------
    # Event Classification
    # --------------------------------------------------------

    transformed_text = event_vectorizer.transform(
        [text]
    )

    event_prediction = event_model.predict(
        transformed_text
    )[0]

    event_confidence = None

    if hasattr(event_model, "predict_proba"):

        event_probabilities = event_model.predict_proba(
            transformed_text
        )[0]

        event_confidence = float(
            max(event_probabilities)
        )


    # --------------------------------------------------------
    # Impact Score
    # --------------------------------------------------------

    impact_score = estimate_impact(
        sentiment,
        sentiment_confidence,
        event_prediction,
        event_confidence
    )


    # --------------------------------------------------------
    # Risk Level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        impact_score
    )


    return {
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "event_type": event_prediction,
        "event_confidence": event_confidence,
        "impact_score": impact_score,
        "risk_level": risk_level
    }


# ============================================================
# HEADER
# ============================================================

st.title("📊 Beacon")

st.subheader(
    "AI-Powered Financial Risk Intelligence Platform"
)

st.success("🟢 AI Risk Engine Online")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Risk Engine",
        "Tactical Rebalancer",
        "Strategic Stress Test"
    ]
)


# ============================================================
# RISK ENGINE
# ============================================================

if page == "Risk Engine":

    st.header("🧠 Unified AI Risk Engine")

    # --------------------------------------------------------
    # Dashboard Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Signals",
        len(risk_data)
    )

    col2.metric(
        "High Risk Signals",
        (risk_data["risk_level"] == "High").sum()
    )

    col3.metric(
        "Medium Risk Signals",
        (risk_data["risk_level"] == "Medium").sum()
    )

    col4.metric(
        "Low Risk Signals",
        (risk_data["risk_level"] == "Low").sum()
    )

    st.divider()


    # ========================================================
    # LIVE ANALYZER
    # ========================================================

    st.subheader("🔍 Analyze Financial News")

    st.write(
        "Paste a financial news headline or event description "
        "to generate an AI-powered risk signal."
    )

    news_input = st.text_area(
        "Financial News / Event",
        placeholder=(
            "Example: The Federal Reserve unexpectedly "
            "raised interest rates, increasing concerns "
            "about economic growth and market stability."
        ),
        height=130
    )

    analyze_button = st.button(
        "🚀 Analyze Risk",
        type="primary"
    )

    if analyze_button:

        if not news_input.strip():

            st.warning(
                "Please enter a financial news/event description."
            )

        else:

            with st.spinner(
                "Running AI risk analysis..."
            ):

                result = analyze_text(
                    news_input
                )

            st.success(
                "Risk analysis completed successfully."
            )

            st.divider()


            # ------------------------------------------------
            # Main Results
            # ------------------------------------------------

            r1, r2, r3, r4 = st.columns(4)

            r1.metric(
                "Sentiment",
                result["sentiment"].upper()
            )

            r2.metric(
                "Event Type",
                result["event_type"]
            )

            r3.metric(
                "Impact Score",
                f"{result['impact_score']}/10"
            )

            r4.metric(
                "Risk Level",
                result["risk_level"]
            )


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            st.subheader("Model Confidence")

            c1, c2 = st.columns(2)

            c1.metric(
                "Sentiment Confidence",
                f"{result['sentiment_confidence'] * 100:.2f}%"
            )

            if result["event_confidence"] is not None:

                c2.metric(
                    "Event Confidence",
                    f"{result['event_confidence'] * 100:.2f}%"
                )


            # ------------------------------------------------
            # Impact Explanation
            # ------------------------------------------------

            st.info(
                "Impact Score is an interpretable estimate based "
                "on sentiment strength, event category and model "
                "confidence. The historical impact methodology "
                "is calibrated using observed 3-day market movement."
            )


            # ------------------------------------------------
            # Generated Signal
            # ------------------------------------------------

            st.subheader(
                "📡 Generated Risk Signal"
            )

            signal_df = pd.DataFrame({
                "Signal": [
                    "Sentiment",
                    "Sentiment Confidence",
                    "Event Classification",
                    "Event Confidence",
                    "Impact Score",
                    "Risk Level"
                ],
                "Value": [
                    result["sentiment"],
                    f"{result['sentiment_confidence'] * 100:.2f}%",
                    result["event_type"],
                    (
                        f"{result['event_confidence'] * 100:.2f}%"
                        if result["event_confidence"] is not None
                        else "N/A"
                    ),
                    f"{result['impact_score']}/10",
                    result["risk_level"]
                ]
            })

            st.dataframe(
                signal_df,
                use_container_width=True,
                hide_index=True
            )


    st.divider()


    # ========================================================
    # HISTORICAL SIGNALS
    # ========================================================

    st.subheader("📡 Historical Risk Signals")

    st.dataframe(
        risk_data[
            [
                "text",
                "sentiment",
                "event_type",
                "impact_score",
                "risk_level"
            ]
        ].head(20),
        use_container_width=True
    )


# ============================================================
# MODULE A
# ============================================================

elif page == "Tactical Rebalancer":

    st.header("📈 Tactical Index Rebalancer")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Equal Weight Return",
            f"{(performance['equal_weight_cumulative'].iloc[-1] - 1) * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Sentiment Strategy Return",
            f"{(performance['sentiment_weight_cumulative'].iloc[-1] - 1) * 100:.2f}%"
        )


    st.subheader("Portfolio Performance")

    fig = px.line(
        performance,
        x="date",
        y=[
            "equal_weight_cumulative",
            "sentiment_weight_cumulative"
        ],
        labels={
            "value": "Portfolio Value",
            "date": "Date",
            "variable": "Strategy"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader("Stock Weights")

    latest_date = weights["date"].max()

    latest_weights = weights[
        weights["date"] == latest_date
    ].sort_values(
        "weight",
        ascending=False
    )

    fig_weights = px.bar(
        latest_weights,
        x="Stock Name",
        y="weight",
        color="sentiment_score",
        title=f"Latest Stock Allocation — {latest_date}"
    )

    st.plotly_chart(
        fig_weights,
        use_container_width=True
    )


# ============================================================
# MODULE B
# ============================================================

else:

    st.header("⚠️ Strategic Portfolio Stress Test")

    summary = dict(
        zip(
            module_b_summary["Metric"],
            module_b_summary["Value"]
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Event",
        summary["Event Type"]
    )

    col2.metric(
        "Impact Score",
        summary["Impact Score"]
    )

    col3.metric(
        "Portfolio Impact",
        f"{summary['Portfolio Impact (%)']}%"
    )

    st.divider()


    st.subheader(
        "Portfolio Before vs After Stress"
    )

    before = summary[
        "Initial Portfolio Value ($M)"
    ]

    after = summary[
        "Stressed Portfolio Value ($M)"
    ]

    comparison = pd.DataFrame({
        "Scenario": [
            "Before Stress",
            "After Stress"
        ],
        "Portfolio Value ($M)": [
            before,
            after
        ]
    })

    fig = px.bar(
        comparison,
        x="Scenario",
        y="Portfolio Value ($M)",
        title="Portfolio Value Impact"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Impact by Asset Type"
    )

    asset_impact = (
        stress_test
        .groupby("asset_type")["total_impact"]
        .sum()
        .reset_index()
    )

    fig_asset = px.bar(
        asset_impact,
        x="asset_type",
        y="total_impact",
        title="Stress Impact by Asset Type"
    )

    st.plotly_chart(
        fig_asset,
        use_container_width=True
    )