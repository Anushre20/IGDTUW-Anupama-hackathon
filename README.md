# Beacon — AI-Powered Financial Risk Intelligence Platform - S&P Global & CRISIL Campus Hackathon

**Candidate Name:** Anupama
**College Email ID:** anupama010btaiml24@igdtuw.ac.in
**College / Campus:** Indira Gandhi Delhi Technical University for Women (IGDTUW), Delhi  
**Demo Video Link:** [YouTube Unlisted Link]  
**Slide Deck Link (if hosted externally):** https://drive.google.com/file/d/1ObpvTI2Vs60QJ_lmg12Vi_rLN0fLrzUW/view?usp=sharing

## 1. Project Overview / Problem Statement & Approach

Financial news and social media generate large volumes of unstructured information that can influence markets. Manually identifying important events and assessing their potential risk is slow and difficult to scale.

**Beacon** converts financial text into structured signals: sentiment, event category, estimated impact score (1–10), and risk level (Low/Medium/High). FinBERT powers sentiment analysis, while TF-IDF + Logistic Regression classifies events. The signals feed two modules: a tactical stock rebalancer and a strategic portfolio stress tester.

## 2. Architecture & Tech Stack

**Data flow:**

```text
Financial News + Stock Tweets
              ↓
           Risk Engine
     ┌────────┼─────────┐
     ↓        ↓         ↓
  FinBERT  TF-IDF +   Impact
 Sentiment  Logistic  Estimation
             Regression
     └────────┼─────────┘
              ↓
       Structured Risk Signal
              ↓
       ┌──────┴──────┐
       ↓             ↓
 Module A         Module B
Rebalancing     Stress Testing
```

**Tech stack:** Python, Pandas, NumPy, Scikit-learn, Hugging Face Transformers, FinBERT, Streamlit, Plotly, Joblib, Git/GitHub.

## 3. Dataset Used

- Financial news: ~4,800 labeled articles.
- Stock tweets: ~80,000 tweets across 25 stocks.
- Stock prices: ~6,300 records.
- Polygon news: ~5,500 financial news articles.
- Module B uses a synthetic 100-asset portfolio.

**Assumptions:** Event labels were created using domain-specific weak-labeling rules. Historical impact calibration uses absolute three-day stock movement. Since news and price datasets cover different periods, direct causal attribution is not claimed. Live impact scores are heuristic estimates, not outputs of a trained text-to-impact model.

## 4. Quickstart & Installation

**Runtime:** Python 3.11 recommended; tested locally on macOS. Deployed on Streamlit Community Cloud.

```bash
git clone <your-repo-url>
cd beacon-financial-risk-engine

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `venv\\Scripts\\activate`.

**Live demo:** https://beacon-financial-risk-engine.streamlit.app/

## 5. Key Results & Domain Impact

- **FinBERT sentiment:** 87.71% accuracy, 0.87 Macro F1.
- **Event classification:** 72.34% accuracy, 0.72 Macro F1.
- **Module A:** Sentiment strategy returned -19.10% versus -25.86% for equal weighting, a difference of +6.76 percentage points in the historical backtest.
- **Module B:** Simulated portfolio value decreased from $5,480.52M to $5,191.22M under the example stress scenario, a loss of $289.30M (-5.28%).

Beacon demonstrates how unstructured financial information can be transformed into structured risk signals and connected to portfolio analysis. It supports faster event exploration, sentiment-based allocation experiments and hypothetical stress testing.

**Limitations:** Event labels use weak supervision; Module A is a simplified backtest; Module B uses a synthetic portfolio and simplified shocks. Beacon is a hackathon prototype, not financial advice.
