# 🔦 Beacon — AI-Powered Financial Risk Intelligence Platform

> **Turning unstructured financial information into actionable risk signals.**

Beacon is an AI-powered financial risk intelligence platform built for the **S&P Global × CRISIL Campus Hackathon 2026**.

🔗 **Live Demo:** https://beacon-financial-risk-engine.streamlit.app/

---

## 🚀 What Beacon Does

Beacon converts financial news and social-media text into structured risk signals:

- **Sentiment:** Positive / Neutral / Negative
- **Event Type:** Financial event category
- **Impact Score:** 1–10
- **Risk Level:** Low / Medium / High

These signals power two portfolio intelligence modules:

1. **Tactical Stock Rebalancer** — adjusts stock weights using sentiment.
2. **Strategic Stress Tester** — evaluates portfolio impact of high-risk events.

---

## 🏗️ Architecture

```text
Financial News / Tweets
          ↓
     Risk Engine
          ↓
 ┌────────┼─────────┐
 ↓        ↓         ↓
Sentiment Event   Impact
 ↓        ↓         ↓
 └────────┼─────────┘
          ↓
     Risk Level
       /      \
      ↓        ↓
 Module A    Module B
Rebalancer  Stress Test
```

## 🤖 AI / ML Models

### 1. Sentiment Analysis — FinBERT

**Model:** `ProsusAI/FinBERT`

FinBERT is used for financial-domain sentiment classification.

```text
Positive / Negative / Neutral
```

**Performance:**

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 76.14% | 0.72 |
| **FinBERT** | **87.71%** | **0.87** |

FinBERT was selected as the final sentiment engine.

---

### 2. Event Classification

Events are classified into:

- Earnings / Financial Results
- Macroeconomic
- M&A / Acquisition
- Geopolitical
- Product / Technology
- Other

**Model:**

```text
TF-IDF + Logistic Regression
```

**Performance:**

```text
Accuracy : 72.34%
Macro F1 : 0.72
```

FinBERT embeddings were also tested but achieved only **57.03% accuracy**, so the simpler TF-IDF model was retained.

---

### 3. Impact Score

Impact is represented on a **1–10 scale**.

Historical calibration uses the absolute **3-day market movement**:

| 3-Day Movement | Score |
|---|---:|
| ≤ 0.5% | 1 |
| 0.5–1% | 2 |
| 1–2% | 3 |
| 2–3% | 4 |
| 3–4% | 5 |
| 4–5% | 6 |
| 5–7% | 7 |
| 7–10% | 8 |
| 10–15% | 9 |
| >15% | 10 |

For live text, Beacon uses an **interpretable estimated impact score** based on sentiment, sentiment confidence, event category and event confidence.

---

### 4. Risk Level

```text
Impact < 5   → LOW
Impact 5–7   → MEDIUM
Impact ≥ 8   → HIGH
```

---

## 📈 Module A — Tactical Stock Rebalancer

Daily sentiment from approximately **80K stock tweets covering 25 stocks** is aggregated by stock and date.

```text
Tweets
  ↓
FinBERT Sentiment
  ↓
Daily Stock Sentiment
  ↓
Dynamic Portfolio Weights
  ↓
Performance Analysis
```

Positive sentiment increases allocation, while negative sentiment decreases allocation.

### Backtest

```text
Equal Weight Strategy     : -25.86%
Sentiment Strategy        : -19.10%
Outperformance            : +6.76 percentage points
```

---

## 🏦 Module B — Strategic Portfolio Stress Test

A synthetic wholesale banking portfolio containing:

```text
53 Loans
34 Bonds
13 Derivatives
```

was created with a total value of approximately **$5.48B**.

### Example Trigger

```text
Event Type  = Geopolitical
Impact Score > 7
```

### Stress Scenario

```text
Equity Shock        = -10%
Interest Rate Shock = +2%
```

### Result

```text
Initial Portfolio   : $5,480.52M
Stressed Portfolio  : $5,191.22M
Portfolio Loss      : $289.30M
Impact              : -5.28%
```

---

## 📊 Datasets

Beacon uses public/synthetic datasets including:

- **Financial News:** ~4,800 labeled articles
- **Stock Tweets:** ~80,000 tweets across 25 stocks
- **Stock Prices:** ~6,300 records
- **Polygon News:** ~5,500 financial news articles

The datasets provide financial text, sentiment information, stock/ticker information and historical market movements.

---

## 💡 Key Benefits

- ⚡ Converts large volumes of financial text into structured signals
- 🤖 Uses a financial-domain NLP model for sentiment
- 🏷️ Automatically identifies important financial events
- ⚠️ Produces interpretable risk and impact signals
- 📈 Connects sentiment with tactical portfolio allocation
- 🏦 Connects major events with portfolio stress testing
- 📊 Provides an interactive decision-support dashboard
- 🔍 Uses transparent logic where reliable supervised labels were unavailable

---

## 🧩 Challenges & Solutions

### Dataset Time Mismatch

News and stock-price datasets covered different periods, so direct event-to-market attribution was avoided.

### Missing Event Labels

Event categories were created using domain-specific weak-labeling rules.

### Model Selection

FinBERT performed best for sentiment, while TF-IDF + Logistic Regression performed better for event classification.

### Target Leakage

An initial impact model achieved unrealistically high performance because the target was derived from an input feature. The model was discarded and replaced with transparent impact calibration.

---

## ⚠️ Limitations

- Event labels use weak supervision rather than fully human-annotated data.
- Live Impact Score is an **estimated heuristic**, not a directly trained market-impact model.
- News and market datasets have different time ranges.
- Module A is a simplified historical backtest.
- Module B uses a synthetic portfolio and simplified stress assumptions.
- Beacon is a prototype and **not financial or investment advice**.

---

## 🛠️ Tech Stack

```text
Python
Pandas / NumPy
Scikit-learn
Hugging Face Transformers
FinBERT
Streamlit
Plotly
Joblib
Git / GitHub
Streamlit Community Cloud
```

---

## 📁 Project Structure

```text
beacon-financial-risk-engine/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── event_model.pkl
│   └── event_vectorizer.pkl
│
├── data/
│   └── stock_prices_clean.csv
│
└── outputs/
    ├── risk_engine_predictions.csv
    ├── module_a_rebalancing_weights.csv
    ├── module_a_portfolio_performance.csv
    ├── module_b_summary.csv
    └── module_b_stress_test.csv
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/beacon-financial-risk-engine.git
cd beacon-financial-risk-engine

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

---

## 🔮 Future Scope

- Real-time financial news and social-media APIs
- Streaming risk detection
- Fine-tuned event classification
- Direct text-to-market-impact modeling
- Real-time risk alerts
- Sector/geographic risk propagation
- Advanced portfolio optimization
- SHAP-based explainability
- Continuous model retraining

---

## 🏆 Key Results

| Component | Result |
|---|---:|
| FinBERT Sentiment Accuracy | **87.71%** |
| Event Classification Accuracy | **72.34%** |
| Module A Outperformance | **+6.76 pp** |
| Module B Stress Impact | **-5.28%** |

---

## 👩‍💻 Built For

**S&P Global × CRISIL Campus Hackathon 2026**

> **Beacon — Turning financial information into actionable risk intelligence.**

