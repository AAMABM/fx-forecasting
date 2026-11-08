# FX & Macroeconomic Forecasting Engine

An end-to-end machine learning pipeline for forecasting exchange rates (7-day, 30-day, 90-day horizons) using macroeconomic indicators, with a live Streamlit dashboard and SHAP explainability.

## 🏗️ Architecture

1. **Data Ingestion:** Automated daily collection of FX rates (via Frankfurter/ECB) and macro data (via FRED).
2. **Feature Engineering:** Leak-free merging of macro features (`merge_asof`), stationary price transformations (returns, volatility, Z-scores).
3. **Modeling:** XGBoost regression with Time-Series Cross-Validation. Compares against a Random Walk baseline.
4. **Verification:** Stores predictions in PostgreSQL and automatically verifies accuracy when target dates mature.
5. **Explainability:** SHAP waterfalls and beeswarm plots to explain *why* a prediction was made.
6. **Dashboard:** 4-page Streamlit app (Overview, Forecasts, Accuracy, Explain).

## 🧠 Honest Results & Interpretation

Predicting FX is notoriously difficult due to market efficiency and low signal-to-noise ratios. This project prioritizes **honest evaluation** over misleading metrics:
- Models are evaluated against a **Random Walk baseline** (predicting 0 return).
- Current out-of-sample direction accuracy ranges from **49% to 53%** (coin flip is 50%). 
- **SHAP analysis reveals the model correctly uses macroeconomic signals** (interest rate changes, CPI, yield spreads) alongside technical volatility, rather than just trend-chasing.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Docker Desktop (for PostgreSQL)

### 2. Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/fx-forecasting.git
cd fx-forecasting

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env and add your FRED API key (free from fred.stlouisfed.org)