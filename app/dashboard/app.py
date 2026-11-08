import sys
from pathlib import Path

# Ensure project root is on sys.path so `from app.* import ...` works
# whether launched via `streamlit run app/dashboard/app.py` or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import select
from app.database.models import Prediction, ActualResult, ExchangeRate
from app.database.session import get_session

st.set_page_config(page_title="FX & Risk", layout="wide")
page = st.sidebar.radio("Page", ["Overview", "Forecasts", "Accuracy", "Explain"])


@st.cache_data(ttl=300)
def load_rates():
    with get_session() as s:
        rows = s.execute(select(ExchangeRate)).scalars().all()
    return pd.DataFrame([{"date": r.rate_date, "pair": f"{r.base}/{r.quote}",
                          "rate": r.rate} for r in rows])


if page == "Overview":
    st.title("📊 FX Overview")
    df = load_rates()
    if df.empty:
        st.warning("No data yet — run the collectors first.")
    else:
        pair = st.selectbox("Pair", sorted(df["pair"].unique()))
        sub = df[df["pair"] == pair].sort_values("date")
        st.plotly_chart(px.line(sub, x="date", y="rate", title=pair),
                        use_container_width=True)

elif page == "Forecasts":
    st.title("🔮 Forecasts")
    with get_session() as s:
        preds = s.execute(select(Prediction).order_by(
            Prediction.created_at.desc()).limit(200)).scalars().all()
    if not preds:
        st.info("No predictions stored yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "pair": p.target, "horizon": p.horizon_days,
            "made": p.prediction_date, "target": p.target_date,
            "predicted": round(p.predicted_value, 6),
        } for p in preds]), use_container_width=True)

elif page == "Accuracy":
    st.title("🎯 Accuracy")
    with get_session() as s:
        results = s.execute(select(ActualResult)).scalars().all()
    if not results:
        st.info("No verified predictions yet.")
    else:
        r = pd.DataFrame([{"error_pct": x.error_pct,
                           "direction_correct": x.direction_correct}
                          for x in results])
        c1, c2, c3 = st.columns(3)
        c1.metric("Verified", len(r))
        c2.metric("Mean |error %|", f"{r.error_pct.abs().mean():.2f}%")
        c3.metric("Direction accuracy", f"{r.direction_correct.mean()*100:.1f}%")

elif page == "Explain":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from app.forecasting.explain import (
        global_importance, explain_single_prediction,
    )
    from app.models.train_fx import HORIZONS

    st.title("🧠 Why the model made this prediction")
    st.caption(
        "SHAP decomposes each prediction into feature contributions. "
        "Red pushes the forecast higher, blue pushes it lower."
    )

    col1, col2 = st.columns(2)
    pair = col1.selectbox(
        "Currency pair",
        ["USD/SGD", "EUR/USD", "GBP/USD", "USD/JPY",
         "USD/CNY", "USD/MYR", "USD/BND"],
    )
    horizon = col2.selectbox("Horizon (days)", HORIZONS)
    base, quote = pair.split("/")

    tab_single, tab_global = st.tabs(
        ["Single prediction (waterfall)", "Global importance (beeswarm)"]
    )

    with tab_single:
        sv, meta = explain_single_prediction(base, quote, horizon)
        if sv is None:
            st.warning(
                f"No data for {pair} h{horizon}. "
                f"Ensure the model exists in data/models/."
            )
        else:
            st.subheader(f"Latest prediction — {meta['as_of'].date()}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Spot", f"{meta['spot']:.4f}")
            c2.metric("Predicted return",
                      f"{meta['predicted_return_clipped']*100:+.2f}%")
            c3.metric("Predicted price",
                      f"{meta['spot'] * (1 + meta['predicted_return_clipped']):.4f}")

            plt.figure()
            shap.plots.waterfall(sv[0], max_display=12, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Interpretation: each bar's length is that feature's "
                "contribution (in return units) to this specific forecast."
            )

    with tab_global:
        sv_global = global_importance(base, quote, horizon)
        if sv_global is None:
            st.warning(f"No training data for {pair} h{horizon}.")
        else:
            st.subheader(f"Feature importance across {len(sv_global.values)} rows")
            plt.figure()
            shap.plots.beeswarm(sv_global, max_display=12, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Each dot = one historical day. x-axis = SHAP contribution. "
                "Color = feature value (red high, blue low)."
            )