"""
Prediction module.

For each (pair, horizon), load the trained model, build today's features
from the latest available rates, predict the forward return, convert to an
absolute price, and store the result in the `predictions` table.

Idempotent: re-running on the same day won't create duplicate rows.
"""
import logging
from datetime import timedelta
import joblib
import pandas as pd
from sqlalchemy import select
from app.config import settings
from app.database.models import ExchangeRate, Prediction
from app.database.session import get_session
from app.features.engineering import build_fx_features
from app.models.train_fx import HORIZONS, load_macros

log = logging.getLogger(__name__)

BASE_FEATURES_REQUIRED = ["ret_1", "ret_5", "ret_20", "vol_20", "ma_ratio", "z_20"]

# Same clip bound used during training evaluation
PRED_CLIP = 0.15


def _load_rates() -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(select(ExchangeRate)).scalars().all()
    return pd.DataFrame(
        [{"rate_date": r.rate_date, "base": r.base,
          "quote": r.quote, "rate": r.rate} for r in rows]
    )


def _predict_for_horizon(rates: pd.DataFrame, macros: pd.DataFrame,
                        horizon: int) -> int:
    if rates.empty:
        return 0

    feat = build_fx_features(rates, macros, horizon)
    stored = 0

    for (base, quote), g in feat.groupby(["base", "quote"]):
        g = g.dropna(subset=BASE_FEATURES_REQUIRED)
        if g.empty:
            continue
        latest = g.sort_values("rate_date").iloc[-1]

        model_path = settings.model_dir / f"xgb_{base}_{quote}_h{horizon}.joblib"
        if not model_path.exists():
            log.warning("Missing model: %s", model_path.name)
            continue

        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]

        missing = [f for f in features
                   if f not in latest.index or pd.isna(latest[f])]
        if missing:
            log.warning("Missing features for %s: %s", model_path.name, missing)
            continue

        x = latest[features].to_numpy(dtype=float).reshape(1, -1)
        pred_ret = float(model.predict(x)[0])
        # Clip to ±15% — same bound used during training evaluation
        pred_ret = max(-PRED_CLIP, min(PRED_CLIP, pred_ret))
        spot = float(latest["spot"])
        pred_price = spot * (1.0 + pred_ret)

        pred_date = latest["rate_date"]
        if hasattr(pred_date, "date"):
            pred_date = pred_date.date()
        target_dt = pred_date + timedelta(days=horizon)

        with get_session() as s:
            existing = s.execute(
                select(Prediction).where(
                    Prediction.target == f"{base}/{quote}",
                    Prediction.horizon_days == horizon,
                    Prediction.prediction_date == pred_date,
                )
            ).scalar_one_or_none()
            if existing is not None:
                continue
            s.add(Prediction(
                model_name=f"xgboost_h{horizon}",
                target=f"{base}/{quote}",
                horizon_days=horizon,
                prediction_date=pred_date,
                target_date=target_dt,
                predicted_value=pred_price,
            ))
        stored += 1

    return stored


def predict_all_horizons() -> None:
    rates = _load_rates()
    if rates.empty:
        log.warning("No FX data — cannot generate predictions")
        return
    macros = load_macros()
    total = 0
    for h in HORIZONS:
        n = _predict_for_horizon(rates, macros, h)
        log.info("Stored %d predictions for horizon %d", n, h)
        total += n
    log.info("Stored %d predictions total", total)