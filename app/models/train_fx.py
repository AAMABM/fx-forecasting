"""
Training module for FX models.

Compares every model against a **random-walk baseline** (predict 0 return)
and reports direction accuracy alongside R². This is the only honest way
to evaluate an FX model — R² alone is meaningless when the naive prediction
is "price doesn't change."
"""
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sqlalchemy import select
from app.config import settings
from app.database.models import ExchangeRate, EconomicIndicator
from app.database.session import get_session
from app.features.engineering import (
    build_fx_features, load_macro_features, FEATURES_BASE, FEATURES_MACRO,
)

log = logging.getLogger(__name__)
HORIZONS = [7, 30, 90]

# Prediction clipping — prevents XGBoost from extrapolating absurd returns
PRED_CLIP = 0.15   # ±15% maximum predicted forward return


def load_rates() -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(select(ExchangeRate)).scalars().all()
    return pd.DataFrame([{"rate_date": r.rate_date, "base": r.base,
                          "quote": r.quote, "rate": r.rate} for r in rows])


def load_macros() -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(select(EconomicIndicator)).scalars().all()
    df = pd.DataFrame([{"obs_date": r.obs_date, "indicator": r.indicator,
                        "value": r.value} for r in rows])
    return load_macro_features(df)


def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of times the model's sign matches actual sign.
    Ignore days where actual return is effectively zero."""
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def _make_model() -> XGBRegressor:
    """Small, regularized model — avoids the overfitting that plagues
    tree models on noisy FX returns."""
    return XGBRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
    )


def train_for_horizon(rates, macros, horizon) -> dict:
    feat = build_fx_features(rates, macros, horizon)
    available = [c for c in FEATURES_BASE + FEATURES_MACRO if c in feat.columns]
    feat = feat.dropna(subset=available + ["target_fwd_ret"])
    feat = feat.sort_values("rate_date")

    if feat.empty:
        log.warning("No feature rows for horizon %d", horizon)
        return {}

    metrics = {}
    for (base, quote), g in feat.groupby(["base", "quote"]):
        X = g[available].values
        y = g["target_fwd_ret"].values
        if len(X) < 200:
            log.warning("Skipping %s/%s h%d — only %d rows",
                        base, quote, horizon, len(X))
            continue

        mae_m, mae_rw, r2_m, dir_m = [], [], [], []
        tscv = TimeSeriesSplit(n_splits=5)
        for tr, te in tscv.split(X):
            m = _make_model()
            m.fit(X[tr], y[tr])
            pred = np.clip(m.predict(X[te]), -PRED_CLIP, PRED_CLIP)
            baseline = np.zeros_like(y[te])   # random-walk: predict 0 return

            mae_m.append(mean_absolute_error(y[te], pred))
            mae_rw.append(mean_absolute_error(y[te], baseline))
            r2_m.append(r2_score(y[te], pred))
            dir_m.append(_direction_accuracy(y[te], pred))

        # Final model on all rows
        final = _make_model()
        final.fit(X, y)
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        path = settings.model_dir / f"xgb_{base}_{quote}_h{horizon}.joblib"
        joblib.dump({"model": final, "features": available,
                     "horizon": horizon, "pair": (base, quote)}, path)

        key = f"{base}/{quote}_h{horizon}"
        metrics[key] = {
            "mae_model": float(np.mean(mae_m)),
            "mae_rw":    float(np.mean(mae_rw)),
            "r2_model":  float(np.mean(r2_m)),
            "dir_acc":   float(np.mean(dir_m)),
            "beats_rw":  float(np.mean(mae_m)) < float(np.mean(mae_rw)),
            "n":         int(len(X)),
        }
        log.info(
            "Trained %s  MAE=%.5f (RW=%.5f, %s)  R2=%.3f  DirAcc=%.1f%%  n=%d",
            key,
            metrics[key]["mae_model"], metrics[key]["mae_rw"],
            "BEATS" if metrics[key]["beats_rw"] else "loses",
            metrics[key]["r2_model"], metrics[key]["dir_acc"] * 100,
            metrics[key]["n"],
        )
    return metrics


def run_all() -> None:
    rates = load_rates()
    if rates.empty:
        log.warning("No rate data — run collectors first")
        return
    macros = load_macros()
    log.info("Loaded %d macro feature rows", len(macros))
    for h in HORIZONS:
        train_for_horizon(rates, macros, h)