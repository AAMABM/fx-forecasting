"""
SHAP explainability module.

Provides global feature importance (beeswarm) and per-prediction
explanations (waterfall) for the trained XGBoost FX models.
"""
import logging

import joblib
import numpy as np
import pandas as pd
import shap

from app.config import settings
from app.features.engineering import build_fx_features
from app.models.train_fx import load_rates, load_macros, PRED_CLIP

log = logging.getLogger(__name__)


def load_model_bundle(base: str, quote: str, horizon: int) -> dict | None:
    path = settings.model_dir / f"xgb_{base}_{quote}_h{horizon}.joblib"
    if not path.exists():
        log.warning("Model not found: %s", path.name)
        return None
    return joblib.load(path)


def build_feature_frame(base: str, quote: str, horizon: int) -> pd.DataFrame:
    rates = load_rates()
    macros = load_macros()
    if rates.empty:
        return pd.DataFrame()

    feat = build_fx_features(rates, macros, horizon)
    sub = feat[(feat["base"] == base) & (feat["quote"] == quote)].copy()
    if sub.empty:
        return pd.DataFrame()

    bundle = load_model_bundle(base, quote, horizon)
    if bundle is None:
        return pd.DataFrame()

    features = bundle["features"]
    sub = sub.dropna(subset=features + ["target_fwd_ret"])
    return sub


def global_importance(base: str, quote: str, horizon: int):
    """SHAP values for ALL training rows. Use .plots.beeswarm / .plots.bar."""
    sub = build_feature_frame(base, quote, horizon)
    if sub.empty:
        return None

    bundle = load_model_bundle(base, quote, horizon)
    model = bundle["model"]
    features = bundle["features"]

    X_df = sub[features]  # DataFrame → feature names auto-attach
    explainer = shap.TreeExplainer(model)
    return explainer(X_df)


def explain_single_prediction(base: str, quote: str, horizon: int):
    """SHAP values for the LATEST row only. Use .plots.waterfall(sv[0])."""
    sub = build_feature_frame(base, quote, horizon)
    if sub.empty:
        return None, None

    bundle = load_model_bundle(base, quote, horizon)
    model = bundle["model"]
    features = bundle["features"]

    latest = sub.sort_values("rate_date").iloc[-1]
    X_df = pd.DataFrame(
        [latest[features].to_numpy(dtype=float)],
        columns=features,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)

    pred_raw = float(model.predict(X_df)[0])
    pred_clip = float(np.clip(pred_raw, -PRED_CLIP, PRED_CLIP))

    meta = {
        "pair": f"{base}/{quote}",
        "horizon": horizon,
        "as_of": latest["rate_date"],
        "spot": float(latest["spot"]),
        "predicted_return_raw": pred_raw,
        "predicted_return_clipped": pred_clip,
        "features": features,
    }
    return shap_values, meta