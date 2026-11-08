"""
Feature engineering for FX prediction.

Design rules:
- Every feature is stationary (returns, ratios, z-scores) — NO raw price levels.
- Every feature is backward-looking (no leakage).
- Macro indicators are processed at their NATIVE frequency, then merge_asof'd
  so the model only sees data that was already published as of each date.
"""
import pandas as pd

FEATURES_BASE = ["ret_1", "ret_5", "ret_20", "vol_20", "ma_ratio", "z_20"]

FEATURES_MACRO = [
    "usd_cpi_yoy", "usd_cpi_yoy_chg",
    "usd_unemployment", "usd_unemployment_chg",
    "usd_fed_funds", "usd_fed_funds_chg",
    "usd_ust_10y", "usd_ust_10y_chg",
]


def load_macro_features(macros_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily, forward-filled macro feature frame from long-form
    EconomicIndicator rows. Each indicator is processed at its native
    frequency, THEN merged. This avoids the "pct_change(12) on daily data"
    bug that wiped out the model previously.
    """
    if macros_df.empty:
        return pd.DataFrame(columns=["rate_date"])

    def _series(indicator: str) -> pd.Series | None:
        sub = macros_df[macros_df["indicator"] == indicator].sort_values("obs_date")
        if sub.empty:
            return None
        # Dedupe: keep the last value per obs_date (handles re-runs that
        # accidentally created duplicate rows before the unique constraint
        # was added).
        return sub.groupby("obs_date")["value"].last()

    levels: dict[str, pd.Series] = {}

    # CPI is an index level (e.g. 300.84) → convert to YoY % at monthly freq
    cpi = _series("CPI_YOY")
    if cpi is not None and len(cpi) > 12:
        levels["usd_cpi_yoy"] = (cpi.pct_change(12, fill_method=None) * 100).dropna()

    # Unemployment at monthly freq
    unemp = _series("UNEMPLOYMENT")
    if unemp is not None:
        levels["usd_unemployment"] = unemp

    # Fed funds & 10Y yield at daily freq
    ff = _series("FED_FUNDS")
    if ff is not None:
        levels["usd_fed_funds"] = ff

    ust = _series("UST_10Y")
    if ust is not None:
        levels["usd_ust_10y"] = ust

    if not levels:
        return pd.DataFrame(columns=["rate_date"])

    # Build a wide frame aligned on the union of all obs dates, then ffill
    out = pd.DataFrame(levels)
    out.index.name = "rate_date"
    out = out.sort_index().ffill().reset_index()

    # Compute changes on the ffill'd daily series (0 most days, spike on release)
    for name in ["usd_cpi_yoy", "usd_unemployment", "usd_fed_funds", "usd_ust_10y"]:
        if name in out.columns:
            out[f"{name}_chg"] = out[name].diff()

    return out


def build_fx_features(rates: pd.DataFrame,
                      macro_df: pd.DataFrame,
                      horizon_days: int) -> pd.DataFrame:
    """
    Build features + targets for one horizon.

    rates:    columns = [rate_date, base, quote, rate]
    macro_df: output of load_macro_features()
    Returns  one row per (base, quote, rate_date) with target_fwd_ret.
    """
    macro_sorted = macro_df.sort_values("rate_date").copy() if not macro_df.empty else None
    if macro_sorted is not None:
        macro_sorted["rate_date"] = pd.to_datetime(macro_sorted["rate_date"])

    out = []
    for (base, quote), g in rates.groupby(["base", "quote"]):
        g = g.sort_values("rate_date").reset_index(drop=True)
        df = pd.DataFrame({"rate_date": g["rate_date"], "spot": g["rate"]})

        # ── Stationary price features ─────────────────────────────────
        df["ret_1"]   = df["spot"].pct_change(1)
        df["ret_5"]   = df["spot"].pct_change(5)
        df["ret_20"]  = df["spot"].pct_change(20)
        df["vol_20"]  = df["ret_1"].rolling(20).std()
        df["ma_ratio"] = df["spot"].rolling(5).mean() / df["spot"].rolling(20).mean()
        df["z_20"]    = (
            (df["spot"] - df["spot"].rolling(20).mean())
            / df["spot"].rolling(20).std()
        )

        # ── Target ────────────────────────────────────────────────────
        df["target_fwd_ret"] = df["spot"].shift(-horizon_days) / df["spot"] - 1
        df["target_spot"]    = df["spot"].shift(-horizon_days)

        # ── Macro merge (past observations only) ──────────────────────
        if macro_sorted is not None and not macro_sorted.empty:
            df["rate_date"] = pd.to_datetime(df["rate_date"])
            df = pd.merge_asof(
                df.sort_values("rate_date"),
                macro_sorted,
                on="rate_date",
                direction="backward",   # ← no future leakage
            )

        df["base"], df["quote"] = base, quote
        out.append(df)

    return pd.concat(out, ignore_index=True)