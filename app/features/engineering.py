import pandas as pd

FEATURES_BASE = ["ret_1", "ret_5", "ret_20", "vol_20", "ma_ratio", "z_20"]

FEATURES_MACRO = [
    "usd_cpi_yoy", "usd_unemployment", "usd_fed_funds", "usd_ust_10y",
    "oil_chg", "rate_diff", "rate_diff_chg"
]

def load_macro_features(macros_df: pd.DataFrame) -> pd.DataFrame:
    if macros_df.empty:
        return pd.DataFrame(columns=["rate_date"])

    def _series(indicator: str) -> pd.Series | None:
        sub = macros_df[macros_df["indicator"] == indicator].sort_values("obs_date")
        if sub.empty:
            return None
        return sub.groupby("obs_date")["value"].last()

    levels = {}

    # US Specific
    cpi = _series("CPI_YOY")
    if cpi is not None and len(cpi) > 12:
        levels["usd_cpi_yoy"] = (cpi.pct_change(12, fill_method=None) * 100).dropna()
    
    for name, col in [("UNEMPLOYMENT", "usd_unemployment"), 
                      ("FED_FUNDS", "usd_fed_funds"),
                      ("UST_10Y", "usd_ust_10y"),
                      ("OIL_BRENT", "oil_brent")]:
        s = _series(name)
        if s is not None:
            levels[col] = s

    # Global Rates (for differentials)
    for ccy in ["USD", "EUR", "GBP", "JPY", "CNY", "MYR", "SGD"]:
        s = _series(f"RATE_{ccy}")
        if s is not None:
            levels[f"rate_{ccy}"] = s

    if not levels:
        return pd.DataFrame(columns=["rate_date"])

    out = pd.DataFrame(levels)
    out.index.name = "rate_date"
    out = out.sort_index().ffill().reset_index()

    # Compute changes and oil returns
    if "usd_cpi_yoy" in out: out["usd_cpi_yoy_chg"] = out["usd_cpi_yoy"].diff()
    if "usd_unemployment" in out: out["usd_unemployment_chg"] = out["usd_unemployment"].diff()
    if "usd_fed_funds" in out: out["usd_fed_funds_chg"] = out["usd_fed_funds"].diff()
    if "usd_ust_10y" in out: out["usd_ust_10y_chg"] = out["usd_ust_10y"].diff()
    if "oil_brent" in out: out["oil_chg"] = out["oil_brent"].pct_change(5, fill_method=None)

    return out

def build_fx_features(rates: pd.DataFrame, macro_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    macro_sorted = macro_df.sort_values("rate_date").copy() if not macro_df.empty else None
    if macro_sorted is not None:
        macro_sorted["rate_date"] = pd.to_datetime(macro_sorted["rate_date"])

    out = []
    for (base, quote), g in rates.groupby(["base", "quote"]):
        g = g.sort_values("rate_date").reset_index(drop=True)
        df = pd.DataFrame({"rate_date": g["rate_date"], "spot": g["rate"]})

        df["ret_1"]   = df["spot"].pct_change(1)
        df["ret_5"]   = df["spot"].pct_change(5)
        df["ret_20"]  = df["spot"].pct_change(20)
        df["vol_20"]  = df["ret_1"].rolling(20).std()
        df["ma_ratio"] = df["spot"].rolling(5).mean() / df["spot"].rolling(20).mean()
        df["z_20"]    = (df["spot"] - df["spot"].rolling(20).mean()) / df["spot"].rolling(20).std()

        df["target_fwd_ret"] = df["spot"].shift(-horizon_days) / df["spot"] - 1
        df["target_spot"]    = df["spot"].shift(-horizon_days)

        if macro_sorted is not None and not macro_sorted.empty:
            df["rate_date"] = pd.to_datetime(df["rate_date"])
            df = pd.merge_asof(
                df.sort_values("rate_date"), macro_sorted,
                on="rate_date", direction="backward",
            )
            
            # Calculate Interest Rate Differential
            base_col = f"rate_{base}"
            quote_col = f"rate_{quote}"
            if base_col in df.columns and quote_col in df.columns:
                df["rate_diff"] = df[base_col] - df[quote_col]
                df["rate_diff_chg"] = df["rate_diff"].diff()
            else:
                df["rate_diff"] = 0.0
                df["rate_diff_chg"] = 0.0

        df["base"], df["quote"] = base, quote
        out.append(df)

    return pd.concat(out, ignore_index=True)