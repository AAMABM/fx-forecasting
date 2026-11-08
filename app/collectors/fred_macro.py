import logging
from datetime import date
import requests
from sqlalchemy.dialects.postgresql import insert
from app.config import settings
from app.database.models import EconomicIndicator
from app.database.session import get_session

log = logging.getLogger(__name__)

SERIES = {
    # US Macro
    "CPIAUCSL": ("USA", "CPI_YOY"),
    "UNRATE": ("USA", "UNEMPLOYMENT"),
    "FEDFUNDS": ("USA", "FED_FUNDS"),
    "DGS10": ("USA", "UST_10Y"),
    
    # 3-Month Interbank Rates (for Interest Rate Differentials)
    "IR3TIB01USM156N": ("USA", "RATE_USD"),
    "IR3TIB01EZM156N": ("EUR", "RATE_EUR"),
    "IR3TIB01GBM156N": ("GBR", "RATE_GBP"),
    "IR3TIB01JPM156N": ("JPN", "RATE_JPY"),
    "IR3TIB01CNM156N": ("CHN", "RATE_CNY"),
 

    # Commodities (Brent Crude Oil)
    "DCOILBRENTEU": ("USA", "OIL_BRENT"),
}

def fetch_series(series_id: str, start: date) -> list[tuple[date, float]]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": settings.fred_api_key,
              "file_type": "json", "observation_start": start.isoformat()}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return [(date.fromisoformat(o["date"]), float(o["value"]))
            for o in r.json()["observations"] if o["value"] != "."]

def collect_macro(start: date) -> None:
    rows = []
    for sid, (cc, name) in SERIES.items():
        try:
            for d, v in fetch_series(sid, start):
                rows.append({"country_code": cc, "indicator": name,
                             "obs_date": d, "value": v, "source": "FRED"})
        except Exception as e:
            log.warning("FRED %s failed: %s", sid, e)
    if rows:
        with get_session() as s:
            s.execute(insert(EconomicIndicator).values(rows).on_conflict_do_nothing())
    log.info("Inserted %d macro rows", len(rows))