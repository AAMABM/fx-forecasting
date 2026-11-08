import logging
from datetime import date
import requests
from sqlalchemy.dialects.postgresql import insert
from app.config import settings
from app.database.models import EconomicIndicator
from app.database.session import get_session

log = logging.getLogger(__name__)

# FRED series → (country_code, indicator_name)
SERIES = {
    "CPIAUCSL": ("USA", "CPI_YOY"),
    "UNRATE":   ("USA", "UNEMPLOYMENT"),
    "DFF":      ("USA", "FED_FUNDS"),
    "DGS10":    ("USA", "UST_10Y"),
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