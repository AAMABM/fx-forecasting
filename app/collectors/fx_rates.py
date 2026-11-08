"""
FX rate collector.

Uses Frankfurter (https://frankfurter.app) — a free, no-key API that exposes
European Central Bank reference rates for ~30 currencies.

Note: BND is NOT published by the ECB. Because the Brunei dollar is pegged 1:1
to the Singapore dollar by a currency board, we proxy USD/BND with USD/SGD and
relabel it. This is a documented, standard practice; the peg has held since 1967.
"""
import logging
from datetime import date
import requests
from sqlalchemy.dialects.postgresql import insert
from app.database.models import ExchangeRate
from app.database.session import get_session

log = logging.getLogger(__name__)

PAIRS = [
    ("USD", "BND"), ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"),
    ("USD", "CNY"), ("USD", "MYR"), ("USD", "SGD"),
]

BASE_URL = "https://api.frankfurter.app"


def fetch_historical(base: str, quote: str, start: date, end: date) -> list[dict]:
    """Fetch daily reference rates from Frankfurter for one pair."""
    if base == quote:
        return []
    url = f"{BASE_URL}/{start.isoformat()}..{end.isoformat()}"
    params = {"from": base, "to": quote}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rates = r.json().get("rates", {})
    return [
        {
            "base": base,
            "quote": quote,
            "rate_date": date.fromisoformat(d),
            "rate": float(v[quote]),
        }
        for d, v in rates.items()
        if quote in v
    ]


def collect_fx(start: date, end: date) -> None:
    rows: list[dict] = []
    for base, quote in PAIRS:
        try:
            # BND fallback: fetch SGD (which Frankfurter supports), then relabel.
            fetch_quote = "SGD" if quote == "BND" else quote
            fetched = fetch_historical(base, fetch_quote, start, end)
            if quote == "BND":
                for row in fetched:
                    row["quote"] = "BND"
            rows.extend(fetched)
        except Exception as e:
            log.warning("FX fetch failed %s/%s: %s", base, quote, e)

    if not rows:
        log.warning("No FX rows fetched — nothing inserted")
        return

    stmt = insert(ExchangeRate).values(rows).on_conflict_do_nothing()
    with get_session() as s:
        s.execute(stmt)
    log.info("Inserted %d FX rows", len(rows))