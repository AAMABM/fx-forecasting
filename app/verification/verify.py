import logging
from datetime import date
from sqlalchemy import select
from app.database.models import Prediction, ActualResult, ExchangeRate
from app.database.session import get_session

log = logging.getLogger(__name__)

def verify_due_predictions(today: date | None = None) -> int:
    today = today or date.today()
    verified = 0
    with get_session() as s:
        due = s.execute(
            select(Prediction).where(
                Prediction.target_date <= today,
                ~Prediction.id.in_(select(ActualResult.prediction_id))
            )
        ).scalars().all()

        for p in due:
            base, quote = p.target.split("/")
            actual = s.execute(
                select(ExchangeRate.rate).where(
                    ExchangeRate.base == base,
                    ExchangeRate.quote == quote,
                    ExchangeRate.rate_date <= p.target_date,
                ).order_by(ExchangeRate.rate_date.desc()).limit(1)
            ).scalar_one_or_none()

            if actual is None:
                continue
            err_pct = (actual - p.predicted_value) / p.predicted_value * 100
            s.add(ActualResult(
                prediction_id=p.id, actual_value=float(actual),
                error_pct=float(err_pct),
                direction_correct=bool((actual - p.predicted_value) *
                                       (p.predicted_value - 1.0) >= 0)
            ))
            verified += 1
    log.info("Verified %d predictions", verified)
    return verified