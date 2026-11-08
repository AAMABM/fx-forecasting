import logging
from datetime import date, timedelta
from app.database.session import init_db
from app.collectors.fx_rates import collect_fx
from app.collectors.fred_macro import collect_macro
from app.models.train_fx import run_all
from app.forecasting.predict import predict_all_horizons
from app.verification.verify import verify_due_predictions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    init_db()
    start = date.today() - timedelta(days=365 * 5)
    collect_fx(start, date.today())
    collect_macro(start)
    run_all()
    predict_all_horizons()
    verify_due_predictions()
    logging.info("Pipeline complete.")