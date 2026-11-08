from datetime import date, datetime
from sqlalchemy import (
    String, Date, DateTime, Float, Integer, ForeignKey, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): ...

class Country(Base):
    __tablename__ = "countries"
    id: Mapped[int] = mapped_column(primary_key=True)
    iso_code: Mapped[str] = mapped_column(String(3), unique=True)
    name: Mapped[str] = mapped_column(String(100))

class ExchangeRate(Base):
    __tablename__ = "exchange_rates"
    __table_args__ = (UniqueConstraint("base", "quote", "rate_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    base: Mapped[str] = mapped_column(String(3))
    quote: Mapped[str] = mapped_column(String(3))
    rate_date: Mapped[date] = mapped_column(Date)
    rate: Mapped[float] = mapped_column(Float)

class EconomicIndicator(Base):
    __tablename__ = "economic_indicators"
    __table_args__ = (
        UniqueConstraint("country_code", "indicator", "obs_date"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    country_code: Mapped[str] = mapped_column(String(3), index=True)
    indicator: Mapped[str] = mapped_column(String(64), index=True)  # e.g. "CPI_YOY"
    obs_date: Mapped[date] = mapped_column(Date)
    value: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))

class CommodityPrice(Base):
    __tablename__ = "commodity_prices"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)  # BRENT, WTI, GOLD...
    price_date: Mapped[date] = mapped_column(Date)
    price: Mapped[float] = mapped_column(Float)

class GeopoliticalEvent(Base):
    __tablename__ = "geopolitical_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    country_code: Mapped[str] = mapped_column(String(3), index=True)
    event_date: Mapped[date] = mapped_column(Date)
    event_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    model_name: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(64))       # e.g. "USD/SGD"
    horizon_days: Mapped[int] = mapped_column(Integer)
    prediction_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    predicted_value: Mapped[float] = mapped_column(Float)
    confidence_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra: Mapped[str | None] = mapped_column(String(1024), nullable=True)

class ActualResult(Base):
    __tablename__ = "actual_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    actual_value: Mapped[float] = mapped_column(Float)
    error_pct: Mapped[float] = mapped_column(Float)
    direction_correct: Mapped[bool] = mapped_column()
    verified_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    prediction: Mapped[Prediction] = relationship()

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(64))
    horizon_days: Mapped[int] = mapped_column(Integer)
    metric: Mapped[str] = mapped_column(String(32))       # mae, rmse, mape, r2, dir_acc
    value: Mapped[float] = mapped_column(Float)
    evaluated_on: Mapped[date] = mapped_column(Date)