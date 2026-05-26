"""Unit tests verifying Signal model metadata for T10/T11 columns."""
from sqlalchemy import inspect

from db.models import Signal


def test_signal_has_forecast_source_column():
    """Signal model must have a forecast_source attribute (T10)."""
    mapper = inspect(Signal)
    column_names = [col.key for col in mapper.mapper.column_attrs]
    assert "forecast_source" in column_names, (
        "Signal.forecast_source column is missing"
    )


def test_signal_is_contaminated_defaults_false():
    """Signal.is_contaminated must exist and default to False (T11)."""
    mapper = inspect(Signal)
    column_names = [col.key for col in mapper.mapper.column_attrs]
    assert "is_contaminated" in column_names, (
        "Signal.is_contaminated column is missing"
    )

    # Verify the Python-side default is False (not NULL)
    col = Signal.__table__.c["is_contaminated"]
    assert col.default is not None or col.server_default is not None or not col.nullable, (
        "Signal.is_contaminated has no default value"
    )
    instance = Signal()
    assert instance.is_contaminated is False, (
        f"Expected Signal().is_contaminated to be False, got {instance.is_contaminated!r}"
    )
