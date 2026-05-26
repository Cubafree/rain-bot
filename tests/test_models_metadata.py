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
    """Signal.is_contaminated must exist, be non-nullable, and declare a False default (T11)."""
    mapper = inspect(Signal)
    column_names = [col.key for col in mapper.mapper.column_attrs]
    assert "is_contaminated" in column_names, (
        "Signal.is_contaminated column is missing"
    )

    col = Signal.__table__.c["is_contaminated"]
    assert not col.nullable, "Signal.is_contaminated must be NOT NULL"

    # Column must declare a default (either Python-side or server-side) that evaluates to False.
    # Note: Column(default=False) sets an INSERT default, not a Python __init__ default,
    # so we inspect the column definition rather than instantiating Signal().
    has_default = col.default is not None or col.server_default is not None
    assert has_default, "Signal.is_contaminated must have a default value"

    if col.default is not None:
        default_val = col.default.arg
        assert default_val is False or default_val == 0, (
            f"Expected default False/0, got {default_val!r}"
        )
