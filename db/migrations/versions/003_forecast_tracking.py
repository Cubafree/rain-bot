"""Add forecast_source and is_contaminated to signals

Revision ID: 003
Revises: 002
Create Date: 2026-05-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("forecast_source", sa.String(30), nullable=True))
    op.add_column(
        "signals",
        sa.Column(
            "is_contaminated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index("idx_signals_forecast_source", "signals", ["forecast_source"])

    # Backfill: mark signals whose forecast used raw ERA5 actuals as contaminated
    op.execute(
        """
        UPDATE signals
        SET is_contaminated = TRUE
        WHERE forecast_data->>'data_source' = 'era5_fallback_raw'
        """
    )


def downgrade() -> None:
    op.drop_index("idx_signals_forecast_source", table_name="signals")
    op.drop_column("signals", "is_contaminated")
    op.drop_column("signals", "forecast_source")
