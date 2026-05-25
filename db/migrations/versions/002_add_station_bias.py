"""Add station_bias table for forecast bias calibration

Revision ID: 002
Revises: 001
Create Date: 2026-05-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "station_bias",
        sa.Column("station", sa.String(10), primary_key=True),
        sa.Column("month", sa.SmallInteger(), primary_key=True),
        sa.Column("bias_f", sa.Numeric(4, 2), nullable=False, server_default="0"),
        sa.Column("rmse_f", sa.Numeric(4, 2), nullable=False, server_default="2.5"),
        sa.Column("sample_count", sa.Integer(), server_default="0"),
        sa.Column("source", sa.String(50)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("station_bias")
