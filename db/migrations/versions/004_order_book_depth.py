"""Add order book depth columns to signals

Revision ID: 004
Revises: 003
Create Date: 2026-05-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("ob_spread", sa.Numeric(6, 4), nullable=True))
    op.add_column("signals", sa.Column("ob_depth_usd", sa.Numeric(12, 2), nullable=True))


def downgrade() -> None:
    op.drop_column("signals", "ob_depth_usd")
    op.drop_column("signals", "ob_spread")
