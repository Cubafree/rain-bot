"""Disable S6 Contrarian Hype strategy

Revision ID: 005
Revises: 004
Create Date: 2026-05-29

S6 went 0 wins / 8 losses (-$40) in paper trading — the single worst strategy.
Its thesis ("fade favorites at high prices") is the wrong side of the
favorite-longshot bias, so it is structurally negative-EV. Deactivate it until
the strategy is redesigned.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE strategies SET is_active = false WHERE code = 'S6'")


def downgrade() -> None:
    op.execute("UPDATE strategies SET is_active = true WHERE code = 'S6'")
