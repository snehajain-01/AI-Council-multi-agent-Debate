"""SQLAlchemy ORM models -- the database schema, distinct from the Pydantic
domain models in app/models/, which describe in-memory data shapes."""

from datetime import datetime

from sqlalchemy import DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class DebateRecord(Base):
    """One completed debate: the question asked and its full Council Verdict.

    The verdict is stored as a single JSONB blob rather than normalized into
    many tables (debates/rounds/scores/claims/...). Postgres's JSONB type is
    genuinely queryable, so this gives real persistence and history without
    designing a multi-table schema before we know how the frontend actually
    needs to query it. Can be normalized later if specific queries demand it.
    """

    __tablename__ = "debates"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    verdict: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
