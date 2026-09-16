"""Async SQLAlchemy engine and session factory."""

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Loaded here, not left to the importing script, so this module works correctly
# regardless of when/whether the caller has already loaded .env itself.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_council")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


async def create_all_tables() -> None:
    """Create any tables that don't exist yet.

    Fine for early development; a proper migration tool (Alembic) should
    replace this once the schema stabilizes and needs to change without
    dropping data.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    async with async_session_factory() as session:
        yield session
