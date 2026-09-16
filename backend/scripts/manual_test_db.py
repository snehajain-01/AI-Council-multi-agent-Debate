"""Manual smoke test for database connectivity and persistence.

Run with: python -m scripts.manual_test_db
"""

import asyncio

from sqlalchemy import select

from app.db.models import DebateRecord
from app.db.session import async_session_factory, create_all_tables


async def main() -> None:
    await create_all_tables()
    print("Tables created (or already existed).")

    async with async_session_factory() as session:
        record = DebateRecord(
            question="Should nuclear energy be expanded?",
            verdict={"final_answer": "Test verdict", "consensus_level": "strong_consensus"},
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        print(f"Inserted debate record with id={record.id}, created_at={record.created_at}")

    async with async_session_factory() as session:
        result = await session.execute(select(DebateRecord).order_by(DebateRecord.id.desc()).limit(1))
        latest = result.scalar_one()
        print(f"Read back: id={latest.id}, question={latest.question!r}, verdict={latest.verdict}")


if __name__ == "__main__":
    asyncio.run(main())
