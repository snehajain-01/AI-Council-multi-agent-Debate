"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.debates import router as debates_router
from app.db.session import create_all_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    yield


app = FastAPI(title="AI Council", lifespan=lifespan)
app.include_router(debates_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
