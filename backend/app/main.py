"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.debates import router as debates_router
from app.db.session import create_all_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    yield


app = FastAPI(title="AI Council", lifespan=lifespan)

# The frontend dev server runs on a different origin (port 5173 vs 8000),
# so the browser blocks requests to this API unless it's explicitly allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(debates_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
