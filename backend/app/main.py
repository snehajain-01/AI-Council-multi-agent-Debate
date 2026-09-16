"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="AI Council")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
