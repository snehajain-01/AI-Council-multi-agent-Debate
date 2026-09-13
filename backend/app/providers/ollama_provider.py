"""LLM provider backed by a local Ollama server."""

import time

import httpx

from app.providers.base import BaseLLMProvider, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """Calls a local Ollama instance's REST API to generate responses."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if json_schema:
            payload["format"] = json_schema

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        return LLMResponse(
            content=data["response"],
            model=self.model,
            provider="ollama",
            latency_ms=latency_ms,
        )
