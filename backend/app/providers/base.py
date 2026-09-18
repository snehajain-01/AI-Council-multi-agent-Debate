"""Provider-independent interface every LLM backend must implement."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """A normalized response from any LLM provider, regardless of backend."""

    content: str
    model: str
    provider: str
    latency_ms: float


class BaseLLMProvider(ABC):
    """Abstract interface for an LLM backend.

    Concrete providers (Ollama, OpenRouter, Gemini, ...) implement this
    so that agent code never depends on a specific provider's API shape.
    """

    model_name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        """Send a prompt to the model and return a normalized response.

        If json_schema is provided, the provider must constrain the model's
        output to valid JSON matching that schema.
        """
        raise NotImplementedError
