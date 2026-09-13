"""Manual smoke test for OllamaProvider. Run with: python scripts/manual_test_provider.py"""

import asyncio
import os

from dotenv import load_dotenv

from app.providers.ollama_provider import OllamaProvider

load_dotenv()


async def main() -> None:
    provider = OllamaProvider(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    response = await provider.generate(
        prompt="What is 2 + 2? Answer in one short sentence.",
    )

    print(f"Provider: {response.provider}")
    print(f"Model:    {response.model}")
    print(f"Latency:  {response.latency_ms:.0f} ms")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
