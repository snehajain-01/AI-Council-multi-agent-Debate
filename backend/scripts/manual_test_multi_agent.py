"""Manual smoke test for running multiple independent agents concurrently.

Run with: python -m scripts.manual_test_multi_agent
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.debate_agent import DebateAgent
from app.providers.ollama_provider import OllamaProvider

load_dotenv()

# All three agents currently share one model (only llama3.2:1b is pulled).
# Swap in different models per agent later (e.g. "phi3:mini", "gemma2:2b")
# for genuine reasoning diversity, not just independent execution.
AGENT_MODELS = {
    "Agent A": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    "Agent B": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    "Agent C": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
}


async def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    agents = [
        DebateAgent(name=name, provider=OllamaProvider(model=model, base_url=base_url))
        for name, model in AGENT_MODELS.items()
    ]

    question = "Should nuclear energy be expanded?"

    # Each agent answers independently and concurrently -- none sees another's output.
    positions = await asyncio.gather(*(agent.answer(question) for agent in agents))

    for agent, position in zip(agents, positions):
        print(f"=== {agent.name} ({agent.provider.model}) ===")
        print(f"Position:   {position.position}")
        print(f"Confidence: {position.confidence}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
