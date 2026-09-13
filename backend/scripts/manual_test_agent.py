"""Manual smoke test for DebateAgent. Run with: python -m scripts.manual_test_agent"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.debate_agent import DebateAgent
from app.providers.ollama_provider import OllamaProvider

load_dotenv()


async def main() -> None:
    provider = OllamaProvider(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    agent = DebateAgent(name="Agent A", provider=provider)

    position = await agent.answer("Should nuclear energy be expanded?")

    print(f"Position:    {position.position}")
    print(f"Confidence:  {position.confidence}")
    print(f"Reasoning:   {position.reasoning}")
    print("Key arguments:")
    for arg in position.key_arguments:
        print(f"  - {arg}")
    print("Evidence requirements:")
    for req in position.evidence_requirements:
        print(f"  - {req}")
    print("Weaknesses:")
    for weakness in position.weaknesses:
        print(f"  - {weakness}")


if __name__ == "__main__":
    asyncio.run(main())
