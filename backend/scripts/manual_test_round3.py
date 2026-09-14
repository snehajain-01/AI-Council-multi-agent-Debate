"""Manual smoke test for Round 1 + Round 2 + Round 3 (counterargument).

Run with: python -m scripts.manual_test_round3
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.debate_agent import DebateAgent
from app.agents.debate_manager import DebateManager
from app.providers.ollama_provider import OllamaProvider

load_dotenv()


async def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    agents = [
        DebateAgent(name=name, provider=OllamaProvider(model=model, base_url=base_url))
        for name in ("Agent A", "Agent B", "Agent C")
    ]
    manager = DebateManager(agents)

    question = "Should nuclear energy be expanded?"

    print("--- Round 1: Independent Answers ---")
    positions = await manager.run_round_1(question)
    for name, position in positions.items():
        print(f"{name}: {position.position} (confidence {position.confidence})")

    print("\n--- Round 2: Blind Cross-Critique ---")
    round2_result = await manager.run_round_2(positions)
    for critiquing_agent, critique_set in round2_result.critiques_by_agent.items():
        label_map = round2_result.label_maps[critiquing_agent]
        for critique in critique_set.critiques:
            real_target = label_map.get(critique.target_label, "UNKNOWN")
            print(f"{critiquing_agent} -> {real_target}: {len(critique.points)} issue(s) raised")

    print("\n--- Round 3: Counterargument ---")
    counterarguments = await manager.run_round_3(positions, round2_result)
    for agent_name, counter in counterarguments.items():
        print(f"\n{agent_name} responds to its critics:")
        if not counter.points:
            print("  (no critiques were received)")
        for point in counter.points:
            print(f"  [{point.critique_category}] {point.response_type}: {point.explanation}")


if __name__ == "__main__":
    asyncio.run(main())
