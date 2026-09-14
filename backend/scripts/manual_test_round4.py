"""Manual smoke test for the full Round 1-4 debate protocol.

Run with: python -m scripts.manual_test_round4
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
    issue_counts: dict[str, int] = {name: 0 for name in positions}
    for critiquing_agent, critique_set in round2_result.critiques_by_agent.items():
        label_map = round2_result.label_maps[critiquing_agent]
        for critique in critique_set.critiques:
            real_target = label_map.get(critique.target_label)
            if real_target:
                issue_counts[real_target] += len(critique.points)
    for name, count in issue_counts.items():
        print(f"{name} received {count} issue(s) total")

    print("\n--- Round 3: Counterargument ---")
    counterarguments = await manager.run_round_3(positions, round2_result)
    for name, counter in counterarguments.items():
        print(f"{name}: responded to {len(counter.points)} criticism(s)")

    print("\n--- Round 4: Revision ---")
    revised = await manager.run_round_4(positions, counterarguments)
    for name, rp in revised.items():
        print(f"\n{name}:")
        print(f"  Original position: {positions[name].position}")
        print(f"  Revised position:  {rp.position}")
        print(f"  Confidence: {positions[name].confidence} -> {rp.confidence}")
        if rp.changes_from_original:
            print("  Changes made:")
            for change in rp.changes_from_original:
                print(f"    - {change}")
        else:
            print("  Changes made: (none)")


if __name__ == "__main__":
    asyncio.run(main())
