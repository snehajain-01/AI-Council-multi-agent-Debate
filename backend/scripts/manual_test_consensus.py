"""Manual smoke test for the full pipeline: debate (Rounds 1-4), judging, and consensus.

Run with: python -m scripts.manual_test_consensus
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.debate_agent import DebateAgent
from app.agents.debate_manager import DebateManager
from app.agents.judge_agent import JudgeAgent
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
    judge = JudgeAgent(provider=OllamaProvider(model=model, base_url=base_url))

    question = "Should nuclear energy be expanded?"

    print("--- Round 1: Independent Answers ---")
    positions = await manager.run_round_1(question)
    for name, position in positions.items():
        print(f"{name}: {position.position} (confidence {position.confidence})")

    print("\n--- Round 2: Blind Cross-Critique ---")
    round2_result = await manager.run_round_2(positions)

    print("\n--- Round 3: Counterargument ---")
    counterarguments = await manager.run_round_3(positions, round2_result)

    print("\n--- Round 4: Revision ---")
    revised = await manager.run_round_4(positions, counterarguments)
    for name, rp in revised.items():
        print(f"{name}: {rp.position} (confidence {rp.confidence})")

    print("\n--- Judging ---")
    judge_result = await manager.run_judging(question, revised, judge)
    for label, verdict in judge_result.verdicts_by_label.items():
        real_name = judge_result.label_map[label]
        print(f"{label} (actually {real_name}): weighted total {verdict.weighted_total:.1f}/100")

    print("\n--- Consensus ---")
    consensus = await manager.run_consensus(question, revised, round2_result, judge_result, judge)
    print(f"Consensus level: {consensus.consensus_level.upper()}")
    print(f"Agreement level (from judge): {consensus.agreement.agreement_level}")
    print(f"Shared conclusion: {consensus.agreement.shared_conclusion or '(none)'}")
    if consensus.agreement.key_disagreements:
        print("Key disagreements:")
        for d in consensus.agreement.key_disagreements:
            print(f"  - {d}")
    print(f"Average judge score: {consensus.average_judge_score:.1f}")
    print(f"Judge score spread: {consensus.judge_score_spread:.1f}")
    print(f"Total critique volume (Round 2): {consensus.total_critique_volume}")
    print(f"Explanation: {consensus.explanation}")


if __name__ == "__main__":
    asyncio.run(main())
