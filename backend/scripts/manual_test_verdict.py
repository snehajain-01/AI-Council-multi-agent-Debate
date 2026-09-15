"""Manual smoke test for the complete pipeline: question -> Council Verdict.

Run with: python -m scripts.manual_test_verdict
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.debate_agent import DebateAgent
from app.agents.debate_manager import DebateManager
from app.agents.judge_agent import JudgeAgent
from app.agents.synthesizer import Synthesizer
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
    synthesizer = Synthesizer(provider=OllamaProvider(model=model, base_url=base_url))

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

    print("\n--- Judging ---")
    judge_result = await manager.run_judging(question, revised, judge)
    for label, verdict in judge_result.verdicts_by_label.items():
        real_name = judge_result.label_map[label]
        print(f"{label} (actually {real_name}): weighted total {verdict.weighted_total:.1f}/100")

    print("\n--- Consensus ---")
    consensus = await manager.run_consensus(question, revised, round2_result, judge_result, judge)
    print(f"Consensus level: {consensus.consensus_level.upper()}")

    print("\n=== FINAL COUNCIL VERDICT ===")
    verdict = await synthesizer.synthesize(question, revised, judge_result, consensus)
    print(f"\nQuestion: {verdict.question}")
    print(f"\nFinal Answer:\n{verdict.final_answer}")
    print(f"\nConsensus: {verdict.consensus_level.upper()}")
    print(f"Consensus explanation: {verdict.consensus_explanation}")
    print(f"Confidence: {verdict.confidence:.1f}/100")
    print(f"Strongest agent: {verdict.strongest_agent} (score {verdict.strongest_agent_score:.1f}/100)")
    if verdict.key_disagreements:
        print("Key disagreements:")
        for d in verdict.key_disagreements:
            print(f"  - {d}")
    print("\nAgent positions:")
    for name, position in verdict.agent_positions.items():
        print(f"  {name}: {position}")


if __name__ == "__main__":
    asyncio.run(main())
