"""Manual smoke test for the full debate protocol (Rounds 1-4) plus judging.

Run with: python -m scripts.manual_test_judge
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
        s = verdict.scores
        print(f"\n{label} (actually {real_name}): weighted total {verdict.weighted_total:.1f}/100")
        print(
            f"  factual_correctness={s.factual_correctness:.0f} "
            f"logical_reasoning={s.logical_reasoning:.0f} "
            f"evidence_support={s.evidence_support:.0f} "
            f"relevance={s.relevance:.0f} "
            f"completeness={s.completeness:.0f} "
            f"clarity={s.clarity:.0f}"
        )
        print(f"  justification: {s.justification}")


if __name__ == "__main__":
    asyncio.run(main())
