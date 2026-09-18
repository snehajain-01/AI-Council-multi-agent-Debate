"""Manual smoke test for the complete pipeline, including fact verification:
question -> debate (Rounds 1-4) -> judging -> claim verification -> consensus -> verdict.

Run with: python -m scripts.manual_test_full_pipeline
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.debate_agent import DebateAgent
from app.agents.debate_manager import DebateManager
from app.agents.fact_checker import FactChecker
from app.agents.judge_agent import JudgeAgent
from app.agents.synthesizer import Synthesizer
from app.providers.ollama_provider import OllamaProvider
from app.services.research_service import ResearchService

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
    fact_checker = FactChecker(provider=OllamaProvider(model=model, base_url=base_url))
    research = ResearchService()

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

    print("\n--- Fact Verification ---")
    evidence_report = await manager.run_verification(revised, fact_checker, research)
    for record in evidence_report.records:
        print(f"[{record.agent_name}] {record.claim.text}")
        print(f"    -> {record.verification.status} ({len(record.evidence)} source(s))")

    print("\n--- Consensus ---")
    consensus = await manager.run_consensus(question, revised, round2_result, judge_result, judge, evidence_report)
    print(f"Consensus level: {consensus.consensus_level.upper()}")
    print(f"Evidence support ratio: {consensus.evidence_support_ratio:.2f}")
    print(f"Explanation: {consensus.explanation}")

    print("\n=== FINAL COUNCIL VERDICT ===")
    verdict = await synthesizer.synthesize(
        question, manager.agents, revised, judge_result, consensus, evidence_report
    )
    print(f"\nFinal Answer:\n{verdict.final_answer}")
    print(f"\nConsensus: {verdict.consensus_level.upper()}")
    print(f"Confidence: {verdict.confidence:.1f}/100")
    print(f"Claims verified: {verdict.claims_supported}/{verdict.claims_checked} supported")
    print(f"Strongest agent: {verdict.strongest_agent} (score {verdict.strongest_agent_score:.1f}/100)")


if __name__ == "__main__":
    asyncio.run(main())
