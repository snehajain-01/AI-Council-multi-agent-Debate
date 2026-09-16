"""Coordinates debate agents through the rounds of the debate protocol."""

import asyncio
import random

from app.agents.debate_agent import DebateAgent
from app.agents.fact_checker import FactChecker
from app.agents.judge_agent import JudgeAgent
from app.core.consensus import determine_consensus
from app.core.verification import compute_evidence_support_ratio
from app.models.claim import Claim, ClaimVerificationRecord, EvidenceReport
from app.models.debate import (
    AgentPosition,
    ConsensusResult,
    Counterargument,
    CritiquePoint,
    CritiqueSet,
    JudgeResult,
    RevisedPosition,
    Round2Result,
)
from app.services.research_service import ResearchService


class DebateManager:
    """Runs a group of DebateAgents through the structured debate rounds."""

    def __init__(self, agents: list[DebateAgent]):
        self.agents = agents

    async def run_round_1(self, question: str) -> dict[str, AgentPosition]:
        """Each agent answers independently and concurrently."""
        positions = await asyncio.gather(*(agent.answer(question) for agent in self.agents))
        return {agent.name: position for agent, position in zip(self.agents, positions)}

    async def run_round_2(self, positions: dict[str, AgentPosition]) -> Round2Result:
        """Each agent critiques the others' anonymized, randomly-relabeled positions."""
        label_maps: dict[str, dict[str, str]] = {}
        critique_tasks = []

        for agent in self.agents:
            other_names = [name for name in positions if name != agent.name]
            shuffled_names = random.sample(other_names, len(other_names))
            label_map = {f"Response {i + 1}": name for i, name in enumerate(shuffled_names)}
            label_maps[agent.name] = label_map

            anonymized_positions = {label: positions[name] for label, name in label_map.items()}
            critique_tasks.append(agent.critique(anonymized_positions))

        critique_sets: list[CritiqueSet] = await asyncio.gather(*critique_tasks)
        critiques_by_agent = {agent.name: cs for agent, cs in zip(self.agents, critique_sets)}

        return Round2Result(critiques_by_agent=critiques_by_agent, label_maps=label_maps)

    async def run_round_3(
        self, positions: dict[str, AgentPosition], round2_result: Round2Result
    ) -> dict[str, Counterargument]:
        """Each agent responds to all critiques made against its own (real) position."""
        points_by_target: dict[str, list[CritiquePoint]] = {agent.name: [] for agent in self.agents}

        for critiquing_agent_name, critique_set in round2_result.critiques_by_agent.items():
            label_map = round2_result.label_maps[critiquing_agent_name]
            for critique in critique_set.critiques:
                target_name = label_map.get(critique.target_label)
                if target_name is None:
                    # Critiquing agent used a label that doesn't resolve to a real
                    # target (a small-model slip) -- skip rather than misattribute.
                    continue
                points_by_target[target_name].extend(critique.points)

        counter_tasks = [
            agent.respond_to_critiques(positions[agent.name], points_by_target[agent.name])
            for agent in self.agents
        ]
        counterarguments = await asyncio.gather(*counter_tasks)
        return {agent.name: ca for agent, ca in zip(self.agents, counterarguments)}

    async def run_round_4(
        self,
        positions: dict[str, AgentPosition],
        counterarguments: dict[str, Counterargument],
    ) -> dict[str, RevisedPosition]:
        """Each agent produces a final position incorporating criticism it accepted."""
        revised = await asyncio.gather(
            *(agent.revise(positions[agent.name], counterarguments[agent.name]) for agent in self.agents)
        )
        return {agent.name: rp for agent, rp in zip(self.agents, revised)}

    async def run_judging(
        self,
        question: str,
        revised_positions: dict[str, RevisedPosition],
        judge: JudgeAgent,
    ) -> JudgeResult:
        """An independent judge scores each agent's final position, anonymized with a
        fresh random relabeling -- a separate blind-evaluation moment from Round 2's."""
        agent_names = list(revised_positions.keys())
        shuffled_names = random.sample(agent_names, len(agent_names))
        label_map = {f"Response {i + 1}": name for i, name in enumerate(shuffled_names)}

        verdicts = await asyncio.gather(
            *(judge.score(question, label, revised_positions[name]) for label, name in label_map.items())
        )
        verdicts_by_label = dict(zip(label_map.keys(), verdicts))

        return JudgeResult(verdicts_by_label=verdicts_by_label, label_map=label_map)

    async def run_verification(
        self,
        revised_positions: dict[str, RevisedPosition],
        fact_checker: FactChecker,
        research_service: ResearchService,
    ) -> EvidenceReport:
        """Extract, retrieve evidence for, and verify each agent's claims.

        Runs fully concurrently: all agents' extractions run together, and within
        each agent, all of its claims are searched and verified together too.
        """

        async def verify_one_claim(agent_name: str, claim: Claim) -> ClaimVerificationRecord:
            evidence = await research_service.search(claim.text)
            verification = await fact_checker.verify_claim(claim, evidence)
            return ClaimVerificationRecord(
                agent_name=agent_name, claim=claim, evidence=evidence, verification=verification
            )

        async def verify_agent(agent_name: str, position: RevisedPosition) -> list[ClaimVerificationRecord]:
            claims = await fact_checker.extract_claims(position)
            return await asyncio.gather(*(verify_one_claim(agent_name, claim) for claim in claims))

        nested_records = await asyncio.gather(
            *(verify_agent(name, position) for name, position in revised_positions.items())
        )
        all_records = [record for records in nested_records for record in records]
        return EvidenceReport(records=all_records)

    async def run_consensus(
        self,
        question: str,
        revised_positions: dict[str, RevisedPosition],
        round2_result: Round2Result,
        judge_result: JudgeResult,
        judge: JudgeAgent,
        evidence_report: EvidenceReport | None = None,
    ) -> ConsensusResult:
        """Combine agreement classification, judge scores, debate intensity, and
        (if provided) evidence support into a final consensus level. Reuses
        judge_result's label map so the agreement assessment sees the same
        anonymized view the judge scored."""
        labeled_positions = {
            label: revised_positions[name] for label, name in judge_result.label_map.items()
        }
        agreement = await judge.assess_agreement(question, labeled_positions)

        total_critique_volume = sum(
            len(critique.points)
            for critique_set in round2_result.critiques_by_agent.values()
            for critique in critique_set.critiques
        )
        evidence_support_ratio = (
            compute_evidence_support_ratio(evidence_report.records) if evidence_report else 1.0
        )

        return determine_consensus(
            agreement=agreement,
            judge_verdicts=list(judge_result.verdicts_by_label.values()),
            total_critique_volume=total_critique_volume,
            evidence_support_ratio=evidence_support_ratio,
        )
