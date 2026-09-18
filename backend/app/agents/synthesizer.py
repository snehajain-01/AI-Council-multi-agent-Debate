"""Produces the final, user-facing Council Verdict from everything the debate produced."""

from app.agents.debate_agent import DebateAgent
from app.core.verification import SUPPORTING_STATUSES
from app.models.claim import EvidenceReport
from app.models.debate import (
    AgentSummary,
    ConsensusResult,
    CouncilVerdict,
    JudgeResult,
    RevisedPosition,
    SynthesizedAnswer,
)
from app.providers.base import BaseLLMProvider

# A code-level safeguard, not just a prompt instruction: testing showed the model
# can ignore the "explicitly state disagreement" instruction and produce a bare,
# confident-sounding answer even when consensus_level is no_consensus. Since "the
# system must be able to say agents disagree" is a non-negotiable design principle,
# not just a preference, this is enforced deterministically rather than left to the
# model's discretion.
CONSENSUS_DISCLAIMERS = {
    "no_consensus": "The analysts did not reach a reliable consensus on this question. ",
    "partial_consensus": "The analysts reached only partial agreement on this question. ",
}

SYNTHESIZER_SYSTEM_PROMPT = (
    "You are writing the final answer on behalf of a panel of independent analysts "
    "who just finished debating a question. You are given their consensus assessment "
    "and must accurately represent it -- do not second-guess it. If consensus is "
    "strong, give a single confident answer. If only partial, give the shared "
    "conclusion but clearly flag the specific disagreements. If there is no "
    "consensus, do not manufacture a unified answer -- explicitly state that the "
    "analysts disagree and summarize the different positions fairly. If many of "
    "their claims were not supported by external evidence, say so plainly rather "
    "than presenting the answer as more reliable than it is."
)


class Synthesizer:
    """Wraps an LLM provider to compose the final Council Verdict."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    async def synthesize(
        self,
        question: str,
        agents: list[DebateAgent],
        revised_positions: dict[str, RevisedPosition],
        judge_result: JudgeResult,
        consensus_result: ConsensusResult,
        evidence_report: EvidenceReport | None = None,
    ) -> CouncilVerdict:
        strongest_label, strongest_verdict = max(
            judge_result.verdicts_by_label.items(), key=lambda item: item[1].weighted_total
        )
        strongest_agent = judge_result.label_map[strongest_label]

        judge_score_by_name = {
            name: judge_result.verdicts_by_label[label].weighted_total
            for label, name in judge_result.label_map.items()
        }

        positions_text = "\n\n".join(
            f"{name} (judge score {judge_result.verdicts_by_label[label].weighted_total:.1f}/100): "
            f"{revised_positions[name].position}\nReasoning: {revised_positions[name].reasoning}"
            for label, name in judge_result.label_map.items()
        )
        agreement = consensus_result.agreement

        records = evidence_report.records if evidence_report else []
        claims_checked = len(records)
        claims_supported = sum(1 for r in records if r.verification.status in SUPPORTING_STATUSES)
        evidence_line = (
            f"Evidence check: {claims_supported}/{claims_checked} claims supported by external evidence\n"
            if claims_checked
            else "Evidence check: no checkable claims were verified\n"
        )

        prompt = (
            f"Question: {question}\n\n"
            f"Consensus assessment: {consensus_result.consensus_level} ({consensus_result.explanation})\n"
            f"Shared conclusion (if any): {agreement.shared_conclusion or 'none'}\n"
            f"Key disagreements: {'; '.join(agreement.key_disagreements) or 'none'}\n"
            f"{evidence_line}\n"
            f"The analysts' final positions:\n{positions_text}\n\n"
            "Write the final answer to the original question."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            json_schema=SynthesizedAnswer.model_json_schema(),
        )
        synthesized = SynthesizedAnswer.model_validate_json(response.content)

        disclaimer = CONSENSUS_DISCLAIMERS.get(consensus_result.consensus_level, "")
        final_answer = f"{disclaimer}{synthesized.final_answer}" if disclaimer else synthesized.final_answer

        agent_summaries = [
            AgentSummary(
                name=agent.name,
                model=agent.provider.model_name,
                position=revised_positions[agent.name].position,
                reasoning=revised_positions[agent.name].reasoning,
                key_arguments=revised_positions[agent.name].key_arguments,
                weaknesses=revised_positions[agent.name].weaknesses,
                changes_from_original=revised_positions[agent.name].changes_from_original,
                confidence=revised_positions[agent.name].confidence,
                judge_score=judge_score_by_name[agent.name],
            )
            for agent in agents
        ]

        return CouncilVerdict(
            question=question,
            final_answer=final_answer,
            consensus_level=consensus_result.consensus_level,
            consensus_explanation=consensus_result.explanation,
            confidence=consensus_result.average_judge_score,
            strongest_agent=strongest_agent,
            strongest_agent_score=strongest_verdict.weighted_total,
            key_disagreements=agreement.key_disagreements,
            agents=agent_summaries,
            claims_supported=claims_supported,
            claims_checked=claims_checked,
        )
