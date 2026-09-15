"""Produces the final, user-facing Council Verdict from everything the debate produced."""

from app.models.debate import ConsensusResult, CouncilVerdict, JudgeResult, RevisedPosition, SynthesizedAnswer
from app.providers.base import BaseLLMProvider

SYNTHESIZER_SYSTEM_PROMPT = (
    "You are writing the final answer on behalf of a panel of independent analysts "
    "who just finished debating a question. You are given their consensus assessment "
    "and must accurately represent it -- do not second-guess it. If consensus is "
    "strong, give a single confident answer. If only partial, give the shared "
    "conclusion but clearly flag the specific disagreements. If there is no "
    "consensus, do not manufacture a unified answer -- explicitly state that the "
    "analysts disagree and summarize the different positions fairly."
)


class Synthesizer:
    """Wraps an LLM provider to compose the final Council Verdict."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    async def synthesize(
        self,
        question: str,
        revised_positions: dict[str, RevisedPosition],
        judge_result: JudgeResult,
        consensus_result: ConsensusResult,
    ) -> CouncilVerdict:
        strongest_label, strongest_verdict = max(
            judge_result.verdicts_by_label.items(), key=lambda item: item[1].weighted_total
        )
        strongest_agent = judge_result.label_map[strongest_label]

        positions_text = "\n\n".join(
            f"{name} (judge score {judge_result.verdicts_by_label[label].weighted_total:.1f}/100): "
            f"{revised_positions[name].position}\nReasoning: {revised_positions[name].reasoning}"
            for label, name in judge_result.label_map.items()
        )
        agreement = consensus_result.agreement

        prompt = (
            f"Question: {question}\n\n"
            f"Consensus assessment: {consensus_result.consensus_level} ({consensus_result.explanation})\n"
            f"Shared conclusion (if any): {agreement.shared_conclusion or 'none'}\n"
            f"Key disagreements: {'; '.join(agreement.key_disagreements) or 'none'}\n\n"
            f"The analysts' final positions:\n{positions_text}\n\n"
            "Write the final answer to the original question."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            json_schema=SynthesizedAnswer.model_json_schema(),
        )
        synthesized = SynthesizedAnswer.model_validate_json(response.content)

        return CouncilVerdict(
            question=question,
            final_answer=synthesized.final_answer,
            consensus_level=consensus_result.consensus_level,
            consensus_explanation=consensus_result.explanation,
            confidence=consensus_result.average_judge_score,
            strongest_agent=strongest_agent,
            strongest_agent_score=strongest_verdict.weighted_total,
            key_disagreements=agreement.key_disagreements,
            agent_positions={name: position.position for name, position in revised_positions.items()},
        )
