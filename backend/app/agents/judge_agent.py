"""An independent judge that scores each agent's final position against a weighted rubric."""

from app.core.scoring import DEFAULT_WEIGHTS, compute_weighted_score
from app.models.debate import AgreementAssessment, JudgeScores, JudgeVerdict, RevisedPosition
from app.providers.base import BaseLLMProvider

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge scoring one analyst's final position on a "
    "question, after a full round of debate. You do not know the analyst's "
    "identity and must not guess at it. Score strictly and independently on "
    "each criterion from 0 to 100 -- a position should not receive a high "
    "score on one criterion just because it is strong on another. Be "
    "skeptical of confident-sounding claims that lack real support."
)

AGREEMENT_SYSTEM_PROMPT = (
    "You are assessing whether several independent analysts, after a full round "
    "of debate, actually agree with each other in substance -- not whether they "
    "merely sound similar or both use hedging language. You do not know their "
    "identities. Classify their agreement honestly: 'full_agreement' only if "
    "they reach the same substantive conclusion for compatible reasons, "
    "'majority_agreement' if most but not all agree, 'split' if there is a "
    "genuine, roughly even divide, and 'full_disagreement' if they reach "
    "incompatible conclusions. Do not force agreement that is not really there."
)


class JudgeAgent:
    """Wraps an LLM provider to score debate positions against a weighted rubric."""

    def __init__(self, provider: BaseLLMProvider, weights: dict[str, float] | None = None):
        self.provider = provider
        self.weights = weights or DEFAULT_WEIGHTS

    async def score(self, question: str, label: str, position: RevisedPosition) -> JudgeVerdict:
        """Score one anonymized position. One call per position: batching several
        positions' scores into a single call is exactly the kind of task where a small
        model would silently under-fill output (as seen with arrays in Round 2/3), and
        here that would be a silent scoring error rather than an obvious crash.
        """
        prompt = (
            f"Question: {question}\n\n"
            f"{label}'s final position: {position.position}\n"
            f"Reasoning: {position.reasoning}\n"
            f"Key arguments: {'; '.join(position.key_arguments)}\n"
            f"Acknowledged weaknesses: {'; '.join(position.weaknesses)}\n"
            f"Stated confidence: {position.confidence}\n\n"
            "Score this position from 0 to 100 on each criterion: factual correctness, "
            "logical reasoning, evidence/support, relevance, completeness, and clarity."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=JUDGE_SYSTEM_PROMPT,
            json_schema=JudgeScores.model_json_schema(),
        )
        scores = JudgeScores.model_validate_json(response.content)
        weighted_total = compute_weighted_score(scores, self.weights)
        return JudgeVerdict(target_label=label, scores=scores, weighted_total=weighted_total)

    async def assess_agreement(
        self, question: str, labeled_positions: dict[str, RevisedPosition]
    ) -> AgreementAssessment:
        """Classify whether the anonymized final positions actually agree in substance."""
        sections = [
            f"{label}: {position.position}\nReasoning: {position.reasoning}"
            for label, position in labeled_positions.items()
        ]
        prompt = (
            f"Question: {question}\n\n"
            "Here are the final positions from several independent analysts:\n\n"
            + "\n\n".join(sections)
            + "\n\nClassify their overall agreement."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=AGREEMENT_SYSTEM_PROMPT,
            json_schema=AgreementAssessment.model_json_schema(),
        )
        return AgreementAssessment.model_validate_json(response.content)
