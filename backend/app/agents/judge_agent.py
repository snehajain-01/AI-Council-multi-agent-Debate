"""An independent judge that scores each agent's final position against a weighted rubric."""

from app.core.scoring import DEFAULT_WEIGHTS, compute_weighted_score
from app.models.debate import JudgeScores, JudgeVerdict, RevisedPosition
from app.providers.base import BaseLLMProvider

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge scoring one analyst's final position on a "
    "question, after a full round of debate. You do not know the analyst's "
    "identity and must not guess at it. Score strictly and independently on "
    "each criterion from 0 to 100 -- a position should not receive a high "
    "score on one criterion just because it is strong on another. Be "
    "skeptical of confident-sounding claims that lack real support."
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
