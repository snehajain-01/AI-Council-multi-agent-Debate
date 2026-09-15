"""Configurable scoring weights and weighted-score computation for the Judge.

The Judge model only produces raw per-criterion scores; combining them into a
single weighted total happens here, in code, rather than being asked of the
model. Language models are unreliable at multi-step weighted arithmetic even
when their individual judgments are sound, so this keeps that failure mode
out of the pipeline entirely.
"""

from app.models.debate import JUDGE_SCORE_FIELDS, JudgeScores

DEFAULT_WEIGHTS: dict[str, float] = {
    "factual_correctness": 0.30,
    "logical_reasoning": 0.25,
    "evidence_support": 0.20,
    "relevance": 0.10,
    "completeness": 0.10,
    "clarity": 0.05,
}


def compute_weighted_score(scores: JudgeScores, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
    """Combine per-criterion scores (0-100 each) into a single weighted total out of 100."""
    total_weight = sum(weights[field] for field in JUDGE_SCORE_FIELDS)
    weighted_sum = sum(getattr(scores, field) * weights[field] for field in JUDGE_SCORE_FIELDS)
    return weighted_sum / total_weight
