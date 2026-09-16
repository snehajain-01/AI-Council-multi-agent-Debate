"""Combines agreement, judge scores, and debate intensity into a final consensus level.

This is deterministic, ordinary Python -- not an LLM call -- because the actual
classification of "do these agents agree" already happened via the Judge's
AgreementAssessment. This module's job is just to combine that with judge score
spread into one of three outcomes, and it must never round genuine disagreement
up to consensus: "no_consensus" is a legitimate, expected result, not a failure
of the system.
"""

from app.models.debate import AgreementAssessment, ConsensusResult, JudgeVerdict

# A judge score spread below this (out of 100) is treated as "close enough" that
# score-based quality differences don't undermine an otherwise-agreeing debate.
STRONG_CONSENSUS_MAX_SPREAD = 15.0
PARTIAL_CONSENSUS_MAX_SPREAD = 25.0

# Evidence-support thresholds. These only ever downgrade a consensus level that
# agreement/scores already produced -- weak evidentiary support must be able to
# undercut apparent agreement (per "Consensus != Correctness"), but strong evidence
# alone should never manufacture a consensus that didn't already exist.
NO_CONSENSUS_EVIDENCE_FLOOR = 0.5
STRONG_CONSENSUS_EVIDENCE_FLOOR = 0.8


def determine_consensus(
    agreement: AgreementAssessment,
    judge_verdicts: list[JudgeVerdict],
    total_critique_volume: int,
    evidence_support_ratio: float = 1.0,
) -> ConsensusResult:
    scores = [verdict.weighted_total for verdict in judge_verdicts]
    average_score = sum(scores) / len(scores)
    score_spread = max(scores) - min(scores)

    if agreement.agreement_level == "full_agreement" and score_spread <= STRONG_CONSENSUS_MAX_SPREAD:
        consensus_level = "strong_consensus"
        explanation = (
            "All agents reached the same substantive conclusion, and the judge scored "
            "their final positions closely enough that the agreement is not just superficial."
        )
    elif (
        agreement.agreement_level in ("full_agreement", "majority_agreement")
        and score_spread <= PARTIAL_CONSENSUS_MAX_SPREAD
    ):
        consensus_level = "partial_consensus"
        explanation = (
            "Agents agree on the main conclusion but diverge on specific points, or the "
            "judge found a meaningful quality gap between their final positions."
        )
    else:
        consensus_level = "no_consensus"
        explanation = (
            "Agents substantially disagree, or judge scores vary too widely to treat "
            "their positions as a single reliable answer."
        )

    if evidence_support_ratio < NO_CONSENSUS_EVIDENCE_FLOOR and consensus_level != "no_consensus":
        consensus_level = "no_consensus"
        explanation = (
            "Agents may have agreed, but most of their checkable claims were not "
            "supported by external evidence, so the result cannot be treated as reliable."
        )
    elif evidence_support_ratio < STRONG_CONSENSUS_EVIDENCE_FLOOR and consensus_level == "strong_consensus":
        consensus_level = "partial_consensus"
        explanation = (
            "Agents reached strong agreement, but some of their claims were not fully "
            "supported by external evidence, so this is reported as partial rather than strong consensus."
        )

    return ConsensusResult(
        consensus_level=consensus_level,
        agreement=agreement,
        average_judge_score=average_score,
        judge_score_spread=score_spread,
        total_critique_volume=total_critique_volume,
        evidence_support_ratio=evidence_support_ratio,
        explanation=explanation,
    )
