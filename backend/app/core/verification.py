"""Deterministic computation over claim verification results.

Turns raw per-claim verification records into a single evidence-support ratio the
Consensus Engine can use, without asking an LLM to do this arithmetic.
"""

from app.models.claim import ClaimVerificationRecord

SUPPORTING_STATUSES = {"SUPPORTED", "PARTIALLY_SUPPORTED"}


def compute_evidence_support_ratio(records: list[ClaimVerificationRecord]) -> float:
    """Fraction of checked claims that were SUPPORTED or PARTIALLY_SUPPORTED.

    Returns 1.0 (neutral -- not "confirmed", just "not contradicted") when there are
    no claims to check, so an absence of checkable claims never looks like evidence
    against the debate.
    """
    if not records:
        return 1.0
    supporting = sum(1 for record in records if record.verification.status in SUPPORTING_STATUSES)
    return supporting / len(records)
