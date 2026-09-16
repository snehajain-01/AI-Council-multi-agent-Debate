"""Data shapes for claim extraction and fact verification."""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.evidence import Evidence

ClaimType = Literal["factual", "statistical", "causal", "definitional"]


class Claim(BaseModel):
    """A discrete, checkable factual assertion extracted from an agent's position."""

    text: str = Field(description="The claim restated as a standalone, self-contained factual assertion")
    claim_type: ClaimType = Field(description="What kind of claim this is")


class ExtractedClaims(BaseModel):
    """The claims extracted from one position, capped to keep extraction reliable and
    to prioritize the claims most worth verifying rather than every minor clause."""

    claims: list[Claim] = Field(
        description="The most significant checkable factual claims in the position, at most 5. "
        "Exclude pure opinions, value judgments, or recommendations."
    )


VerificationStatus = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


class ClaimVerification(BaseModel):
    """The result of comparing one claim against its retrieved evidence.

    UNVERIFIED must be a real, reachable outcome -- when evidence is thin,
    off-topic, or absent, this should say so rather than guessing SUPPORTED.

    `reasoning` is declared before `status` deliberately: with JSON-schema-constrained
    generation, fields are produced in declaration order, so putting the classification
    first forces the model to commit to a status before it has "thought through" the
    evidence that should justify it. Testing showed this produces answers where the
    reasoning describes clearly supporting or contradicting evidence while status is
    left as UNVERIFIED regardless -- reasoning-first lets the conclusion follow from
    the analysis instead of the other way around.
    """

    reasoning: str = Field(
        description="Compare the claim to the evidence first, before deciding a status: what "
        "does the evidence actually say, and does it confirm, partially confirm, conflict with, "
        "or fail to address the claim?"
    )
    status: VerificationStatus = Field(description="The status that follows from the reasoning above")


class ClaimVerificationRecord(BaseModel):
    """One claim, which agent made it, the evidence retrieved, and its verification result."""

    agent_name: str
    claim: Claim
    evidence: list[Evidence]
    verification: ClaimVerification


class EvidenceReport(BaseModel):
    """All claim verifications gathered across every agent's final position."""

    records: list[ClaimVerificationRecord]
