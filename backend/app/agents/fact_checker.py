"""Extracts and (eventually) verifies checkable factual claims from agent positions."""

from app.models.claim import Claim, ClaimVerification, ExtractedClaims
from app.models.debate import RevisedPosition
from app.models.evidence import Evidence
from app.providers.base import BaseLLMProvider

CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You extract checkable factual claims from a piece of analysis. A claim is "
    "checkable if it asserts something about the world that could, in principle, be "
    "confirmed or contradicted by external evidence -- a fact, a statistic, a causal "
    "relationship, or a definition. Exclude pure opinions, value judgments, or "
    "recommendations ('we should...', 'this is important', 'I believe...'). Restate "
    "each claim as a standalone sentence that makes sense without the surrounding "
    "context. Extract at most the 5 most significant checkable claims -- do not pad "
    "the list with minor or redundant restatements of the same point."
)

VERIFICATION_SYSTEM_PROMPT = (
    "You are verifying a factual claim against retrieved evidence. Judge only what "
    "the evidence actually supports -- do not use outside knowledge, and do not fill "
    "gaps with assumptions. Use SUPPORTED if the evidence clearly confirms the claim; "
    "PARTIALLY_SUPPORTED if the evidence confirms part of it or supports it with "
    "caveats the claim omits; CONTRADICTED if the evidence conflicts with the claim; "
    "and UNVERIFIED if the evidence is missing, off-topic, or too thin to judge "
    "either way. UNVERIFIED is a legitimate, expected outcome -- do not stretch weak "
    "or tangential evidence into SUPPORTED just to give a more decisive answer."
)


class FactChecker:
    """Wraps an LLM provider to extract and verify factual claims."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    async def extract_claims(self, position: RevisedPosition) -> list[Claim]:
        """Pull out the most significant checkable claims from a final position."""
        prompt = (
            f"Position: {position.position}\n\n"
            f"Reasoning: {position.reasoning}\n\n"
            "Extract the checkable factual claims from this analysis."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            json_schema=ExtractedClaims.model_json_schema(),
        )
        extracted = ExtractedClaims.model_validate_json(response.content)
        return extracted.claims

    async def verify_claim(self, claim: Claim, evidence: list[Evidence]) -> ClaimVerification:
        """Compare one claim against its retrieved evidence and classify the result."""
        if not evidence:
            return ClaimVerification(
                status="UNVERIFIED",
                reasoning="No evidence was found for this claim.",
            )

        evidence_text = "\n\n".join(
            f"Source: {e.source.title} (reliability: {e.source.reliability})\nExcerpt: {e.excerpt}"
            for e in evidence
        )
        prompt = (
            f"Claim: {claim.text}\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            "Does this evidence support, partially support, contradict, or fail to "
            "address this claim?"
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=VERIFICATION_SYSTEM_PROMPT,
            json_schema=ClaimVerification.model_json_schema(),
        )
        return ClaimVerification.model_validate_json(response.content)
