"""Extracts and (eventually) verifies checkable factual claims from agent positions."""

from app.models.claim import Claim, ExtractedClaims
from app.models.debate import RevisedPosition
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
