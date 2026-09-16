"""Manual smoke test for evidence retrieval.

Reuses the same claims from manual_test_claims.py's example position, so this can
be tested independently before wiring claim extraction and evidence retrieval
together in the full pipeline.

Run with: python -m scripts.manual_test_evidence
"""

import asyncio

from app.models.claim import Claim
from app.services.research_service import ResearchService


async def main() -> None:
    research = ResearchService()

    claims = [
        Claim(
            text="Nuclear power plants produce about 3-4 grams of CO2 equivalent per kilowatt-hour.",
            claim_type="statistical",
        ),
        Claim(text="France generates over 60% of its electricity from nuclear power.", claim_type="factual"),
        Claim(
            text="The Chernobyl disaster in 1986 was caused by a flawed reactor design combined with operator error.",
            claim_type="causal",
        ),
    ]

    for claim in claims:
        print(f"\nClaim: {claim.text}")
        evidence_list = await research.search(claim.text)
        if not evidence_list:
            print("  No evidence found.")
        for evidence in evidence_list:
            print(f"  - [{evidence.source.reliability}] {evidence.source.title} ({evidence.source.url})")
            print(f"      {evidence.excerpt}")


if __name__ == "__main__":
    asyncio.run(main())
