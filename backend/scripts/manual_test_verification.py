"""Manual smoke test for the full fact-verification pipeline:
claim extraction -> evidence retrieval -> verification.

Includes one deliberately false claim (wrong year for Chernobyl) to confirm
CONTRADICTED is actually reachable, not just SUPPORTED by default.

Run with: python -m scripts.manual_test_verification
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.fact_checker import FactChecker
from app.models.claim import Claim
from app.models.debate import RevisedPosition
from app.providers.ollama_provider import OllamaProvider
from app.services.research_service import ResearchService

load_dotenv()


async def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    fact_checker = FactChecker(provider=OllamaProvider(model=model, base_url=base_url))
    research = ResearchService()

    position = RevisedPosition(
        position="Nuclear energy should be expanded",
        reasoning=(
            "Nuclear power plants produce about 3-4 grams of CO2 equivalent per "
            "kilowatt-hour, far lower than coal plants. France generates over 60% "
            "of its electricity from nuclear power. I believe this makes nuclear a "
            "morally responsible choice for reducing emissions. The Chernobyl "
            "disaster in 1986 was caused by a flawed reactor design combined with "
            "operator error."
        ),
        key_arguments=["low emissions", "proven track record in France"],
        evidence_requirements=["lifecycle emissions data", "safety statistics"],
        confidence=0.8,
        weaknesses=["waste disposal"],
        changes_from_original=[],
    )

    claims = await fact_checker.extract_claims(position)

    # Deliberately false, to confirm CONTRADICTED is reachable rather than the
    # system defaulting to SUPPORTED whenever evidence is found.
    claims.append(Claim(text="The Chernobyl disaster occurred in 1990.", claim_type="factual"))

    for claim in claims:
        print(f"\nClaim: {claim.text}")
        evidence = await research.search(claim.text)
        verification = await fact_checker.verify_claim(claim, evidence)
        print(f"  Status: {verification.status}")
        print(f"  Reasoning: {verification.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
