"""Manual smoke test for claim extraction.

Uses a hand-crafted position (rather than a full 4-round debate) so this can be
tested quickly on its own before wiring it into the full pipeline.

Run with: python -m scripts.manual_test_claims
"""

import asyncio
import os

from dotenv import load_dotenv

from app.agents.fact_checker import FactChecker
from app.models.debate import RevisedPosition
from app.providers.ollama_provider import OllamaProvider

load_dotenv()


async def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    fact_checker = FactChecker(provider=OllamaProvider(model=model, base_url=base_url))

    # Deliberately mixes checkable facts with one opinion sentence, to see whether
    # extraction correctly separates them.
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
    print(f"Extracted {len(claims)} claim(s):\n")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. [{claim.claim_type}] {claim.text}")


if __name__ == "__main__":
    asyncio.run(main())
