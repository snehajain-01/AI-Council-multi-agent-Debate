"""Data shapes for claim extraction and fact verification."""

from typing import Literal

from pydantic import BaseModel, Field

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
