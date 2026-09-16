"""Data shapes for information sources used in evidence retrieval."""

from typing import Literal

from pydantic import BaseModel

SourceReliability = Literal["very_high", "high", "medium_high", "medium", "low"]


class Source(BaseModel):
    """A single retrieved source, with a heuristic reliability tier.

    Reliability reflects the general trustworthiness of the source type (e.g.
    peer-reviewed research vs. a random blog), not a judgment on whether this
    specific claim is true -- it is a heuristic, not an absolute measure of truth.
    """

    title: str
    url: str
    reliability: SourceReliability
