"""Data shapes for evidence gathered in support of, or against, a claim."""

from pydantic import BaseModel, Field

from app.models.source import Source


class Evidence(BaseModel):
    """A snippet of text retrieved from a source, relevant to a specific claim."""

    source: Source
    excerpt: str = Field(description="The relevant excerpt or snippet from the source")
