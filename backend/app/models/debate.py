"""Data shapes produced during a debate."""

from typing import Literal

from pydantic import BaseModel, Field


class AgentPosition(BaseModel):
    """An agent's structured answer to a question (Round 1: Independent Reasoning)."""

    position: str = Field(description="A concise statement of the agent's stance or answer")
    reasoning: str = Field(description="The agent's full reasoning behind the position")
    key_arguments: list[str] = Field(description="The strongest points supporting the position")
    evidence_requirements: list[str] = Field(
        description="What evidence would be needed to confirm or challenge this position"
    )
    confidence: float = Field(ge=0, le=1, description="Self-reported confidence from 0 to 1")
    weaknesses: list[str] = Field(description="Potential weaknesses the agent sees in its own position")


CritiqueCategory = Literal[
    "factual_error",
    "unsupported_claim",
    "logical_fallacy",
    "contradiction",
    "missing_information",
    "weak_assumption",
    "relevance",
    "completeness",
]


class CritiquePoint(BaseModel):
    """A single issue found in another agent's position."""

    category: CritiqueCategory
    description: str = Field(description="A specific, concrete explanation of the issue")


class Critique(BaseModel):
    """One agent's critique of one anonymized response (Round 2: Blind Cross-Critique)."""

    target_label: str = Field(description="The anonymized label of the response being critiqued, e.g. 'Response 1'")
    points: list[CritiquePoint] = Field(description="Specific issues found; empty if none found")
    overall_assessment: str = Field(description="A brief summary judgment of this response's quality")


class CritiqueSet(BaseModel):
    """All of one agent's critiques of the other (anonymized) agents' positions."""

    critiques: list[Critique]


class Round2Result(BaseModel):
    """The full output of Round 2, plus the de-anonymization key for internal use."""

    critiques_by_agent: dict[str, CritiqueSet]
    label_maps: dict[str, dict[str, str]] = Field(
        description="critiquing agent name -> {anonymized label -> real target agent name}"
    )
