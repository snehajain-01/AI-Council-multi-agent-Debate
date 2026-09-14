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


ResponseType = Literal["accepted", "disputed", "clarified"]


class SingleCounterResponse(BaseModel):
    """What we ask the model for per criticism -- the category is already known, so we
    don't ask the model to restate it, removing one more chance for it to drift."""

    response_type: ResponseType = Field(
        description="'accepted' if the criticism is fair, 'disputed' if it is wrong, "
        "'clarified' if it stems from a misunderstanding"
    )
    explanation: str = Field(description="Why the agent accepts, disputes, or clarifies this point")


class CounterargumentPoint(BaseModel):
    """One agent's response to a single critique point made against it."""

    critique_category: CritiqueCategory = Field(description="Which critique this responds to")
    response_type: ResponseType
    explanation: str = Field(description="Why the agent accepts, disputes, or clarifies this point")


class Counterargument(BaseModel):
    """An agent's full response to all critiques made against its own position (Round 3)."""

    points: list[CounterargumentPoint]
