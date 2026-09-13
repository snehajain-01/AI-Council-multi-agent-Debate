"""Data shapes produced during a debate."""

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
