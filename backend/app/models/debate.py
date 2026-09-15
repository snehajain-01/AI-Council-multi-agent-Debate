"""Data shapes produced during a debate."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        """Small local models occasionally ignore the 0-1 instruction (e.g. returning 80
        or 1.1). Clamp rather than reject: this is untrusted model output, not internal
        state, and a crashed debate is worse than a saturated confidence value."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value
        if value > 1:
            value = value / 100 if value > 1.5 else 1.0
        return max(0.0, min(1.0, value))


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


class RevisedPosition(AgentPosition):
    """An agent's final position after incorporating accepted criticism (Round 4: Revision)."""

    changes_from_original: list[str] = Field(
        description="Concrete changes made from the original position and why each was made; "
        "empty only if no criticism was accepted"
    )


JUDGE_SCORE_FIELDS = (
    "factual_correctness",
    "logical_reasoning",
    "evidence_support",
    "relevance",
    "completeness",
    "clarity",
)


class JudgeScores(BaseModel):
    """A judge's rubric scores for one agent's final position, each on a 0-100 scale."""

    factual_correctness: float = Field(ge=0, le=100)
    logical_reasoning: float = Field(ge=0, le=100)
    evidence_support: float = Field(ge=0, le=100)
    relevance: float = Field(ge=0, le=100)
    completeness: float = Field(ge=0, le=100)
    clarity: float = Field(ge=0, le=100)
    justification: str = Field(description="A brief explanation of why these scores were given")

    @model_validator(mode="before")
    @classmethod
    def clamp_scores(cls, data):
        """Same rationale as AgentPosition.clamp_confidence: local models periodically
        ignore numeric range instructions, and a crashed debate is worse than a
        saturated score."""
        if not isinstance(data, dict):
            return data
        for field in JUDGE_SCORE_FIELDS:
            if field in data:
                try:
                    data[field] = max(0.0, min(100.0, float(data[field])))
                except (TypeError, ValueError):
                    pass
        return data


class JudgeVerdict(BaseModel):
    """One judge's full evaluation of one (anonymized) agent's final position."""

    target_label: str = Field(description="The anonymized label of the position that was scored")
    scores: JudgeScores
    weighted_total: float = Field(description="Weighted combination of the scores, computed in code, not by the model")


class JudgeResult(BaseModel):
    """The full output of judging, plus the de-anonymization key for internal use."""

    verdicts_by_label: dict[str, JudgeVerdict]
    label_map: dict[str, str] = Field(description="anonymized label -> real agent name")


AgreementLevel = Literal["full_agreement", "majority_agreement", "split", "full_disagreement"]


class AgreementAssessment(BaseModel):
    """An LLM's classification of whether the final (anonymized) positions actually
    agree in substance, not just in tone or phrasing."""

    agreement_level: AgreementLevel
    shared_conclusion: str = Field(
        description="The conclusion shared by agreeing agents, or an empty string if there is none"
    )
    key_disagreements: list[str] = Field(
        description="Specific substantive points where positions genuinely differ; empty if none"
    )


ConsensusLevel = Literal["strong_consensus", "partial_consensus", "no_consensus"]


class ConsensusResult(BaseModel):
    """The Consensus Engine's final determination, combining agreement, judge scores,
    and how much the agents challenged each other during debate.

    Consensus is not majority vote and is not truth: it is possible, and must remain
    possible, for this to resolve to "no_consensus" when agents genuinely disagree or
    the judge finds their final positions too uneven in quality to treat as reliable.
    """

    consensus_level: ConsensusLevel
    agreement: AgreementAssessment
    average_judge_score: float
    judge_score_spread: float = Field(description="max weighted_total minus min weighted_total across agents")
    total_critique_volume: int = Field(description="Total critique points raised across all agents in Round 2")
    explanation: str = Field(description="Human-readable summary of why this consensus level was reached")


class SynthesizedAnswer(BaseModel):
    """What we ask the model for: just the final answer text. Every other field on
    CouncilVerdict is already known from earlier, code-computed pipeline stages, so
    there's no reason to re-expose them to another LLM call and risk drift."""

    final_answer: str = Field(
        description="The final answer to the original question, written to honestly reflect the "
        "actual consensus level -- hedged or explicitly split if agents disagreed, confident only "
        "if they genuinely agreed"
    )


class CouncilVerdict(BaseModel):
    """The complete, user-facing output of a debate."""

    question: str
    final_answer: str
    consensus_level: ConsensusLevel
    consensus_explanation: str
    confidence: float = Field(ge=0, le=100, description="Average judge score across final positions, out of 100")
    strongest_agent: str = Field(description="The agent whose final position the judge scored highest")
    strongest_agent_score: float
    key_disagreements: list[str]
    agent_positions: dict[str, str] = Field(description="Each agent's final position, for transparency")
