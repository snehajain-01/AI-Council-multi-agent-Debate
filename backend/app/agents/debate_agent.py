"""A single debate participant, backed by any LLMProvider."""

from app.models.debate import AgentPosition, CritiqueSet
from app.providers.base import BaseLLMProvider

ROUND_1_SYSTEM_PROMPT = (
    "You are one of several independent analysts examining a question. "
    "You do not know what other analysts will say, and you must not assume "
    "consensus. Reason carefully and honestly, and be explicit about what "
    "would need to be true, or what evidence would be needed, to support or "
    "challenge your position. Acknowledge genuine weaknesses in your own "
    "reasoning rather than hiding them."
)

ROUND_2_SYSTEM_PROMPT = (
    "You are reviewing responses written by other analysts to the same question "
    "you just answered. You do not know their identities and must not guess at "
    "them -- judge each response purely on its content. Be rigorous and "
    "adversarial: actively look for factual errors, unsupported claims, logical "
    "fallacies, contradictions, missing information, weak assumptions, "
    "irrelevance, and incompleteness. Only report an issue if it is genuinely "
    "present -- do not invent problems in an otherwise sound response."
)


class DebateAgent:
    """Wraps an LLM provider to produce structured debate positions."""

    def __init__(self, name: str, provider: BaseLLMProvider):
        self.name = name
        self.provider = provider

    async def answer(self, question: str) -> AgentPosition:
        """Round 1: produce an independent, structured position on the question."""
        prompt = (
            f"Question: {question}\n\n"
            "Analyze this question on its own merits. "
            "Report confidence as a decimal fraction between 0 and 1 (e.g. 0.75), never as a percentage."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=ROUND_1_SYSTEM_PROMPT,
            json_schema=AgentPosition.model_json_schema(),
        )
        return AgentPosition.model_validate_json(response.content)

    async def critique(self, anonymized_positions: dict[str, AgentPosition]) -> CritiqueSet:
        """Round 2: critique other agents' positions, identified only by anonymized label."""
        sections = []
        for label, position in anonymized_positions.items():
            sections.append(
                f"{label}:\n"
                f"Position: {position.position}\n"
                f"Reasoning: {position.reasoning}\n"
                f"Key arguments: {'; '.join(position.key_arguments)}\n"
                f"Stated confidence: {position.confidence}"
            )
        valid_labels = list(anonymized_positions.keys())
        prompt = (
            "Here are the other analysts' responses, identified only by label:\n\n"
            + "\n\n".join(sections)
            + f"\n\nProduce exactly one critique object for each of these labels: {valid_labels}. "
            + "Use these exact strings as target_label -- do not invent, rename, or split labels. "
            + "If a response genuinely has no issues, leave its points list empty but still include it."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system=ROUND_2_SYSTEM_PROMPT,
            json_schema=CritiqueSet.model_json_schema(),
        )
        return CritiqueSet.model_validate_json(response.content)
