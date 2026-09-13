"""A single debate participant, backed by any LLMProvider."""

from app.models.debate import AgentPosition
from app.providers.base import BaseLLMProvider

ROUND_1_SYSTEM_PROMPT = (
    "You are one of several independent analysts examining a question. "
    "You do not know what other analysts will say, and you must not assume "
    "consensus. Reason carefully and honestly, and be explicit about what "
    "would need to be true, or what evidence would be needed, to support or "
    "challenge your position. Acknowledge genuine weaknesses in your own "
    "reasoning rather than hiding them."
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
