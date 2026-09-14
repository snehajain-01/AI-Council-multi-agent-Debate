"""Coordinates debate agents through the rounds of the debate protocol."""

import asyncio
import random

from app.agents.debate_agent import DebateAgent
from app.models.debate import AgentPosition, CritiqueSet, Round2Result


class DebateManager:
    """Runs a group of DebateAgents through the structured debate rounds."""

    def __init__(self, agents: list[DebateAgent]):
        self.agents = agents

    async def run_round_1(self, question: str) -> dict[str, AgentPosition]:
        """Each agent answers independently and concurrently."""
        positions = await asyncio.gather(*(agent.answer(question) for agent in self.agents))
        return {agent.name: position for agent, position in zip(self.agents, positions)}

    async def run_round_2(self, positions: dict[str, AgentPosition]) -> Round2Result:
        """Each agent critiques the others' anonymized, randomly-relabeled positions."""
        label_maps: dict[str, dict[str, str]] = {}
        critique_tasks = []

        for agent in self.agents:
            other_names = [name for name in positions if name != agent.name]
            shuffled_names = random.sample(other_names, len(other_names))
            label_map = {f"Response {i + 1}": name for i, name in enumerate(shuffled_names)}
            label_maps[agent.name] = label_map

            anonymized_positions = {label: positions[name] for label, name in label_map.items()}
            critique_tasks.append(agent.critique(anonymized_positions))

        critique_sets: list[CritiqueSet] = await asyncio.gather(*critique_tasks)
        critiques_by_agent = {agent.name: cs for agent, cs in zip(self.agents, critique_sets)}

        return Round2Result(critiques_by_agent=critiques_by_agent, label_maps=label_maps)
