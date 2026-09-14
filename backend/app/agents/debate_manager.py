"""Coordinates debate agents through the rounds of the debate protocol."""

import asyncio
import random

from app.agents.debate_agent import DebateAgent
from app.models.debate import AgentPosition, Counterargument, CritiquePoint, CritiqueSet, Round2Result


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

    async def run_round_3(
        self, positions: dict[str, AgentPosition], round2_result: Round2Result
    ) -> dict[str, Counterargument]:
        """Each agent responds to all critiques made against its own (real) position."""
        points_by_target: dict[str, list[CritiquePoint]] = {agent.name: [] for agent in self.agents}

        for critiquing_agent_name, critique_set in round2_result.critiques_by_agent.items():
            label_map = round2_result.label_maps[critiquing_agent_name]
            for critique in critique_set.critiques:
                target_name = label_map.get(critique.target_label)
                if target_name is None:
                    # Critiquing agent used a label that doesn't resolve to a real
                    # target (a small-model slip) -- skip rather than misattribute.
                    continue
                points_by_target[target_name].extend(critique.points)

        counter_tasks = [
            agent.respond_to_critiques(positions[agent.name], points_by_target[agent.name])
            for agent in self.agents
        ]
        counterarguments = await asyncio.gather(*counter_tasks)
        return {agent.name: ca for agent, ca in zip(self.agents, counterarguments)}
