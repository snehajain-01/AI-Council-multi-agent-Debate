"""Endpoints for creating and checking on debates."""

import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.debate_agent import DebateAgent
from app.agents.debate_manager import DebateManager
from app.agents.fact_checker import FactChecker
from app.agents.judge_agent import JudgeAgent
from app.agents.synthesizer import Synthesizer
from app.db.models import DebateRecord
from app.db.session import async_session_factory, get_session
from app.providers.ollama_provider import OllamaProvider
from app.services.research_service import ResearchService

router = APIRouter(prefix="/debates", tags=["debates"])


class CreateDebateRequest(BaseModel):
    question: str


class DebateSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    status: str


class DebateDetail(DebateSummary):
    verdict: dict | None
    error: str | None


def _build_pipeline_components() -> tuple[DebateManager, JudgeAgent, Synthesizer, FactChecker, ResearchService]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    agents = [
        DebateAgent(name=name, provider=OllamaProvider(model=model, base_url=base_url))
        for name in ("Agent A", "Agent B", "Agent C")
    ]
    manager = DebateManager(agents)
    judge = JudgeAgent(provider=OllamaProvider(model=model, base_url=base_url))
    synthesizer = Synthesizer(provider=OllamaProvider(model=model, base_url=base_url))
    fact_checker = FactChecker(provider=OllamaProvider(model=model, base_url=base_url))
    research = ResearchService()
    return manager, judge, synthesizer, fact_checker, research


async def _run_debate_pipeline(debate_id: int, question: str) -> None:
    """Runs the full pipeline and updates the debate record when done.

    Uses its own sessions rather than the request's, since this runs as a
    detached background task outside the request/response lifecycle.
    """
    manager, judge, synthesizer, fact_checker, research = _build_pipeline_components()

    async with async_session_factory() as session:
        record = await session.get(DebateRecord, debate_id)
        record.status = "running"
        await session.commit()

    try:
        positions = await manager.run_round_1(question)
        round2_result = await manager.run_round_2(positions)
        counterarguments = await manager.run_round_3(positions, round2_result)
        revised = await manager.run_round_4(positions, counterarguments)
        judge_result = await manager.run_judging(question, revised, judge)
        evidence_report = await manager.run_verification(revised, fact_checker, research)
        consensus = await manager.run_consensus(
            question, revised, round2_result, judge_result, judge, evidence_report
        )
        verdict = await synthesizer.synthesize(question, revised, judge_result, consensus, evidence_report)

        async with async_session_factory() as session:
            record = await session.get(DebateRecord, debate_id)
            record.status = "completed"
            record.verdict = verdict.model_dump()
            await session.commit()
    except Exception as exc:
        async with async_session_factory() as session:
            record = await session.get(DebateRecord, debate_id)
            record.status = "failed"
            record.error = str(exc)
            await session.commit()


@router.post("", response_model=DebateSummary, status_code=201)
async def create_debate(
    request: CreateDebateRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> DebateRecord:
    record = DebateRecord(question=request.question, status="pending")
    session.add(record)
    await session.commit()
    await session.refresh(record)

    background_tasks.add_task(_run_debate_pipeline, record.id, request.question)

    return record


@router.get("/{debate_id}", response_model=DebateDetail)
async def get_debate(debate_id: int, session: AsyncSession = Depends(get_session)) -> DebateRecord:
    record = await session.get(DebateRecord, debate_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Debate not found")
    return record


@router.get("", response_model=list[DebateSummary])
async def list_debates(session: AsyncSession = Depends(get_session)) -> list[DebateRecord]:
    result = await session.execute(select(DebateRecord).order_by(DebateRecord.id.desc()))
    return list(result.scalars().all())
