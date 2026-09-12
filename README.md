# AI Council — Multi-Agent Debate & Consensus Engine

A multi-agent AI system where independent AI models analyze the same question, debate competing answers, critique each other's reasoning, verify factual claims using external evidence, and produce a consensus-aware final response with confidence and disagreement analysis.

## Research Question

> Can structured debate among independent AI agents improve the reliability, reasoning quality, and factual accuracy of generated answers?

## Project Status

🚧 **Under active development.** This project is being built incrementally, one phase at a time. See [docs/architecture.md](docs/architecture.md) (coming soon) for the full system design.

**Current phase:** Phase 1 — Project Foundation

## Cost

This project is designed to run at **$0 / ₹0**. The primary AI backend is [Ollama](https://ollama.com) running local open-source models. Paid APIs are never required.

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic
- **AI:** Ollama (local LLMs)
- **Frontend:** React, Tailwind CSS
- **Database:** PostgreSQL
- **Vector Search:** FAISS / Chroma

## Project Structure

```text
ai-council/
├── backend/     # FastAPI application, agents, providers, services
├── frontend/    # React application
├── docs/        # Architecture and design documentation
└── README.md
```

## Setup

Setup instructions will be added as each phase of the project is built.

## License

TBD
