"""Retrieves evidence for claims from free, public sources.

Currently backed by Wikipedia's public search API, which requires no API key and
has no meaningful rate limit for this project's usage. Domain-specific sources
(OpenAlex, Semantic Scholar, arXiv, etc.) for academic/research questions can be
added here later as additional lookup methods, per the "add domain-specific
evaluation progressively" principle -- they aren't needed for the general factual
claims the debate agents currently produce.
"""

import re

import httpx

from app.models.evidence import Evidence
from app.models.source import Source

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Community-edited, not a primary source, but generally accurate and heavily
# cross-checked -- a reasonable default tier for general factual claims.
WIKIPEDIA_RELIABILITY = "medium_high"

# Wikimedia's API policy blocks generic HTTP client User-Agents (e.g. httpx's
# default) with 403 Forbidden and requires real contact info -- an email or a
# URL. Using the project's public repo here rather than a personal email.
# https://meta.wikimedia.org/wiki/User-Agent_policy
REQUEST_HEADERS = {
    "User-Agent": "AICouncil/0.1 (https://github.com/snehajain-01/AI-Council-multi-agent-Debate)"
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class ResearchService:
    """Searches free public sources for evidence relevant to a claim."""

    def __init__(self, max_results: int = 3, timeout: float = 15.0):
        self.max_results = max_results
        self.timeout = timeout

    async def search(self, query: str) -> list[Evidence]:
        """Search Wikipedia for pages relevant to the query and return their search
        snippets as evidence. Snippets come directly from Wikipedia's search index,
        so no second page-fetch is needed for this first pass."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": self.max_results,
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers=REQUEST_HEADERS) as client:
            response = await client.get(WIKIPEDIA_API_URL, params=params)
            response.raise_for_status()
        data = response.json()

        results = data.get("query", {}).get("search", [])
        evidence = []
        for result in results:
            title = result["title"]
            snippet = _HTML_TAG_RE.sub("", result.get("snippet", ""))
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            evidence.append(
                Evidence(
                    source=Source(title=title, url=url, reliability=WIKIPEDIA_RELIABILITY),
                    excerpt=snippet,
                )
            )
        return evidence
