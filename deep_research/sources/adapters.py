"""
deep_research/sources/adapters.py
Pluggable search source adapters. Each returns List[SourceResult].
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import requests
import arxiv

from ..core.config import cfg


@dataclass
class SourceResult:
    url: str
    title: str
    snippet: str
    source_type: str          # tavily | serper | arxiv | claude_web | custom
    doi: Optional[str] = None
    authors: Optional[str] = None
    raw_content: Optional[str] = None   # full text if fetched


# ── Tavily ────────────────────────────────────────────────────────────────────

def search_tavily(query: str, max_results: int = 10, config=None) -> list[SourceResult]:
    c = config or cfg
    if not c.tavily_api_key:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=c.tavily_api_key)
        resp = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
            search_depth="advanced",
        )
        results = []
        for r in resp.get("results", []):
            results.append(SourceResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                raw_content=r.get("raw_content", ""),
                source_type="tavily",
            ))
        return results
    except Exception as e:
        print(f"[Tavily] Error: {e}")
        return []


# ── Serper / SerpAPI ──────────────────────────────────────────────────────────

def search_serper(query: str, max_results: int = 10, config=None) -> list[SourceResult]:
    c = config or cfg
    if not c.serper_api_key:
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": c.serper_api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("organic", []):
            results.append(SourceResult(
                url=r.get("link", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                source_type="serper",
            ))
        return results
    except Exception as e:
        print(f"[Serper] Error: {e}")
        return []


# ── ArXiv ─────────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 8, config=None) -> list[SourceResult]:
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append(SourceResult(
                url=paper.entry_id,
                title=paper.title,
                snippet=paper.summary[:800],
                source_type="arxiv",
                doi=paper.doi,
                authors=", ".join(str(a) for a in paper.authors[:5]),
                raw_content=paper.summary,
            ))
        return results
    except Exception as e:
        print(f"[ArXiv] Error: {e}")
        return []


# ── Custom user sources ───────────────────────────────────────────────────────

def search_custom(query: str, custom_urls: list[str], config=None) -> list[SourceResult]:
    """
    Given a list of user-registered URLs, fetch and return as SourceResults.
    Basic keyword relevance filter applied.
    """
    results = []
    keywords = set(query.lower().split())
    for url in custom_urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "DeepResearchBot/1.0"})
            resp.raise_for_status()
            text = resp.text[:5000]
            # Simple relevance: at least 2 query keywords appear
            matched = sum(1 for kw in keywords if kw in text.lower())
            if matched >= 2:
                title = re.search(r"<title>(.*?)</title>", text, re.I)
                results.append(SourceResult(
                    url=url,
                    title=title.group(1) if title else url,
                    snippet=text[:400],
                    raw_content=text,
                    source_type="custom",
                ))
        except Exception as e:
            print(f"[Custom] Could not fetch {url}: {e}")
    return results


# ── Fan-out search: all sources combined ──────────────────────────────────────

def search_all(
    query: str,
    custom_urls: list[str] | None = None,
    max_per_source: int = 8,
    config=None,
) -> list[SourceResult]:
    """Run all enabled sources in sequence, deduplicate by URL."""
    c = config or cfg
    seen_urls: set[str] = set()
    combined: list[SourceResult] = []

    sources = [
        ("Tavily",  lambda: search_tavily(query, max_per_source, c)),
        ("Serper",  lambda: search_serper(query, max_per_source, c)),
        ("ArXiv",   lambda: search_arxiv(query, max_per_source, c)),
    ]
    if custom_urls:
        sources.append(("Custom", lambda: search_custom(query, custom_urls, c)))

    for name, fn in sources:
        try:
            results = fn()
            for r in results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    combined.append(r)
        except Exception as e:
            print(f"[{name}] Failed: {e}")

    return combined
