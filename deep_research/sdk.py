"""
deep_research/sdk.py
Public Python SDK — import and use deep_research as a library.

Usage:
    from deep_research.sdk import DeepResearch

    dr = DeepResearch(provider="claude", anthropic_api_key="sk-ant-...")
    result = dr.research("What is the state of fusion energy in 2025?")
    print(result.report_md)
    print(result.bullets)
    for qa in result.qa_data:
        print(qa["question"], "->", qa["answer"])
"""
from __future__ import annotations
from typing import Optional, List
from pathlib import Path

from .core.config import Config
from .agents.pipeline import run_pipeline, ResearchContext
from .graph.reputation import ReputationGraph
from .vault.manager import VaultManager


class DeepResearch:
    """
    High-level SDK for the Deep Research system.
    Supports Claude, OpenAI, and Gemini. Works without a database (local JSON vault).
    """

    def __init__(
        self,
        provider: str = "claude",
        model: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        database_url: Optional[str] = None,
        vault_path: Optional[str] = None,
        cloud_sync_url: Optional[str] = None,
        auto_sync: bool = False,
        warn_cb=None,
    ):
        from .core.config import cfg as _cfg
        self.cfg = _cfg.override(
            default_provider=provider,
            default_model=model or "",
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            tavily_api_key=tavily_api_key,
            serper_api_key=serper_api_key,
            database_url=database_url,
            vault_local_path=Path(vault_path) if vault_path else None,
            cloud_sync_url=cloud_sync_url,
            vault_auto_sync=auto_sync,
        )
        if self.cfg.database_url:
            from .vault.db import init_db
            init_db(self.cfg.database_url)

        self.graph = ReputationGraph(config=self.cfg, warn_cb=warn_cb)
        self.vault = VaultManager(
            self.cfg.vault_local_path,
            self.cfg.cloud_sync_url,
            self.cfg.vault_auto_sync,
        )

    def research(
        self,
        query: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        custom_urls: Optional[List[str]] = None,
        save: bool = True,
    ) -> ResearchContext:
        """
        Run the full pipeline. Returns ResearchContext with:
          .report_md   — full Markdown report
          .bullets     — bullet-point summary
          .qa_data     — list of {question, answer, source_indices}
          .critiqued   — scored source list
          .run_id      — vault reference
        """
        ctx = run_pipeline(
            query=query,
            provider=provider or self.cfg.default_provider,
            model=model,
            custom_urls=custom_urls,
            graph=self.graph,
            config=self.cfg,
        )
        if save:
            self.vault.save(ctx)
        return ctx

    def add_source(self, url: str, score: float = 60.0):
        """Register a custom trusted source."""
        self.graph.register_custom_source(url, score)

    def blacklist(self, url: str, reason: str = "SDK blacklist"):
        """Blacklist a source."""
        self.graph.blacklist_manual(url, reason)

    def list_runs(self, limit: int = 20) -> list[dict]:
        """List recent research runs from the vault."""
        return self.vault.list_runs(limit)

    def load_run(self, run_id: str) -> Optional[dict]:
        """Load a saved research run."""
        return self.vault.load(run_id)

    def source_scores(self, limit: int = 50) -> list[dict]:
        """Return current reputation scores for all known sources."""
        # Try DB first
        if self.cfg.database_url:
            try:
                from .vault.db import get_session, Source
                session = get_session()
                sources = session.query(Source).order_by(Source.score.desc()).limit(limit).all()
                session.close()
                return [
                    {"domain": s.domain, "score": s.score, "blacklisted": s.blacklisted}
                    for s in sources
                ]
            except Exception:
                pass
        # Fallback: in-memory graph
        scores = self.graph.get_all_scores()
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]
