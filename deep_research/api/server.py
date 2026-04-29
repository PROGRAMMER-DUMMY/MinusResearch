"""
deep_research/api/server.py
FastAPI REST API — exposes all pipeline + vault + graph operations.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ..core.config import cfg
from ..agents.pipeline import run_pipeline
from ..graph.reputation import ReputationGraph
from ..vault.manager import VaultManager

_graph: Optional[ReputationGraph] = None
_vault: Optional[VaultManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _vault
    if cfg.database_url:
        from ..vault.db import init_db
        init_db(cfg.database_url)
    _vault = VaultManager(cfg.vault_local_path, cfg.cloud_sync_url, cfg.vault_auto_sync)
    _graph = ReputationGraph(vault=_vault, config=cfg, warn_cb=_api_warn_cb)
    yield


app = FastAPI(title="Deep Research API", version="1.0.0", lifespan=lifespan)


def _api_warn_cb(domain: str, score: float, notif_id: int) -> str:
    return "dismissed"


# ── Request / Response models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    provider: Optional[str] = None
    model: Optional[str] = None
    custom_urls: Optional[List[str]] = None
    save: bool = True


class ResearchResponse(BaseModel):
    run_id: str
    query: str
    provider: str
    model: str
    report_md: str
    bullets: str
    qa_data: list
    sources: list


class SourceRequest(BaseModel):
    url: str
    initial_score: float = 60.0


class BlacklistRequest(BaseModel):
    url: str
    reason: str = "API blacklist"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    ctx = run_pipeline(
        query=req.query,
        provider=req.provider,
        model=req.model,
        custom_urls=req.custom_urls,
        graph=_graph,
        vault=_vault,
        config=cfg,
    )
    return ResearchResponse(
        run_id=ctx.run_id,
        query=ctx.query,
        provider=ctx.provider,
        model=ctx.model,
        report_md=ctx.report_md,
        bullets=ctx.bullets,
        qa_data=ctx.qa_data,
        sources=ctx.critiqued,
    )


@app.get("/vault")
def list_vault(limit: int = 20):
    if not _vault:
        raise HTTPException(503, "Vault not initialized")
    return _vault.list_runs(limit)


@app.get("/vault/{run_id}")
def get_run(run_id: str):
    if not _vault:
        raise HTTPException(503, "Vault not initialized")
    data = _vault.load(run_id)
    if not data:
        raise HTTPException(404, f"Run {run_id} not found")
    return data


@app.post("/sources/add")
def add_source(req: SourceRequest):
    _graph.register_custom_source(req.url, req.initial_score)
    return {"status": "registered", "url": req.url, "score": req.initial_score}


@app.post("/sources/blacklist")
def blacklist_source(req: BlacklistRequest):
    _graph.blacklist_manual(req.url, req.reason)
    return {"status": "blacklisted", "url": req.url}


@app.get("/graph/scores")
def graph_scores(limit: int = 50):
    if _graph:
        return _graph.get_all_scores()[:limit]
    return []


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


def start(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    uvicorn.run("deep_research.api.server:app", host=host, port=port, reload=reload)
