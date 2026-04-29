"""
deep_research/agents/pipeline.py
Full Self-Healing GraphRAG Pipeline:
  Planner → Retriever (Fast) → Confidence Check → Crawler/Searcher (Deep) → Extractor → Critic → Writer
"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.console import Group
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..core import llm
from ..core.config import cfg
from ..sources.adapters import SourceResult, search_all
from ..graph.reputation import ReputationGraph
from ..vault.retriever import HybridRetriever
from ..sources.crawler import GuidedCrawler
from .extractor import ExtractorAgent


from ..vault.interface import BatchResearchResult, ExtractedFact
from ..vault.manager import VaultManager


@dataclass
class ResearchContext:
    run_id: str
    query: str
    sub_questions: list[str] = field(default_factory=list)
    raw_results: list[SourceResult] = field(default_factory=list)
    compressed: list[dict] = field(default_factory=list)
    extracted_facts: list[ExtractedFact] = field(default_factory=list)
    critiqued: list[dict] = field(default_factory=list)
    report_md: str = ""
    bullets: str = ""
    qa_data: list[dict] = field(default_factory=list)
    provider: str = "claude"
    model: str = ""
    confidence_score: float = 0.0


# ── Live UI Helper ────────────────────────────────────────────────────────────

class LiveStatus:
    def __init__(self, query: str):
        self.query = query
        self.current_step = "Initializing"
        self.steps_completed = []
        self.active_agent = "Orchestrator"
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TaskProgressColumn(),
        )
        self.crawl_task = None

    def __rich__(self) -> Group:
        # 1. Query Panel (Gemini Style)
        header = Panel(
            f"[bold white]Query:[/] [dim]{self.query}[/]",
            border_style="bright_black",
            padding=(0, 1)
        )

        # 2. History & Activity (Agentic Style)
        lines = []
        for s in self.steps_completed:
            lines.append(f" [green]●[/] [white]{s}[/]")
        
        # Current active step with "branch" connector
        agent_chip = f"[black on cyan] {self.active_agent} [/]"
        lines.append(f" [cyan]○[/] {agent_chip} [bold white]{self.current_step}[/] [dim]...[/]")
        
        # 3. Progress (if crawling)
        renderables = [header, "\n".join(lines)]
        
        if self.crawl_task is not None:
            renderables.append("\n" + "   ⎿  ")
            renderables.append(self.progress)
            
        return Group(*renderables)


# ── Agent 1: Planner ──────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a research planning expert. 
Break a user research query into 3-6 precise sub-questions that together cover the topic fully.
Respond ONLY with a JSON array of strings. No preamble."""

def planner_agent(query: str, provider: str, model: str, config=None) -> list[str]:
    prompt = f"Research query: {query}\n\nGenerate sub-questions:"
    raw = llm.complete(prompt, system=PLANNER_SYSTEM, provider=provider, model=model, config=config)
    try:
        raw = raw.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(raw)
    except Exception:
        return [query]


# ── Agent 2: Critic (Scores Sources) ──────────────────────────────────────────

CRITIC_SYSTEM = """You are a research credibility critic.
Evaluate the source summary on factual consistency, specificity, and bias.
Return ONLY a JSON object with:
  "credibility_score": float 0-100,
  "llm_judgment": "high"|"medium"|"low",
  "issues": [list of strings, or empty],
  "delta": float (reward/penalty -20 to +20)"""

def critic_agent(
    compressed: list[dict],
    graph: ReputationGraph,
    run_id: str,
    provider: str,
    model: str,
    config=None,
) -> list[dict]:
    critiqued = []
    for item in compressed:
        prompt = f"URL: {item['url']}\nTitle: {item.get('title', '')}\nSummary:\n{item['summary']}"
        try:
            raw = llm.complete(prompt, system=CRITIC_SYSTEM, provider=provider, model=model, max_tokens=256, config=config)
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            verdict = json.loads(raw)
        except Exception:
            verdict = {"credibility_score": 50.0, "llm_judgment": "medium", "issues": [], "delta": 0}

        delta = verdict.get("delta", 0)
        if delta != 0:
            graph.adjust(item["url"], delta, f"LLM critic judgment: {verdict.get('llm_judgment', 'medium')}", run_id)

        graph_score = graph.score(item["url"])
        llm_score   = verdict.get("credibility_score", 50.0)
        composite   = round((graph_score * 0.5) + (llm_score * 0.5), 1)

        critiqued.append({
            **item,
            "graph_score": graph_score,
            "llm_score": llm_score,
            "composite_score": composite,
            "llm_judgment": verdict.get("llm_judgment", "medium"),
            "issues": verdict.get("issues", []),
        })

    critiqued.sort(key=lambda x: x["composite_score"], reverse=True)
    return critiqued


# ── Agent 3: Writer & Confidence Evaluator ────────────────────────────────────

WRITER_SYSTEM = """You are an expert research report writer and evaluator.
You receive a research query and scored source summaries.
First, output a confidence score (0.0 to 1.0) indicating if the provided sources sufficiently answer the query.
Then, write the report.

Respond ONLY with a JSON object:
{
  "confidence": float,
  "report_md": "Full markdown report...",
  "bullets": "Bullet points...",
  "qa_data": [{"question": "...", "answer": "...", "source_indices": []}]
}"""

def writer_agent(
    query: str,
    sub_questions: list[str],
    critiqued: list[dict],
    provider: str,
    model: str,
    config=None,
) -> tuple[float, str, str, list[dict]]:
    sources_text = ""
    for i, item in enumerate(critiqued[:15], 1):
        sources_text += f"\n[Source {i}] {item.get('title', item['url'])}\nURL: {item['url']}\nSummary: {item['summary']}\n"

    prompt = f"""Query: {query}
Sub-questions: {sub_questions}
Sources:
{sources_text}"""
    
    try:
        raw = llm.complete(prompt, system=WRITER_SYSTEM, provider=provider, model=model, max_tokens=8192, config=config)
        raw = raw.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(raw)
        return data.get("confidence", 0.0), data.get("report_md", ""), data.get("bullets", ""), data.get("qa_data", [])
    except Exception as e:
        print(f"[Writer] Failed to parse JSON: {e}")
        return 0.0, "Generation failed.", "", []


# ── Phase 3: The Orchestration Layer (Self-Healing Loop) ──────────────────────

def run_pipeline(
    query: str,
    provider: str | None = None,
    model: str | None = None,
    custom_urls: list[str] | None = None,
    graph: ReputationGraph | None = None,
    vault: VaultManager | None = None,
    warn_cb=None,
    config=None,
) -> ResearchContext:
    c = config or cfg
    p, m = c.resolve_model(provider)
    if model:
        m = model

    run_id = str(uuid.uuid4())[:8]
    ctx = ResearchContext(run_id=run_id, query=query, provider=p, model=m)
    
    # Initialize Storage Layer (The Vault Seam)
    vault = vault or VaultManager(c.vault_local_path, c.cloud_sync_url, c.vault_auto_sync)
    graph = graph or ReputationGraph(vault=vault, config=c, warn_cb=warn_cb)

    retriever = HybridRetriever(vault)
    crawler = GuidedCrawler(config=c, provider=p, model=m)
    extractor = ExtractorAgent(provider=p, model=m)

    status = LiveStatus(query)
    with Live(status, refresh_per_second=10) as live:
        status.current_step = "Planning sub-queries"
        status.active_agent = "PlannerAgent"
        ctx.sub_questions = planner_agent(query, p, m, c)
        status.steps_completed.append("Research plan generated")

        # ATTEMPT 1: Fast GraphRAG Retrieval (The Vault)
        status.current_step = "Retrieving from Vault (Hybrid GraphRAG)"
        status.active_agent = "RetrieverAgent"
        vault_results = retriever.retrieve(query, top_k=10)

        if vault_results:
            ctx.compressed = vault_results
            status.current_step = "Evaluating source credibility"
            status.active_agent = "CriticAgent"
            ctx.critiqued = critic_agent(ctx.compressed, graph, run_id, p, m, c)

            status.current_step = "Synthesizing vault findings"
            status.active_agent = "WriterAgent"
            conf, r_md, bul, qa = writer_agent(query, ctx.sub_questions, ctx.critiqued, p, m, c)
            ctx.confidence_score = conf

            if conf > 0.85:
                status.steps_completed.append(f"Vault retrieval sufficient (Confidence: {conf:.2f})")
                ctx.report_md, ctx.bullets, ctx.qa_data = r_md, bul, qa
                status.current_step = "Finalizing report"
                return ctx

        # ATTEMPT 2: Fallback to Web Search + Deep Crawl (Firecrawl)
        status.steps_completed.append(f"Vault coverage low. Falling back to web search.")
        status.current_step = "Searching the web"
        status.active_agent = "SearchAdapter"
        raw_results = []
        for sq in ctx.sub_questions:
            raw_results.extend(search_all(sq, custom_urls, config=c))

        # Semantic Routing: pick best links
        links = [r.url for r in raw_results if not graph.is_blacklisted(r.url)]
        chosen_links = crawler.semantic_route(query, list(set(links)))

        status.steps_completed.append(f"Discovered {len(links)} sources, routing {len(chosen_links)} high-signal links")
        status.current_step = "Crawling & distilling content"
        status.active_agent = "Firecrawl + Extractor"
        status.crawl_task = status.progress.add_task("Crawling...", total=len(chosen_links))

        for url in chosen_links:
            raw_md = crawler.crawl(url)
            if raw_md:
                fact = extractor.distill(url, raw_md, query)
                ctx.extracted_facts.append(fact)
                ctx.compressed.append({"url": url, "summary": fact.summary, "title": url})
            status.progress.advance(status.crawl_task)

        # Final Evaluation & Writing
        status.steps_completed.append("Deep crawl completed")
        status.current_step = "Evaluating newly crawled sources"
        status.active_agent = "CriticAgent"
        ctx.critiqued = critic_agent(ctx.compressed, graph, run_id, p, m, c)

        status.current_step = "Generating final report"
        status.active_agent = "WriterAgent"
        conf, r_md, bul, qa = writer_agent(query, ctx.sub_questions, ctx.critiqued, p, m, c)
        ctx.confidence_score = conf
        ctx.report_md, ctx.bullets, ctx.qa_data = r_md, bul, qa

        # Atomic Batch Save
        status.current_step = "Saving atomic batch result"
        batch_result = BatchResearchResult(
            run_id=run_id,
            query=query,
            provider=p,
            model=m,
            report_md=ctx.report_md,
            bullets=ctx.bullets,
            qa_data=ctx.qa_data,
            extracted_facts=ctx.extracted_facts,
            reputation_updates=graph.pending_updates
        )
        vault.save_run(batch_result)
        status.steps_completed.append("Research run persisted to Vault")
        status.current_step = "Done"

    # Self-Healing Penalty: If still low confidence, penalize these sources
    if conf < 0.5:
        print("   ⚠️ Final confidence still low. Penalizing sources for poor signal.")
        for item in ctx.critiqued:
            graph.penalize(item['url'], 2.0, "Led to low confidence synthesis.", run_id)

    return ctx
