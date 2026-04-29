"""
deep_research/cli/main.py
Typer CLI — research, vault, sources, graph commands.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Annotated
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(name="deepresearch", help="Deep Research Agent — grounded, cited, scored.")
console = Console()


def _bootstrap(database_url: str):
    from ..vault.db import init_db
    init_db(database_url)


# ── research ──────────────────────────────────────────────────────────────────

@app.command()
def research(
    query: Annotated[str, typer.Argument(help="Research query")],
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="claude | openai | gemini")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Override model name")] = None,
    custom: Annotated[List[str], typer.Option("--custom", "-c", help="Custom source URLs (repeat flag)")] = [],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save report to file")] = None,
    no_save: Annotated[bool, typer.Option("--no-save", help="Don't persist to vault")] = False,
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL", help="PostgreSQL URL")] = "",
):
    """Run the full 5-agent research pipeline on a query."""
    from ..core.config import cfg
    from ..agents.pipeline import run_pipeline
    from ..graph.reputation import ReputationGraph
    from ..vault.manager import VaultManager

    c = cfg.override(
        default_provider=provider or cfg.default_provider,
        database_url=db or cfg.database_url,
    )
    _bootstrap(c.database_url)

    vault = VaultManager(c.vault_local_path, c.cloud_sync_url, c.vault_auto_sync)
    graph = ReputationGraph(vault=vault, config=c)

    console.print(Panel(f"[bold cyan]Deep Research[/]\n[dim]{query}[/dim]", expand=False))

    ctx = run_pipeline(
        query=query,
        provider=provider,
        model=model or None,
        custom_urls=list(custom) or None,
        graph=graph,
        vault=vault,
        config=c,
    )

    console.print("\n" + "─" * 60)
    console.print(Markdown(ctx.report_md))

    console.print("\n[bold]Bullet Summary[/bold]")
    console.print(Markdown(ctx.bullets))

    if ctx.qa_data:
        console.print("\n[bold]Q&A Drill-down[/bold]")
        for qa in ctx.qa_data:
            console.print(f"[cyan]Q: {qa['question']}[/]")
            console.print(f"   {qa['answer']}\n")

    _print_source_table(ctx.critiqued)

    if output:
        output.write_text(ctx.report_md, encoding="utf-8")
        console.print(f"\n[green]Report saved to {output}[/green]")

    if not no_save:
        run_id = vault.save(ctx)
        console.print(f"\n[dim]Vault run ID: {run_id}[/dim]")


def _print_source_table(critiqued: list[dict]):
    if not critiqued:
        return
    table = Table(title="Source Credibility Scores", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Domain", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Judgment", justify="center")
    table.add_column("Issues")
    for i, s in enumerate(critiqued[:15], 1):
        from urllib.parse import urlparse
        domain = urlparse(s["url"]).netloc or s["url"]
        composite = s.get("composite_score", 0)
        score_color = "green" if composite >= 70 else ("yellow" if composite >= 40 else "red")
        table.add_row(
            str(i),
            domain,
            f"[{score_color}]{composite:.0f}[/]",
            s.get("llm_judgment", "—"),
            "; ".join(s["issues"][:2]) if s.get("issues") else "—",
        )
    console.print(table)


# ── vault ─────────────────────────────────────────────────────────────────────

@app.command()
def vault_list(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL")] = "",
):
    """List recent research runs in the vault."""
    from ..core.config import cfg
    from ..vault.manager import VaultManager
    c = cfg.override(database_url=db or cfg.database_url)
    _bootstrap(c.database_url)
    vm = VaultManager(c.vault_local_path)
    runs = vm.list_runs(limit)
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return
    table = Table(title="Vault — Recent Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Query")
    table.add_column("Status")
    table.add_column("Created")
    for r in runs:
        table.add_row(r["run_id"], r["query"][:60], r["status"], r["created_at"][:19])
    console.print(table)


@app.command()
def vault_show(
    run_id: Annotated[str, typer.Argument()],
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL")] = "",
):
    """Show a saved research report from the vault."""
    from ..core.config import cfg
    from ..vault.manager import VaultManager
    c = cfg.override(database_url=db or cfg.database_url)
    _bootstrap(c.database_url)
    vm = VaultManager(c.vault_local_path)
    data = vm.load(run_id)
    if not data:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)
    console.print(Markdown(data["report_md"]))


# ── sources ───────────────────────────────────────────────────────────────────

@app.command()
def source_add(
    url: Annotated[str, typer.Argument(help="URL to register as custom source")],
    score: Annotated[float, typer.Option("--score", "-s")] = 60.0,
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL")] = "",
):
    """Register a custom trusted source."""
    from ..core.config import cfg
    from ..graph.reputation import ReputationGraph
    c = cfg.override(database_url=db or cfg.database_url)
    _bootstrap(c.database_url)
    g = ReputationGraph(config=c)
    g.register_custom_source(url, initial_score=score)
    console.print(f"[green]Registered {url} with score {score}[/green]")


@app.command()
def source_blacklist(
    url: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason", "-r")] = "Manual blacklist",
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL")] = "",
):
    """Manually blacklist a source."""
    from ..core.config import cfg
    from ..graph.reputation import ReputationGraph
    c = cfg.override(database_url=db or cfg.database_url)
    _bootstrap(c.database_url)
    g = ReputationGraph(config=c)
    g.blacklist_manual(url, reason)
    console.print(f"[red]Blacklisted {url}[/red]")


# ── graph ─────────────────────────────────────────────────────────────────────

@app.command()
def graph_scores(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 30,
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL")] = "",
):
    """Show current reputation scores for all sources."""
    from ..core.config import cfg
    from ..graph.reputation import ReputationGraph
    c = cfg.override(database_url=db or cfg.database_url)
    _bootstrap(c.database_url)
    g = ReputationGraph(config=c)
    scores = g.get_all_scores()
    if not scores:
        console.print("[dim]No sources tracked yet.[/dim]")
        return
    table = Table(title="Reputation Graph — Source Scores")
    table.add_column("Domain", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Blacklisted", justify="center")
    for s in scores[:limit]:
        bl = "[red]x[/]" if s["blacklisted"] else "[green]ok[/]"
        sc = s["score"]
        color = "green" if sc >= 70 else ("yellow" if sc >= 40 else "red")
        table.add_row(s["domain"], f"[{color}]{sc:.1f}[/]", bl)
    console.print(table)


@app.command()
def chat(
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="claude | openai | gemini")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Override model name")] = None,
    db: Annotated[str, typer.Option("--db", envvar="DATABASE_URL", help="PostgreSQL URL")] = "",
):
    """Start an interactive research session (Claude-style)."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style
    from ..core.config import cfg
    from ..agents.pipeline import run_pipeline
    from ..graph.reputation import ReputationGraph
    from ..vault.manager import VaultManager

    c = cfg.override(
        default_provider=provider or cfg.default_provider,
        database_url=db or cfg.database_url,
    )
    _bootstrap(c.database_url)

    vault = VaultManager(c.vault_local_path, c.cloud_sync_url, c.vault_auto_sync)
    graph = ReputationGraph(vault=vault, config=c)

    style = Style.from_dict({
        "prompt": "bold cyan",
    })
    session = PromptSession(style=style)

    console.print(Panel("[bold cyan]Deep Research Interactive Shell[/]\n[dim]Type 'exit' or 'quit' to end session.[/dim]", expand=False))

    while True:
        try:
            query = session.prompt("deep-research > ")
            if not query.strip():
                continue
            if query.lower() in ("exit", "quit"):
                break

            ctx = run_pipeline(
                query=query,
                provider=provider,
                model=model or None,
                graph=graph,
                vault=vault,
                config=c,
            )

            console.print("\n" + "─" * 60)
            console.print(Markdown(ctx.report_md))
            console.print("\n" + "─" * 60)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[yellow]Goodbye![/yellow]")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Default to chat mode if no command is provided."""
    if ctx.invoked_subcommand is None:
        # Check if we are in a TTY before defaulting to chat
        import sys
        if sys.stdin.isatty():
            chat()
        else:
            console.print(ctx.get_help())


def main():
    app()


if __name__ == "__main__":
    main()
