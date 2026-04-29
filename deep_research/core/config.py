"""
deep_research/core/config.py
Loads .env, applies runtime overrides, exposes typed settings.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=False)   # .env is baseline; env vars already set win


@dataclass
class Config:
    # ── LLM ──────────────────────────────────────────────────
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str    = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str    = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    default_provider: str  = field(default_factory=lambda: os.getenv("DEFAULT_PROVIDER", "claude"))
    default_model: str     = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", ""))

    # ── Search ────────────────────────────────────────────────
    tavily_api_key: str  = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    serper_api_key: str  = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    firecrawl_api_key: str = field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY", ""))

    # ── DB ────────────────────────────────────────────────────
    database_url: str    = field(default_factory=lambda: os.getenv("DATABASE_URL") or "sqlite:///./vault/vault.db")
    cloud_sync_url: str  = field(default_factory=lambda: os.getenv("CLOUD_SYNC_URL", ""))

    # ── Vault ─────────────────────────────────────────────────
    vault_local_path: Path = field(default_factory=lambda: Path(os.getenv("VAULT_LOCAL_PATH", "./vault")))
    vault_auto_sync: bool  = field(default_factory=lambda: os.getenv("VAULT_AUTO_SYNC", "false").lower() == "true")

    # ── Reputation ────────────────────────────────────────────
    blacklist_warn_threshold: int  = field(default_factory=lambda: int(os.getenv("BLACKLIST_WARN_THRESHOLD", "20")))
    blacklist_auto_threshold: int  = field(default_factory=lambda: int(os.getenv("BLACKLIST_AUTO_THRESHOLD", "10")))

    # ── Trust seeds ───────────────────────────────────────────
    trust_seed: dict[str, int] = field(default_factory=lambda: {
        ".edu": int(os.getenv("TRUST_SEED_EDU", "90")),
        ".gov": int(os.getenv("TRUST_SEED_GOV", "85")),
        ".org": int(os.getenv("TRUST_SEED_ORG", "70")),
        ".com": int(os.getenv("TRUST_SEED_COM", "50")),
        "default": int(os.getenv("TRUST_SEED_DEFAULT", "40")),
    })

    def override(self, **kwargs) -> "Config":
        """Return a new Config with runtime overrides applied."""
        import copy
        c = copy.copy(self)
        for k, v in kwargs.items():
            if v is not None and hasattr(c, k):
                setattr(c, k, v)
        return c

    def resolve_model(self, provider: str | None = None) -> tuple[str, str]:
        """Return (provider, model) respecting override > default > sensible fallback."""
        p = (provider or self.default_provider or "claude").lower()
        defaults = {
            "claude":     "claude-sonnet-4-6",
            "openai":     "gpt-4o",
            "gemini":     "gemini-1.5-pro",
            "claude-cli": "",   # CLI handles model selection unless overridden
            "gemini-cli": "",
            "codex":      "",
        }
        m = self.default_model or defaults.get(p, "")
        return p, m


# Singleton — import and use directly
cfg = Config()
