"""
deep_research/graph/reputation.py
Source Reputation Graph — scoring, reward/penalty, blacklist pipeline.
Works in-memory when no database is configured; persists to PostgreSQL when available.
"""
from __future__ import annotations
from urllib.parse import urlparse
from typing import Optional, Callable, List, Dict
import networkx as nx

from ..core.config import cfg
from ..vault.interface import VaultInterface


class ReputationGraph:
    """
    Manages source trust scores (0–100). Each source is a graph node.
    All methods work through the VaultInterface to ensure thread-safety and abstraction.
    """

    def __init__(self, vault: VaultInterface, config=None, warn_cb: Optional[Callable] = None):
        self.cfg = config or cfg
        self.vault = vault
        self.G = nx.DiGraph()
        self.warn_cb = warn_cb or self._default_warn
        self.pending_updates: List[Dict] = []
        self._load_from_vault()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_from_vault(self):
        try:
            scores = self.vault.get_all_scores()
            for s in scores:
                self.G.add_node(s["domain"], score=s["score"], blacklisted=s["blacklisted"])
        except Exception as e:
            print(f"[Graph] Failed to load from vault: {e}")

    # ── Score helpers ─────────────────────────────────────────────────────────

    def _tld_bonus(self, domain: str) -> float:
        for tld, bonus in self.cfg.trust_seed.items():
            if tld == "default":
                continue
            if domain.endswith(tld):
                return float(bonus)
        return float(self.cfg.trust_seed.get("default", 40))

    def _ensure_node(self, domain: str) -> float:
        if domain not in self.G:
            # Check vault first
            src = self.vault.get_source(domain)
            if src:
                self.G.add_node(domain, score=src.score, blacklisted=src.blacklisted)
                return src.score
            
            # Not in vault, use TLD bonus
            initial = self._tld_bonus(domain)
            self.G.add_node(domain, score=initial, blacklisted=False)
            return initial
        return self.G.nodes[domain]["score"]

    def score(self, url: str) -> float:
        domain = urlparse(url).netloc or url
        return self._ensure_node(domain)

    # ── Reward / Penalty ──────────────────────────────────────────────────────

    def adjust(self, url: str, delta: float, reason: str, run_id: str = ""):
        """Apply a score delta. Updates in-memory and adds to pending_updates."""
        domain = urlparse(url).netloc or url
        current = self._ensure_node(domain)

        new_score = max(0.0, min(100.0, current + delta))
        self.G.nodes[domain]["score"] = new_score

        # Queue for atomic batch save
        self.pending_updates.append({
            "url": url,
            "delta": delta,
            "reason": reason,
            "run_id": run_id
        })

    def reward(self, url: str, points: float, reason: str, run_id: str = ""):
        self.adjust(url, +abs(points), reason, run_id)

    def penalize(self, url: str, points: float, reason: str, run_id: str = ""):
        self.adjust(url, -abs(points), reason, run_id)

    # ── Blacklist ─────────────────────────────────────────────────────────────

    def blacklist_manual(self, url: str, reason: str = "Manual blacklist"):
        domain = urlparse(url).netloc or url
        self._ensure_node(domain)
        self.G.nodes[domain]["blacklisted"] = True
        # We also want this to be semi-atomic, but manual ones can hit vault directly
        self.vault.update_reputation(url, -100.0, reason)

    def is_blacklisted(self, url: str) -> bool:
        domain = urlparse(url).netloc or url
        if domain in self.G:
            return self.G.nodes[domain].get("blacklisted", False)
        # Check vault if not in memory
        src = self.vault.get_source(domain)
        if src:
            self.G.add_node(domain, score=src.score, blacklisted=src.blacklisted)
            return src.blacklisted
        return False

    # ── Co-citation edges ─────────────────────────────────────────────────────

    def add_cocitation(self, url_a: str, url_b: str):
        da = urlparse(url_a).netloc or url_a
        db_domain = urlparse(url_b).netloc or url_b
        if not self.G.has_edge(da, db_domain):
            self.G.add_edge(da, db_domain, weight=1)
        else:
            self.G[da][db_domain]["weight"] += 1

    # ── Custom sources ────────────────────────────────────────────────────────

    def register_custom_source(self, url: str, initial_score: float = 60.0):
        domain = urlparse(url).netloc or url
        self.G.add_node(domain, score=initial_score, blacklisted=False, custom=True)
        # This is a config change, hit vault directly
        self.vault.update_reputation(url, initial_score - self._tld_bonus(domain), "Manual registration")

    def get_all_scores(self) -> list[dict]:
        """Return all in-memory scores."""
        return [
            {
                "domain": d,
                "score": data.get("score", 0),
                "blacklisted": data.get("blacklisted", False),
            }
            for d, data in self.G.nodes(data=True)
        ]

    # ── Default warning (CLI) ─────────────────────────────────────────────────

    @staticmethod
    def _default_warn(domain: str, score: float, notif_id: int) -> str:
        print(f"\n WARNING: '{domain}' has low trust score ({score:.1f}/100).")
        print("   This source may be unreliable. Blacklist it? [y/N]: ", end="")
        ans = input().strip().lower()
        return "confirmed" if ans == "y" else "dismissed"
