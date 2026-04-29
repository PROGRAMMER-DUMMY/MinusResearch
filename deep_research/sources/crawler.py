"""
deep_research/sources/crawler.py
Deep Domain Crawler using Firecrawl for Markdown extraction.
"""
from __future__ import annotations
import requests
from typing import List, Optional

from ..core.config import cfg
from ..core.llm import complete


class GuidedCrawler:
    def __init__(self, config=None, provider: str = None, model: str = None):
        self.c = config or cfg
        self.api_key = self.c.firecrawl_api_key
        self.provider = provider
        self.model = model

    def crawl(self, url: str, max_depth: int = 2) -> Optional[str]:
        """
        Uses Firecrawl to extract clean Markdown from a URL.
        Falls back to a standard requests fetch if Firecrawl API key is missing.
        """
        if not self.api_key:
            return self._fallback_crawl(url)
            
        try:
            print(f"[Crawler] Fetching deep markdown via Firecrawl: {url}")
            resp = requests.post(
                "https://api.firecrawl.dev/v0/scrape",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"url": url},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", {}).get("markdown", "")
        except Exception as e:
            print(f"[Crawler] Firecrawl failed for {url}: {e}")
            return self._fallback_crawl(url)

    def _fallback_crawl(self, url: str) -> Optional[str]:
        """Basic request fallback when Firecrawl fails or is unconfigured."""
        try:
            resp = requests.get(
                url, 
                timeout=15, 
                headers={"User-Agent": "DeepResearchBot/1.0"}
            )
            resp.raise_for_status()
            # In a real scenario, we might use BeautifulSoup to strip tags here
            return resp.text
        except Exception as e:
            print(f"[Crawler] Fallback fetch failed for {url}: {e}")
            return None

    def semantic_route(self, query: str, links: List[str]) -> List[str]:
        """
        Filters a list of links to return ONLY the ones semantically relevant 
        to the user's research query.
        """
        if not links:
            return []
            
        system = (
            "You are a routing agent for a web crawler. "
            "You will be given a RESEARCH QUERY and a list of URLs. "
            "Return ONLY the URLs that are highly likely to contain information "
            "relevant to the query. Ignore navigation, terms of service, etc. "
            "Output your chosen URLs one per line, nothing else."
        )
        
        prompt = f"RESEARCH QUERY: {query}\n\nLINKS:\n" + "\n".join(links[:50])
        
        try:
            result = complete(
                prompt=prompt,
                system=system,
                provider=self.provider,
                model=self.model,
                max_tokens=1000,
                temperature=0.0
            )
            
            chosen = [line.strip() for line in result.split("\n") if line.strip() in links]
            return chosen
        except Exception as e:
            print(f"[Crawler] Semantic routing failed: {e}")
            return links[:5]  # Fallback: just return first 5 links
