"""
deep_research/agents/extractor.py
Context Compressor Agent (Summarizer):
Reads raw web text chunks and distills *only* the facts relevant to the query.
This prevents LLM context bloat and saves token costs.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from ..core.llm import complete
from ..vault.interface import ExtractedFact


class ExtractorAgent:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider
        self.model = model
        
    def distill(self, url: str, raw_content: str, query: str) -> ExtractedFact:
        """
        Takes raw scraped HTML/Markdown, asks the LLM to extract only facts relevant
        to the 'query', and returns an ExtractedFact.
        """
        system_prompt = (
            "You are an expert fact-extractor and context compressor. "
            "Your job is to read raw text from a source and extract ONLY the facts, "
            "quotes, and statistics that are directly relevant to the user's research query. "
            "Ignore all boilerplate, navigation, ads, and irrelevant tangents. "
            "Return the compressed information as a dense bulleted list."
        )
        
        prompt = f"RESEARCH QUERY: {query}\n\nRAW SOURCE TEXT:\n{raw_content[:20000]}"
        
        try:
            # Call the LLM to summarize/distill
            summary = complete(
                prompt=prompt,
                system=system_prompt,
                provider=self.provider,
                model=self.model,
                max_tokens=1500,
                temperature=0.1
            )
        except Exception as e:
            print(f"[Extractor] LLM compression failed for {url}: {e}")
            summary = raw_content[:1500]  # Fallback to pure truncation
            
        # Optional: Generate Vector Embeddings here if an embedding model is configured.
        # embedding = generate_embedding(summary)
        embedding = None
            
        return ExtractedFact(
            url=url,
            raw_content=raw_content,
            summary=summary,
            embedding=embedding
        )
