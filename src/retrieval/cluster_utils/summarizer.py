"""
Cluster summarization (Part 3): LLM summarization at internal (non-leaf) nodes only,
YAKE keyphrase extraction (no LLM call) at leaf nodes and for human-readable labels.

Scoping the LLM calls to internal nodes only keeps the number of summarization calls
roughly n_chunks / avg_branching_factor -- report this against HSG-RAG's Table 5
overhead (extraction + reflection + entity-linking) for an honest cost comparison.
"""
from typing import List, Tuple
import os


SUMMARY_PROMPT = """You are an expert in embedded system documentation.
Given the following related text excerpts (all belonging to the same topical cluster),
write ONE dense paragraph (2-3 sentences) that summarizes what this cluster is about.
Do not enumerate each excerpt separately; synthesize a single coherent description.

Excerpts:
{excerpts}

Summary:"""


class ClusterSummarizer:
    def __init__(self, model: str = "gpt-4o-mini", max_excerpts: int = 12, max_chars_each: int = 500):
        self.model = model
        self.max_excerpts = max_excerpts
        self.max_chars_each = max_chars_each
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def summarize_cluster(self, texts: List[str], is_leaf: bool) -> Tuple[str, List[str]]:
        keywords = self._extract_keyphrases(texts)
        if is_leaf:
            # Part 3: leaf clusters use concatenated top-keyphrases, no LLM call.
            summary = "; ".join(keywords[:8]) if keywords else (texts[0][:200] if texts else "")
            return summary, keywords
        summary = self._llm_summarize(texts)
        return summary, keywords

    def _extract_keyphrases(self, texts: List[str], top_n: int = 10) -> List[str]:
        try:
            import yake
            extractor = yake.KeywordExtractor(top=top_n)
            joined = " ".join(texts)[:5000]
            kws = extractor.extract_keywords(joined)
            return [kw for kw, _score in sorted(kws, key=lambda x: x[1])]
        except Exception:
            return []

    def _llm_summarize(self, texts: List[str]) -> str:
        excerpts = texts[: self.max_excerpts]
        excerpts = [t[: self.max_chars_each] for t in excerpts]
        prompt = SUMMARY_PROMPT.format(excerpts="\n---\n".join(excerpts))
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
