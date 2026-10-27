"""Local, dependency-free cluster labels and summaries.

No API client is used here: internal-node summaries use the project's LocalLLM.
"""
from typing import List, Tuple


SUMMARY_PROMPT = """Summarize this technical-document cluster in two concise sentences.
State its common topic, important APIs/components, and dependencies. Do not invent facts.

Excerpts:
{excerpts}

Summary:"""


class ClusterSummarizer:
    def __init__(self, llm=None, max_excerpts: int = 12, max_chars_each: int = 500):
        self.llm = llm
        self.max_excerpts = max_excerpts
        self.max_chars_each = max_chars_each

    def summarize_cluster(self, texts: List[str], is_leaf: bool) -> Tuple[str, List[str]]:
        keywords = self._extract_keyphrases(texts)
        if is_leaf:
            return "; ".join(keywords[:8]) if keywords else (texts[0][:200] if texts else ""), keywords
        if self.llm is None:
            return "; ".join(keywords[:12]), keywords
        excerpts = "\n---\n".join(t[:self.max_chars_each] for t in texts[:self.max_excerpts])
        return self.llm.generate(SUMMARY_PROMPT.format(excerpts=excerpts), max_new_tokens=150), keywords

    @staticmethod
    def _extract_keyphrases(texts: List[str], top_n: int = 10) -> List[str]:
        try:
            import yake
            extractor = yake.KeywordExtractor(top=top_n)
            return [keyword for keyword, _ in extractor.extract_keywords(" ".join(texts)[:5000])]
        except Exception:
            words = [w.strip(".,:;()[]{}\n\t").lower() for w in " ".join(texts).split()]
            seen = []
            for word in words:
                if len(word) > 4 and word.isalnum() and word not in seen:
                    seen.append(word)
            return seen[:top_n]
