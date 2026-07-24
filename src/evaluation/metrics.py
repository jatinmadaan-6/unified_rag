import json
from typing import List, Dict, Set
from src.utils.prompts import RESPONSE_EVAL_PROMPT

def calculate_recall_at_k(retrieved_units: List[str], gold_units: Set[str], k_values: List[int]) -> Dict[str, float]:
    """Calculates Recall@k for a given query."""
    results = {}
    for k in k_values:
        topk = set(retrieved_units[:k])
        hit = len(topk & gold_units)
        results[f"R@{k}"] = hit / max(len(gold_units), 1)
    return results

def evaluate_response_llm_as_judge(llm, question: str, reference_answer: str, response_a: str, response_b: str) -> Dict[str, str]:
    """Uses the local LLM to evaluate responses based on Specificity, Conciseness, and Completeness."""
    prompt = RESPONSE_EVAL_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        response_a=response_a,
        response_b=response_b
    )
    
    raw = llm.generate(prompt)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"raw": raw}
