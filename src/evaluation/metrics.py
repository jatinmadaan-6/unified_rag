import json
import math
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


def calculate_ranking_metrics(retrieved_units: List[str], gold_units: Set[str], k_values: List[int]) -> Dict[str, float]:
    """Return retrieval metrics for one query using its supporting chunks as relevance labels."""
    results = calculate_recall_at_k(retrieved_units, gold_units, k_values)
    for k in k_values:
        topk = retrieved_units[:k]
        hits = sum(unit in gold_units for unit in topk)
        results[f"P@{k}"] = hits / k if k else 0.0
        results[f"Hit@{k}"] = float(hits > 0)
        dcg = sum(1.0 / math.log2(rank + 2) for rank, unit in enumerate(topk) if unit in gold_units)
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(gold_units))))
        results[f"nDCG@{k}"] = dcg / ideal if ideal else 0.0
    first_rank = next((rank for rank, unit in enumerate(retrieved_units, 1) if unit in gold_units), None)
    results["MRR"] = 1.0 / first_rank if first_rank else 0.0
    return results


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def evaluate_response_llm_as_judge(llm, question: str, reference_answer: str, response_a: str, response_b: str) -> Dict[str, str]:
    """Uses the local LLM to evaluate responses based on Specificity, Conciseness, and Completeness."""
    prompt = RESPONSE_EVAL_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        response_a=response_a,
        response_b=response_b
    )
    
    raw = llm.generate(prompt, max_new_tokens=120)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"raw": raw, "specificity": "Tie", "conciseness": "Tie", "completeness": "Tie"}


def aggregate_pairwise_judgments(judgments: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Aggregate A/B/Tie decisions for all three answer-quality dimensions."""
    dimensions = ("specificity", "conciseness", "completeness")
    output = {}
    for dimension in dimensions:
        counts = {"A": 0, "B": 0, "Tie": 0}
        for judgment in judgments:
            decision = judgment.get(dimension, "Tie")
            counts[decision if decision in counts else "Tie"] += 1
        total = max(sum(counts.values()), 1)
        output[dimension] = {key: value / total for key, value in counts.items()}
    return output
