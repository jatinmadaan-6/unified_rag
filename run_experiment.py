"""Reproducible, fully local evaluation for the unified RAG framework."""
import argparse
import copy
import json
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import yaml

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import get_embedder
from src.evaluation.metrics import aggregate_pairwise_judgments, calculate_ranking_metrics, evaluate_response_llm_as_judge, mean_metrics
from src.ingestion.local_loader import load_local_documents
from src.llm.generator import get_llm
from src.retrieval.cluster_rag import ClusterRAG
from src.retrieval.hsg_rag import HSGRAG
from src.retrieval.raptor_rag import RaptorRAG
from src.retrieval.vanilla_rag import VanillaRAG
from src.utils.prompts import ANSWER_PROMPT

METHODS = ("vanilla", "raptor", "hsg", "flat_cluster", "cluster", "cluster_no_siblings")


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_qa_pairs(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def qa_slice(qa):
    hop = qa.get("hop_count")
    hop_label = f"{hop}_hop" if hop is not None else ("multi_hop" if qa.get("cross_cluster") else "single_hop")
    return {"hop": hop_label, "question_type": qa.get("question_type", "unknown"), "difficulty": qa.get("difficulty", "unknown")}


def build_retriever(method, llm, embedder, cfg):
    if method == "vanilla":
        return VanillaRAG(embedder), "O(N)"
    if method == "hsg":
        return HSGRAG(llm, embedder, cfg["retrieval"]["hsg_rag"]), "O(V + E) per document"
    if method == "raptor":
        return RaptorRAG(llm, embedder, cfg["retrieval"]["raptor"]), "Recursive summary-tree indexing"
    if method == "flat_cluster":
        return ClusterRAG(llm, embedder, cfg["retrieval"]["flat_cluster_rag"]), "O(NK) flat clustering"
    cluster_cfg = copy.deepcopy(cfg["retrieval"]["cluster_rag"])
    if method == "cluster_no_siblings":
        cluster_cfg["use_sibling_expansion"] = False
    return ClusterRAG(llm, embedder, cluster_cfg), "O(N log K) hierarchical clustering"


def answer_question(llm, question, retrieved, max_new_tokens):
    context = "\n\n---\n\n".join(item["text"] for item in retrieved)
    return llm.generate(ANSWER_PROMPT.format(context=context, question=question), max_new_tokens=max_new_tokens)


def format_table(results, k_values):
    header = " | ".join(["Method"] + [f"R@{k}" for k in k_values] + ["MRR", "nDCG@10", "Query ms", "Index s"])
    print("\n" + header + "\n" + "-" * len(header))
    for method, result in results.items():
        retrieval, efficiency = result.get("retrieval", {}), result.get("efficiency", {})
        ndcg = retrieval.get("nDCG@10", retrieval.get(f"nDCG@{k_values[-1]}", 0))
        cells = [method] + [f"{retrieval.get(f'R@{k}', 0):.4f}" for k in k_values]
        cells += [f"{retrieval.get('MRR', 0):.4f}", f"{ndcg:.4f}", f"{efficiency.get('mean_query_latency_ms', 0):.2f}", f"{efficiency.get('index_time_sec', 0):.2f}"]
        print(" | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Fully local Unified RAG evaluation; no API key required.")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--qa_dataset", required=True)
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--skip_answers", action="store_true", help="Run retrieval and efficiency metrics only.")
    args = parser.parse_args()

    cfg, qa_pairs = load_config(args.config), load_qa_pairs(args.qa_dataset)
    docs = load_local_documents(cfg["data"]["raw_dir"])
    chunks = chunk_documents(docs, cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"])
    if not docs or not qa_pairs:
        raise ValueError("Both raw documents and a non-empty QA JSONL dataset are required.")
    print(f"Loaded {len(docs)} documents, {len(chunks)} chunks, and {len(qa_pairs)} QA pairs.")
    selected = list(METHODS) if args.method == "all" else [args.method]
    embedder = get_embedder(cfg["model"]["embedding"]["name"])
    needs_llm = any(method != "vanilla" for method in selected) or (cfg["evaluation"].get("evaluate_answers", True) and not args.skip_answers)
    llm = get_llm(cfg["model"]["llm"]["name"], cfg["model"]["llm"]["load_in_4bit"]) if needs_llm else None
    k_values, max_k = cfg["evaluation"]["k_values"], max(cfg["evaluation"]["k_values"])
    results, answers = {}, defaultdict(dict)

    for method in selected:
        print(f"\n--- {method} ---")
        retriever, complexity = build_retriever(method, llm, embedder, cfg)
        tracemalloc.start(); started = time.perf_counter()
        retriever.index(chunks)
        index_time = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows, by_slice, latencies, answer_inputs = [], defaultdict(list), [], []
        for qa in qa_pairs:
            gold = set(qa.get("supporting_chunk_ids") or qa.get("chunk_ids", []))
            if not gold:
                continue
            started = time.perf_counter()
            retrieved = retriever.retrieve(qa.get("question", ""), top_k=max_k)
            latencies.append((time.perf_counter() - started) * 1000)
            row = calculate_ranking_metrics([item["unit_id"] for item in retrieved], gold, k_values)
            rows.append(row)
            for dimension, value in qa_slice(qa).items():
                by_slice[f"{dimension}:{value}"].append(row)
            answer_inputs.append((qa, retrieved))
        results[method] = {"retrieval": mean_metrics(rows), "slices": {label: mean_metrics(values) for label, values in sorted(by_slice.items())}, "efficiency": {"index_time_sec": index_time, "peak_memory_mb": peak / 1_000_000, "mean_query_latency_ms": sum(latencies) / max(len(latencies), 1), "p95_query_latency_ms": sorted(latencies)[max(0, int(len(latencies) * .95) - 1)] if latencies else 0.0, "complexity": complexity}}
        if cfg["evaluation"].get("evaluate_answers", True) and not args.skip_answers:
            for qa, retrieved in answer_inputs[:cfg["evaluation"].get("max_answer_questions", 100)]:
                answers[qa.get("id", qa["question"])][method] = answer_question(llm, qa["question"], retrieved, cfg["evaluation"].get("answer_max_new_tokens", 384))

    if answers and len(selected) > 1:
        for method_a in selected:
            comparisons = {}
            for method_b in selected:
                if method_a == method_b:
                    continue
                judgments = []
                for qa in qa_pairs:
                    key = qa.get("id", qa["question"])
                    if method_a in answers[key] and method_b in answers[key]:
                        judgments.append(evaluate_response_llm_as_judge(llm, qa["question"], qa.get("reference_answer", ""), answers[key][method_a], answers[key][method_b]))
                comparisons[f"vs_{method_b}"] = aggregate_pairwise_judgments(judgments)
            results[method_a]["answer_quality"] = comparisons
    results["local_model_usage"] = llm.stats_snapshot() if llm is not None else {"calls": 0, "note": "No LLM was needed for this run."}
    results["metadata"] = {"qa_pairs": len(qa_pairs), "methods": selected, "hosted_api_required": False}
    format_table({method: results[method] for method in selected}, k_values)
    Path(args.results).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved full results to {args.results}")


if __name__ == "__main__":
    main()
