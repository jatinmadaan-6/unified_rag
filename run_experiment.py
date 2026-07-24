import argparse
import yaml
import time
import json
import tracemalloc
from typing import Dict, Any

from src.ingestion.local_loader import load_local_documents
from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import get_embedder
from src.llm.generator import get_llm
from src.retrieval.vanilla_rag import VanillaRAG
from src.retrieval.hsg_rag import HSGRAG
from src.retrieval.cluster_rag import ClusterRAG
from src.evaluation.metrics import calculate_recall_at_k, evaluate_response_llm_as_judge
from src.utils.prompts import ANSWER_PROMPT

def load_config(path="configs/config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)

def format_table(results):
    print(f"\n{'='*80}")
    print(f"{'Method':<15} | {'Recall@5':<10} | {'Recall@7':<10} | {'Recall@10':<10} | {'Recall@20':<10}")
    print(f"{'-'*80}")
    for method, res in results.items():
        r = res.get('retrieval', {})
        print(f"{method:<15} | {r.get('R@5', 0):<10.4f} | {r.get('R@7', 0):<10.4f} | {r.get('R@10', 0):<10.4f} | {r.get('R@20', 0):<10.4f}")
    
    print(f"\n{'='*80}")
    print(f"{'Method':<15} | {'Index Time (s)':<15} | {'Memory (MB)':<15} | {'Complexity (approx)':<20}")
    print(f"{'-'*80}")
    for method, res in results.items():
        eff = res.get('efficiency', {})
        print(f"{method:<15} | {eff.get('index_time_sec', 0):<15.2f} | {eff.get('peak_memory_mb', 0):<15.2f} | {eff.get('complexity', 'O(N)'):<20}")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Unified RAG Evaluation Framework")
    parser.add_argument("--method", type=str, choices=["vanilla", "hsg", "cluster", "all"], default="all")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--qa_dataset", type=str, help="Path to QA dataset JSONL", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # 1. Load Data
    print("Loading documents...")
    docs = load_local_documents(cfg['data']['raw_dir'])
    chunks = chunk_documents(docs, chunk_size=cfg['chunking']['chunk_size'], chunk_overlap=cfg['chunking']['chunk_overlap'])
    print(f"Loaded {len(docs)} documents, chunked into {len(chunks)} units.")

    # 2. Init Models
    print("Initializing local models...")
    embedder = get_embedder(cfg['model']['embedding']['name'])
    llm = get_llm(cfg['model']['llm']['name'], cfg['model']['llm']['load_in_4bit'])

    methods_to_run = ["vanilla", "hsg", "cluster"] if args.method == "all" else [args.method]
    results = {}

    for method in methods_to_run:
        print(f"\n--- Running {method.upper()} RAG ---")
        tracemalloc.start()
        start_time = time.time()
        
        if method == "vanilla":
            retriever = VanillaRAG(embedder)
            complexity = "O(N)"
        elif method == "hsg":
            retriever = HSGRAG(llm, embedder, cfg['retrieval']['hsg_rag'])
            complexity = "O(V + E) per doc"
        elif method == "cluster":
            retriever = ClusterRAG(llm, embedder, cfg['retrieval']['cluster_rag'])
            complexity = "O(N log K)"
            
        print("Indexing...")
        retriever.index(chunks)
        index_time = time.time() - start_time
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results[method] = {
            "efficiency": {
                "index_time_sec": index_time,
                "peak_memory_mb": peak / 10**6,
                "complexity": complexity
            }
        }
        print(f"Indexed in {index_time:.2f}s (Peak Memory: {peak / 10**6:.2f}MB)")

        # Load QA Dataset and Evaluate Retrieval
        print("Evaluating Retrieval...")
        try:
            with open(args.qa_dataset, 'r') as f:
                qa_pairs = [json.loads(line) for line in f]
        except Exception as e:
            print(f"Failed to load QA dataset: {e}. Skipping retrieval eval.")
            continue
            
        k_values = cfg['evaluation']['k_values']
        all_hits = {f"R@{k}": [] for k in k_values}
        
        for idx, qa in enumerate(qa_pairs):
            query = qa.get("question", "")
            gold_units = set(qa.get("chunk_ids", []))
            
            if not gold_units: continue
            
            retrieved = retriever.retrieve(query, top_k=max(k_values))
            retrieved_ids = [r["unit_id"] for r in retrieved]
            
            recall_dict = calculate_recall_at_k(retrieved_ids, gold_units, k_values)
            for k in k_values:
                all_hits[f"R@{k}"].append(recall_dict[f"R@{k}"])
                
        results[method]["retrieval"] = {k: sum(v)/len(v) if v else 0.0 for k, v in all_hits.items()}
        print(f"Retrieval Results for {method}: {results[method]['retrieval']}")

    format_table(results)
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results.json")

if __name__ == "__main__":
    main()
