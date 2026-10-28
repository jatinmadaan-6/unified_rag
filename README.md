# Unified RAG Research Framework

This repository contains the codebase for evaluating embedding-based Cluster-RAG against Vanilla RAG, RAPTOR, and HSG-RAG.

The framework is completely local and does not require paid API keys or an `OPENAI_API_KEY`. It uses Hugging Face models for generation, local LLM judging, and HSG-style extraction, plus `sentence-transformers` for embeddings.

## Requirements

The default local model is `Qwen/Qwen2.5-3B-Instruct`. A CUDA GPU is recommended for practical experiment time; CPU execution is supported but may be slow. Configure the generator and embedding model in `configs/config.yaml` before a full run.

## Setup

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Models:**
    Open `configs/config.yaml` to adjust the models if necessary.

## Usage

### 1. Generating the QA Dataset
To generate the evaluation dataset from the raw documents in `data/raw/`:
```bash
python -m src.generation.pipeline --folder data/raw --output data/evaluation/qa_pairs.jsonl
```

### 2. Running Experiments
Run the unified evaluation pipeline. By default it compares Vanilla RAG, an independent RAPTOR reproduction, HSG-RAG, flat clustering, hierarchical Cluster-RAG, and Cluster-RAG without sibling expansion.

```bash
python run_experiment.py --method all --qa_dataset data/evaluation/qa_pairs.jsonl
```

For a quick retrieval-only pass (skips local answer generation and LLM judging):

```bash
python run_experiment.py --method all --qa_dataset data/evaluation/qa_pairs.jsonl --skip_answers
```

Run answer generation and local pairwise judging only after the retrieval-only run is complete and its metrics have been checked:

```bash
python run_experiment.py --method all --qa_dataset data/evaluation/qa_pairs.jsonl --results experiments/results_full.json
```

You can also run specific methods:
```bash
python run_experiment.py --method hsg --qa_dataset data/evaluation/qa_pairs.jsonl
```
Valid methods: `vanilla`, `raptor`, `hsg`, `flat_cluster`, `cluster`, `cluster_no_siblings`, `all`.

The `raptor` method independently reproduces the published RAPTOR construction: global/local UMAP reduction, BIC-selected Gaussian mixtures, soft cluster memberships, recursive LLM summaries and summary re-embedding, then collapsed-tree retrieval. It is not copied from or endorsed by the original authors; document any remaining implementation deviations in the paper's baseline appendix.

## Paper-readiness checklist

Before reporting a result, retain the exact `configs/config.yaml`, the generated QA JSONL, and the corresponding `results.json`. Report the embedding model, local LLM, seed, corpus list, and whether answer judging was enabled. Treat HSG-RAG and RAPTOR as independent reproductions, not official-author code. Do not report synthetic-QA results as human-annotated evaluation; manually audit a random sample and state the audit protocol in the paper.

## Results
The experiment script outputs a table and saves detailed metrics to `results.json`: Recall@k, Precision@k, Hit@k, MRR, nDCG@k, mean/p95 query latency, indexing time, tracked peak memory, and QA slices by hop label, question type, and difficulty. When answer evaluation is enabled, it also runs local pairwise LLM judging for specificity, conciseness, and completeness. `results.json` records local model call/token/time totals; it does not estimate API cost because no hosted API is used.
