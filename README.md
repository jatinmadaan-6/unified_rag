# Unified RAG Research Framework

This repository contains the codebase for evaluating embedding-based Cluster-RAG against Vanilla RAG and HSG-RAG. 

The framework is completely local and does not require paid API keys or an `OPENAI_API_KEY`. It uses Hugging Face models for generation, local LLM judging, and HSG-style extraction, plus `sentence-transformers` for embeddings.

## Requirements

The codebase expects a GPU-enabled environment (preferably 80GB VRAM if using the default Qwen2.5-32B model, or you can configure a smaller model in `configs/config.yaml`).

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
python src/generation/pipeline.py --folder data/raw --output data/evaluation/qa_pairs.jsonl
```

### 2. Running Experiments
Run the unified evaluation pipeline. By default it compares Vanilla RAG, a local RAPTOR-style recursive summary-tree baseline, HSG-RAG, flat clustering, hierarchical Cluster-RAG, and Cluster-RAG without sibling expansion.

```bash
python run_experiment.py --method all --qa_dataset data/evaluation/qa_pairs.jsonl
```

For a quick retrieval-only pass (skips local answer generation and LLM judging):

```bash
python run_experiment.py --method all --qa_dataset data/evaluation/qa_pairs.jsonl --skip_answers
```

You can also run specific methods:
```bash
python run_experiment.py --method hsg --qa_dataset data/evaluation/qa_pairs.jsonl
```
Valid methods: `vanilla`, `raptor`, `hsg`, `flat_cluster`, `cluster`, `cluster_no_siblings`, `all`.

The `raptor` method is an in-repository RAPTOR-style baseline: it recursively clusters chunks, summarizes and re-embeds each cluster, then performs collapsed-tree retrieval before returning original leaf evidence. It is not represented as the authors' exact UMAP+GMM implementation; retain that distinction in any paper.

## Results
The experiment script outputs a table and saves detailed metrics to `results.json`: Recall@k, Precision@k, Hit@k, MRR, nDCG@k, mean/p95 query latency, indexing time, tracked peak memory, and QA slices by hop label, question type, and difficulty. When answer evaluation is enabled, it also runs local pairwise LLM judging for specificity, conciseness, and completeness. `results.json` records local model call/token/time totals; it does not estimate API cost because no hosted API is used.
