# Unified RAG Research Framework

This repository contains the codebase for evaluating embedding-based Cluster-RAG against Vanilla RAG and HSG-RAG. 

The framework is completely local and does not require paid API keys. It uses HuggingFace models for language generation/extraction and `sentence-transformers` for embeddings.

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
Run the unified evaluation pipeline which will index documents, perform retrieval across the different methods, calculate Recall@k, and generate a comparison table.

```bash
python run_experiment.py --method all --qa_dataset data/evaluation/qa_pairs.jsonl
```

You can also run specific methods:
```bash
python run_experiment.py --method hsg --qa_dataset data/evaluation/qa_pairs.jsonl
```
Valid methods: `vanilla`, `hsg`, `cluster`, `all`.

## Results
The experiment script outputs a table comparing the methods and saves detailed metrics to `results.json`.
