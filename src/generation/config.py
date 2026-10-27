"""
Central configuration for the Cluster-RAG benchmark QA generation
pipeline — 100% free, Colab friendly, no API keys and no credential
files.

Just set DATA_FOLDER to a local path (typically a mounted Google Drive
folder inside Colab) and run.

NOTE: this pipeline generates a benchmark for evaluating Cluster-RAG,
which itself uses flat semantic (KMeans) clustering for benchmark
generation even though Cluster-RAG's actual retrieval algorithm is
hierarchical/tree-based. That's intentional — benchmark generation stays
on the fast, flat clustering path; nothing here builds a tree, a graph,
or does hierarchical traversal.
"""
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------
# Simple, top-level knobs — the things you actually need to change.
# ---------------------------------------------------------------------
DATA_FOLDER = "/content/drive/MyDrive/RAG_DATASET"
OUTPUT_FILE = "qa_pairs.jsonl"
TARGET_QA_PAIRS = 600


@dataclass
class Config:
    # --- Data location (local filesystem / mounted Google Drive) ---
    # In Colab: run `from google.colab import drive; drive.mount('/content/drive')`
    # first, then point this at a folder under /content/drive/MyDrive/...
    data_folder: str = DATA_FOLDER
    recursive_scan: bool = True
    supported_extensions: tuple = (".pdf", ".docx", ".txt", ".md", ".markdown")

    # --- LLM (local, free, HuggingFace) ---
    # Pick ONE of:
    #   "Qwen/Qwen2.5-3B-Instruct"            (default — best quality/speed balance on a T4)
    #   "mistralai/Mistral-7B-Instruct-v0.3"  (heavier — set load_in_4bit=True on a T4)
    #   "google/flan-t5-base"                 (lowest memory footprint, use if you hit OOM)
    generation_model: str = "Qwen/Qwen2.5-3B-Instruct"
    judge_model: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = False    # set True when using Mistral-7B-Instruct on a T4
    max_new_tokens: int = 400
    generation_temperature: float = 0.85   # sampling temperature for question/answer generation
    generation_top_p: float = 0.95

    # --- Embeddings (local, no API needed) ---
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    max_chunk_tokens: int = 300
    chunk_overlap_tokens: int = 40
    min_chunk_tokens: int = 40

    # --- Clustering (flat / KMeans — intentional design choice for
    # benchmark generation speed, even though Cluster-RAG's retrieval
    # algorithm itself is hierarchical) ---
    clustering_method: str = "kmeans"   # "kmeans" or "hdbscan"
    target_chunks_per_cluster: int = 6  # used to auto-pick k for kmeans
    cross_cluster_top_k: int = 1        # nearest neighbouring cluster(s) per cluster, for multi-hop pairing
    cross_cluster_min_similarity: float = 0.35
    cross_cluster_max_similarity: float = 0.85  # too similar = not really "multi-hop"

    # --- Taxonomy-driven CLUSTER-level generation ---
    # Relative sampling weights (need not sum to 1). Keys here that also
    # exist in taxonomy.TAXONOMY are used for per-cluster question-type
    # sampling; "unanswerable" is used separately as a fraction of chunks
    # for negative-question generation.
    # Keys must match taxonomy.TAXONOMY exactly (12 cluster-level types);
    # "unanswerable" is handled separately (fraction of chunks, not a
    # per-cluster taxonomy draw).
    taxonomy_distribution: Dict[str, float] = field(default_factory=lambda: {
        "factual": 0.16,
        "definition": 0.08,
        "explanation": 0.10,
        "procedure": 0.08,
        "comparison": 0.07,
        "constraint": 0.08,
        "exception": 0.06,
        "scenario": 0.08,
        "troubleshooting": 0.06,
        "cause_effect": 0.08,
        "dependency": 0.05,
        "synthesis": 0.10,
        "unanswerable": 0.10,
    })

    # --- Cluster-based question generation (replaces fixed
    # questions-per-chunk: count now depends on cluster size/richness) ---
    min_questions_per_cluster: int = 2
    max_questions_per_cluster: int = 14
    chunks_per_question: float = 1.2        # ~1 question per this many chunks in a cluster, before clamping
    cluster_context_max_chunks: int = 10    # cap on chunks concatenated into a cluster's context passage
    cluster_context_max_chars: int = 6000   # char budget for that concatenated context

    # --- Multi-hop / cross-cluster generation ---
    multi_hop_chunks_per_side: int = 3      # chunks pulled from each side of a cross-cluster pair

    # --- Difficulty augmentation ---
    paraphrase_fraction: float = 0.3     # fraction of base Qs to also paraphrase (harder, lexical gap)
    compound_fraction: float = 0.15      # fraction of eligible pairs combined into compound questions

    # --- Duplicate removal ---
    dedup_similarity_threshold: float = 0.93   # cosine similarity above which a question is a duplicate

    # --- Round-trip / quality filtering ---
    retrieval_top_k: int = 6
    answer_similarity_threshold: float = 0.58   # calibrated for technical answers and MiniLM embeddings
    use_llm_judge_for_borderline: bool = True
    borderline_band: float = 0.12   # if sim within [threshold-band, threshold+band], ask the LLM judge

    # --- Target dataset size ---
    # Keep generating cluster-based + multi-hop rounds until ~this many
    # VALIDATED pairs remain after validation, dedup, and round-trip
    # filtering (not ~600 generated candidates).
    target_qa_pairs: int = TARGET_QA_PAIRS
    target_tolerance: float = 0.05      # stop once within +/-5% of target
    max_generation_rounds: int = 6      # safety cap on top-up rounds

    # --- Output ---
    output_path: str = OUTPUT_FILE
    save_rejected: bool = True
    rejected_output_path: str = "qa_pairs_rejected.jsonl"

    def validate(self):
        import os
        if not self.data_folder:
            raise ValueError("data_folder must be set (e.g. /content/drive/MyDrive/RAG_DATASET).")
        if not os.path.isdir(self.data_folder):
            raise ValueError(
                f"data_folder does not exist: {self.data_folder}\n"
                f"On Colab, make sure you've run:\n"
                f"  from google.colab import drive\n"
                f"  drive.mount('/content/drive')\n"
                f"and that the folder exists under /content/drive/MyDrive/..."
            )
