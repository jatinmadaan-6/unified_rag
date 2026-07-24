"""
End-to-end orchestrator for the Cluster-RAG benchmark — 100% free, Google
Colab compatible.

Usage (Colab or local):
    python pipeline.py --folder "/content/drive/MyDrive/RAG_DATASET" \
                        --output qa_pairs.jsonl

Or just edit DATA_FOLDER / OUTPUT_FILE at the top of config.py and run
`python pipeline.py` with no arguments.

No API keys. No credentials files. No paid services.

Pipeline stages:
  1. local_loader   -> read files straight off disk / mounted Drive folder      [unchanged]
  2. chunker        -> structure-aware chunking (headings, not a graph)         [unchanged]
  3. clustering     -> embedding + KMeans clustering (flat, relatedness proxy)  [unchanged]
  4. qa_pairs        -> CLUSTER-level generation, augmentation, dedup,
                        round-trip filtering — CHECKPOINTED: every accepted
                        pair is written to disk the instant it's accepted,
                        not batched and written at the end of a round.
  5. write output   -> JSONL of verified QA pairs (+ optional rejected set),
                        appended to incrementally throughout the run

Benchmark generation intentionally stays on this flat clustering path —
Cluster-RAG's actual retrieval algorithm is hierarchical, but that is
untouched here; only the QA generation portion has changed.

CHECKPOINTING / RESUME
-----------------------
Every QA pair is written to cfg.output_path (and rejected ones to
cfg.rejected_output_path) THE MOMENT it's verified — via checkpoint.py's
CheckpointWriter, which appends + flushes + fsyncs on every single write.
Nothing is buffered in memory and dumped at the end.

If Colab disconnects mid-run, just re-run the exact same command. On
startup, run() reads whatever is already on cfg.output_path, seeds the
in-memory dedup/seen-questions state from it, figures out how many
questions per cluster (and how many negatives) are still needed, and
continues from there — it will not lose or duplicate anything already on
disk, and will not restart cluster generation from zero.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List

from config import Config
import local_loader
import chunker
import clustering
import qa_pairs as qap
import checkpoint
from generator import QAGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Local folder -> Cluster-RAG benchmark QA generation pipeline")
    parser.add_argument("--folder", default=cfg.data_folder,
                         help="Path to the folder of documents, e.g. /content/drive/MyDrive/RAG_DATASET")
    parser.add_argument("--output", default=cfg.output_path)
    parser.add_argument("--generation-model", default=cfg.generation_model,
                         help="HF model id, e.g. Qwen/Qwen2.5-3B-Instruct, "
                              "mistralai/Mistral-7B-Instruct-v0.3, or google/flan-t5-base")
    parser.add_argument("--judge-model", default=None,
                         help="Defaults to --generation-model if not set")
    parser.add_argument("--load-in-4bit", action="store_true", default=cfg.load_in_4bit,
                         help="4-bit quantize the model (recommended for Mistral-7B on a T4)")
    parser.add_argument("--target-qa-pairs", type=int, default=cfg.target_qa_pairs,
                         help="Approximate number of VALIDATED QA pairs to reach (default: 600)")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    cfg.data_folder = args.folder
    cfg.output_path = args.output
    cfg.generation_model = args.generation_model
    cfg.judge_model = args.judge_model or args.generation_model
    cfg.load_in_4bit = args.load_in_4bit
    cfg.target_qa_pairs = args.target_qa_pairs
    cfg.recursive_scan = not args.no_recursive
    return cfg


def _load_checkpoint_state(cfg: Config):
    """Read whatever is already on disk from a previous (possibly
    interrupted) run and derive everything needed to resume: how many
    pairs are already accepted, how many are negatives, how many were
    already generated per cluster, and the set of question texts already
    seen (accepted OR rejected) so we don't regenerate/re-verify them."""
    existing_accepted = checkpoint.load_existing_records(cfg.output_path)
    existing_rejected = checkpoint.load_existing_records(cfg.rejected_output_path) if cfg.save_rejected else []

    accepted_count = len(existing_accepted)
    negatives_count = sum(1 for r in existing_accepted if r.get("question_type") == "unanswerable")

    already_by_cluster: Dict[int, int] = {}
    for r in existing_accepted:
        if not r.get("cross_cluster") and len(r.get("cluster_ids", [])) == 1 \
                and r.get("question_type") != "unanswerable":
            cid = r["cluster_ids"][0]
            already_by_cluster[cid] = already_by_cluster.get(cid, 0) + 1

    seen_questions_lower = {r["question"].strip().lower() for r in existing_accepted}
    seen_questions_lower |= {r["question"].strip().lower() for r in existing_rejected}

    all_accepted_questions = [r["question"] for r in existing_accepted]

    if accepted_count:
        logger.info("Resuming from checkpoint: %d QA pairs already on disk in %s "
                     "(%d of them negatives).", accepted_count, cfg.output_path, negatives_count)

    return accepted_count, negatives_count, already_by_cluster, seen_questions_lower, all_accepted_questions


def run(cfg: Config):
    cfg.validate()
    t0 = time.time()

    # ---- 1. Load documents from local folder / mounted Drive ----
    logger.info("Loading documents from %s ...", cfg.data_folder)
    documents = local_loader.load_local_documents(
        folder=cfg.data_folder,
        recursive=cfg.recursive_scan,
        supported_extensions=cfg.supported_extensions,
    )
    if not documents:
        logger.error("No readable documents found in %s. Exiting.", cfg.data_folder)
        sys.exit(1)
    logger.info("Loaded %d documents.", len(documents))

    # ---- 2. Structure-aware chunking ----
    chunks = chunker.chunk_documents(
        documents,
        max_chunk_tokens=cfg.max_chunk_tokens,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
        min_chunk_tokens=cfg.min_chunk_tokens,
    )
    logger.info("Produced %d chunks.", len(chunks))
    if not chunks:
        logger.error("No chunks produced. Exiting.")
        sys.exit(1)

    # ---- 3. Embedding + flat (KMeans) clustering ----
    embeddings = clustering.embed_chunks(chunks, model_name=cfg.embedding_model_name)
    clustering.cluster_chunks(chunks, embeddings, target_chunks_per_cluster=cfg.target_chunks_per_cluster)
    n_clusters = len(set(c.cluster_id for c in chunks))
    logger.info("Clustered chunks into %d clusters.", n_clusters)

    cross_pairs = clustering.find_cross_cluster_pairs(
        chunks, embeddings,
        top_k=cfg.cross_cluster_top_k,
        min_similarity=cfg.cross_cluster_min_similarity,
        max_similarity=cfg.cross_cluster_max_similarity,
    )
    logger.info("Found %d cross-cluster candidate pairs for multi-hop questions.", len(cross_pairs))

    # ---- 4. Generation (free, local HuggingFace model) ----
    generator = QAGenerator(
        generation_model=cfg.generation_model,
        judge_model=cfg.judge_model,
        load_in_4bit=cfg.load_in_4bit,
        max_new_tokens=cfg.max_new_tokens,
        generation_temperature=cfg.generation_temperature,
        generation_top_p=cfg.generation_top_p,
    )

    def embed_fn(texts):
        return embed_fn._model.encode(texts, normalize_embeddings=True)
    from sentence_transformers import SentenceTransformer
    embed_fn._model = SentenceTransformer(cfg.embedding_model_name)

    # ---- Resume state from checkpoint (no-op on a fresh run) ----
    accepted_count, negatives_count, already_by_cluster, seen_questions_lower, existing_questions = \
        _load_checkpoint_state(cfg)

    deduper = qap.StreamingDeduper(embed_fn, threshold=cfg.dedup_similarity_threshold)
    deduper.seed(existing_questions)

    target_negatives = max(1, int(len(chunks) * cfg.taxonomy_distribution.get("unanswerable", 0.1)))
    lower_bound = cfg.target_qa_pairs * (1 - cfg.target_tolerance)
    upper_bound = cfg.target_qa_pairs * (1 + cfg.target_tolerance)

    compound_pairer = qap.CompoundPairer(generator, cfg.compound_fraction)

    # cfg.output_path / cfg.rejected_output_path opened in APPEND mode —
    # this is what makes resuming safe: nothing already on disk is ever
    # touched, only new records are added.
    writer = checkpoint.CheckpointWriter(cfg.output_path)
    rejected_writer = checkpoint.CheckpointWriter(cfg.rejected_output_path) if cfg.save_rejected else None

    def handle_candidate(pair: "qap.QAPair") -> bool:
        """Dedup -> verify -> checkpoint-write ONE candidate, immediately.
        This is the only place pairs get written to disk, and it happens
        the instant a pair is accepted -- never batched. Also tries to
        spin off a paraphrase/compound variant of an accepted pair right
        away, each individually checkpointed the same way. Returns True
        if the original candidate was accepted."""
        nonlocal accepted_count, negatives_count

        key = pair.question.strip().lower()
        if key in seen_questions_lower:
            return False
        seen_questions_lower.add(key)
        if deduper.is_duplicate(pair.question):
            return False

        passed = qap.verify_pair(
            pair, chunks, embeddings, embed_fn, generator,
            retrieval_top_k=cfg.retrieval_top_k,
            similarity_threshold=cfg.answer_similarity_threshold,
            borderline_band=cfg.borderline_band,
            use_llm_judge=cfg.use_llm_judge_for_borderline,
        )
        if not passed:
            if rejected_writer:
                rejected_writer.write(pair.to_debug_dict())
            return False

        deduper.add(pair.question)
        writer.write(pair.to_dict())  # <-- checkpointed the instant it's accepted
        accepted_count += 1
        if pair.question_type == "unanswerable":
            negatives_count += 1

        _try_augment(pair)
        return True

    def _try_augment(base_pair: "qap.QAPair") -> None:
        """Streaming augmentation: immediately attempt a paraphrase and a
        compound-merge off an accepted base pair, each verified and
        checkpointed on the spot instead of waiting for a batch."""
        nonlocal accepted_count

        para = qap.maybe_paraphrase(base_pair, generator, cfg.paraphrase_fraction)
        if para:
            para_key = para.question.strip().lower()
            if para_key not in seen_questions_lower and not deduper.is_duplicate(para.question):
                seen_questions_lower.add(para_key)
                if qap.verify_pair(para, chunks, embeddings, embed_fn, generator,
                                    retrieval_top_k=cfg.retrieval_top_k,
                                    similarity_threshold=cfg.answer_similarity_threshold,
                                    borderline_band=cfg.borderline_band,
                                    use_llm_judge=cfg.use_llm_judge_for_borderline):
                    deduper.add(para.question)
                    writer.write(para.to_dict())
                    accepted_count += 1
                elif rejected_writer:
                    rejected_writer.write(para.to_debug_dict())

        compound = compound_pairer.offer(base_pair)
        if compound:
            compound_key = compound.question.strip().lower()
            if compound_key not in seen_questions_lower and not deduper.is_duplicate(compound.question):
                seen_questions_lower.add(compound_key)
                if qap.verify_pair(compound, chunks, embeddings, embed_fn, generator,
                                    retrieval_top_k=cfg.retrieval_top_k,
                                    similarity_threshold=cfg.answer_similarity_threshold,
                                    borderline_band=cfg.borderline_band,
                                    use_llm_judge=cfg.use_llm_judge_for_borderline):
                    deduper.add(compound.question)
                    writer.write(compound.to_dict())
                    accepted_count += 1
                elif rejected_writer:
                    rejected_writer.write(compound.to_debug_dict())

    try:
        if accepted_count >= lower_bound:
            logger.info("Checkpoint already has %d/%d validated pairs — target already met, nothing to generate.",
                         accepted_count, cfg.target_qa_pairs)
        else:
            for round_idx in range(cfg.max_generation_rounds):
                logger.info("=== Generation round %d/%d === (%d/%d validated so far)",
                            round_idx + 1, cfg.max_generation_rounds, accepted_count, cfg.target_qa_pairs)

                relaxed_chunks_per_question = max(cfg.chunks_per_question - 0.15 * round_idx, 0.4)

                cluster_stream = qap.stream_cluster_based_pairs(
                    chunks, generator, cfg.taxonomy_distribution,
                    min_questions_per_cluster=cfg.min_questions_per_cluster,
                    max_questions_per_cluster=cfg.max_questions_per_cluster,
                    chunks_per_question=relaxed_chunks_per_question,
                    cluster_context_max_chunks=cfg.cluster_context_max_chunks,
                    cluster_context_max_chars=cfg.cluster_context_max_chars,
                    # Only apply the resume-quota reduction on round 0 —
                    # later rounds intentionally ask for a fresh (relaxed)
                    # quota per cluster regardless of round-0 counts.
                    already_generated_by_cluster=already_by_cluster if round_idx == 0 else None,
                )
                for pair in cluster_stream:
                    handle_candidate(pair)
                    if accepted_count >= upper_bound:
                        logger.info("Hit upper bound mid-stream (%d pairs); stopping cluster generation early.",
                                    accepted_count)
                        break

                if round_idx == 0:
                    for pair in qap.stream_multi_hop_pairs(cross_pairs, chunks, generator,
                                                            cfg.multi_hop_chunks_per_side):
                        handle_candidate(pair)

                    negatives_needed = max(0, target_negatives - negatives_count)
                    for pair in qap.stream_unanswerable_pairs(chunks, generator, negatives_needed):
                        handle_candidate(pair)
                        if negatives_count >= target_negatives:
                            break

                logger.info("After round %d: %d validated total toward target of %d.",
                            round_idx + 1, accepted_count, cfg.target_qa_pairs)

                if accepted_count >= lower_bound:
                    logger.info("Reached ~%d validated QA pairs after %d round(s).",
                                cfg.target_qa_pairs, round_idx + 1)
                    break
            else:
                logger.warning(
                    "Stopped after max_generation_rounds=%d with %d validated pairs (target %d). "
                    "Everything generated so far is already safely checkpointed in %s — just re-run "
                    "the same command to keep going, or raise max_generation_rounds/max_questions_per_cluster.",
                    cfg.max_generation_rounds, accepted_count, cfg.target_qa_pairs, cfg.output_path,
                )
    finally:
        # Always close cleanly, even on an interrupted/crashed run, so
        # any final buffered OS-level state is flushed. Every record up
        # to this point was already individually flushed+fsync'd by
        # writer.write() as it happened.
        writer.close()
        if rejected_writer:
            rejected_writer.close()

    elapsed = time.time() - t0
    final_records = checkpoint.load_existing_records(cfg.output_path)
    logger.info("Done in %.1fs. %d QA pairs on disk at %s. Type breakdown: %s",
                elapsed, len(final_records), cfg.output_path, _type_breakdown(final_records))
    return final_records


def _type_breakdown(records: List[dict]) -> dict:
    breakdown = {}
    for r in records:
        t = r.get("question_type", "unknown")
        breakdown[t] = breakdown.get(t, 0) + 1
    return breakdown


if __name__ == "__main__":
    run(parse_args())
