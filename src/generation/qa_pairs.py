"""
Defines the QAPair record, drives CLUSTER-level taxonomy-guided generation
+ augmentation, and applies the round-trip filter + embedding-based
duplicate removal that act as the quality gate.

CHANGE LOG (adaptation for Cluster-RAG benchmark generation):
  - generate_single_hop_pairs() (per-CHUNK) is replaced by cluster-level
    generation. Each KMeans cluster is treated as one semantic topic; the
    LLM sees the concatenated text of the cluster's most informative
    chunks and the number of questions per cluster scales with cluster
    size instead of a fixed questions-per-chunk constant.
  - Multi-hop generation keeps the existing flat cross-cluster pairing
    mechanism (still no tree traversal / hierarchy) but now pulls several
    relevant chunks from BOTH clusters instead of one representative
    chunk per side.
  - QAPair's serialized schema matches the requested Cluster-RAG
    benchmark format exactly: id, question, reference_answer,
    cluster_ids, supporting_chunk_ids, question_type, difficulty,
    cross_cluster. Internal QC bookkeeping (verified / verification_score
    / verification_method) is kept on the dataclass for filtering logic
    but deliberately left out of to_dict().
  - Added embedding-based duplicate removal (StreamingDeduper /
    deduplicate_pairs). This wasn't present in the originally uploaded
    qa_pairs.py despite being referenced as an existing QC step; added
    here since cluster-level generation (several questions requested per
    topic, with sampling for phrasing diversity) makes near-duplicates
    more likely than the old one-question-per-chunk approach.

CHANGE LOG (checkpointing / streaming, this revision):
  - generate_cluster_based_pairs / generate_multi_hop_pairs /
    generate_unanswerable_pairs (batch, return a List[QAPair]) are now
    thin wrappers around STREAMING generator equivalents
    (stream_cluster_based_pairs / stream_multi_hop_pairs /
    stream_unanswerable_pairs) that `yield` each QAPair the moment it is
    produced. pipeline.py consumes the streaming versions directly so
    every pair can be verified and written to disk immediately instead of
    waiting for a whole batch to finish.
  - round_trip_filter (batch) is now a thin wrapper around verify_pair(),
    which verifies exactly one pair at a time — the unit pipeline.py
    checkpoints on.
  - augment_with_paraphrases / augment_with_compounds (batch, fraction of
    a finished list) are replaced as pipeline.py's primary path by
    maybe_paraphrase() (a per-pair Bernoulli trial) and CompoundPairer (a
    small streaming buffer that merges two eligible pairs into a compound
    the moment a second candidate becomes available), so augmentation
    also gets checkpointed immediately rather than only after a batch.
  - Added StreamingDeduper: an incremental version of deduplicate_pairs()
    that checks/updates one question at a time against a running
    embedding set (which can be seeded from a previous run's checkpoint
    file on resume).
"""
from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from chunker import Chunk
from clustering import CrossClusterPair
from generator import QAGenerator
from taxonomy import TAXONOMY, MULTI_HOP_INSTRUCTION, UNANSWERABLE_INSTRUCTION, pick_applicable_types

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    id: str
    question: str
    reference_answer: str
    cluster_ids: List[int]
    supporting_chunk_ids: List[str]
    question_type: str                 # factual | definition | explanation | procedure | comparison
                                        # | constraint | exception | scenario | troubleshooting
                                        # | cause_effect | dependency | synthesis | multi_hop | unanswerable
    difficulty: str = "base"           # "base" | "paraphrased" | "compound"
    cross_cluster: bool = False
    # --- internal QC bookkeeping, not part of the output schema ---
    verified: bool = False
    verification_score: Optional[float] = None
    verification_method: Optional[str] = None  # "embedding" | "llm_judge" | "n/a (negative)"

    def to_dict(self) -> dict:
        """Serialize to exactly the Cluster-RAG benchmark schema:
        id, question, reference_answer, cluster_ids, supporting_chunk_ids,
        question_type, difficulty, cross_cluster."""
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "cluster_ids": self.cluster_ids,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "cross_cluster": self.cross_cluster,
        }

    def to_debug_dict(self) -> dict:
        """Full dataclass dump, including QC fields — useful for
        inspecting the rejected set, not written to the main output file."""
        return asdict(self)


# ---------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------

def _group_by_cluster(chunks: List[Chunk]) -> Dict[int, List[Chunk]]:
    groups: Dict[int, List[Chunk]] = {}
    for c in chunks:
        groups.setdefault(c.cluster_id, []).append(c)
    return groups


def _build_cluster_context(cluster_chunks: List[Chunk], max_chunks: int,
                            max_chars: int) -> Tuple[str, List[str]]:
    """Build the aggregated context passage for a cluster: the most
    informative chunks (longest-text-first, as a cheap information-density
    proxy) concatenated up to a chunk-count and character budget. Returns
    (concatenated_text, chunk_ids_used)."""
    ranked = sorted(cluster_chunks, key=lambda c: len(c.text), reverse=True)
    selected, used_ids, total_chars = [], [], 0
    for chunk in ranked[:max_chunks]:
        if total_chars >= max_chars:
            break
        selected.append(chunk.text)
        used_ids.append(chunk.chunk_id)
        total_chars += len(chunk.text)
    text = "\n\n".join(selected)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, used_ids


def _questions_for_cluster_size(n_chunks: int, min_q: int, max_q: int,
                                 chunks_per_question: float) -> int:
    """Dynamically size the question count to cluster size/richness:
    small clusters get fewer questions, large clusters get more,
    clamped to [min_q, max_q]. No fixed questions-per-chunk constant."""
    raw = max(1, round(n_chunks / max(chunks_per_question, 0.1)))
    return max(min_q, min(max_q, raw))


def _select_relevant_chunks(cluster_chunks: List[Chunk], anchor: Chunk, k: int) -> Tuple[List[str], List[str]]:
    """Pick up to k chunks from a cluster for multi-hop context: always
    includes the anchor chunk from the cross-cluster pair, plus its
    longest/most-informative cluster-mates, so the LLM gets several
    relevant chunks per side instead of just the one representative
    chunk."""
    others = [c for c in cluster_chunks if c.chunk_id != anchor.chunk_id]
    others_ranked = sorted(others, key=lambda c: len(c.text), reverse=True)
    selected = [anchor] + others_ranked[: max(0, k - 1)]
    texts = [c.text for c in selected]
    ids = [c.chunk_id for c in selected]
    return texts, ids


# ---------------------------------------------------------------------
# STREAMING generation — yields each QAPair the instant it's produced
# ---------------------------------------------------------------------

def stream_cluster_based_pairs(chunks: List[Chunk], generator: QAGenerator, distribution: dict,
                                min_questions_per_cluster: int = 2, max_questions_per_cluster: int = 14,
                                chunks_per_question: float = 1.2, cluster_context_max_chunks: int = 10,
                                cluster_context_max_chars: int = 6000,
                                already_generated_by_cluster: Optional[Dict[int, int]] = None
                                ) -> Iterator[QAPair]:
    """Cluster-level QA generation, streamed. Each KMeans cluster is
    treated as one semantic topic: the LLM receives the concatenated text
    of the cluster's most informative chunks (NOT isolated single chunks)
    and this yields a dynamically-sized, taxonomy-balanced set of
    questions per cluster, one at a time, as soon as each is generated.
    Small clusters -> fewer questions; large/rich clusters -> more.

    `already_generated_by_cluster` (cluster_id -> count) lets a resumed
    run pick up where it left off: clusters that already have enough
    checkpointed questions from a previous, interrupted run are skipped
    or given a reduced quota instead of being fully regenerated.
    """
    already_generated_by_cluster = already_generated_by_cluster or {}
    clusters = _group_by_cluster(chunks)
    cluster_type_keys = [name for name in distribution if name in TAXONOMY]
    if not cluster_type_keys:
        logger.warning("No taxonomy-compatible keys found in distribution; skipping cluster generation.")
        return

    for cluster_id, cluster_chunks in clusters.items():
        cluster_text, used_chunk_ids = _build_cluster_context(
            cluster_chunks, cluster_context_max_chunks, cluster_context_max_chars
        )
        if not cluster_text.strip():
            continue

        n_questions = _questions_for_cluster_size(
            len(cluster_chunks), min_questions_per_cluster, max_questions_per_cluster, chunks_per_question
        )
        already = already_generated_by_cluster.get(cluster_id, 0)
        remaining = max(0, n_questions - already)
        if remaining <= 0:
            continue

        applicable = pick_applicable_types(cluster_text, distribution)
        applicable = [t for t in applicable if t in cluster_type_keys]
        if not applicable:
            # taxonomy.pick_applicable_types has no fallback of its own —
            # rather than silently skipping this cluster (0 questions),
            # fall back to whichever configured types are always-applicable
            # (applies=lambda text: True), and only skip if truly none exist.
            applicable = [t for t in cluster_type_keys if TAXONOMY[t].applies("")]
        if not applicable:
            continue

        # Expand each applicable type into the pool proportionally to its
        # configured weight, then shuffle, so the realized mix for this
        # cluster tracks the configured taxonomy_distribution instead of
        # drawing the same weighted type repeatedly.
        weighted_pool: List[str] = []
        for t in applicable:
            weighted_pool.extend([t] * max(1, round(distribution[t] * 100)))
        random.shuffle(weighted_pool)

        for i in range(remaining):
            qtype = weighted_pool[i % len(weighted_pool)] if weighted_pool else random.choice(applicable)
            result = generator.generate_from_cluster(cluster_text, TAXONOMY[qtype].instruction)
            if not result:
                continue
            yield QAPair(
                id=str(uuid.uuid4())[:8],
                question=result["question"],
                reference_answer=result["answer"],
                cluster_ids=[cluster_id],
                supporting_chunk_ids=list(used_chunk_ids),
                question_type=qtype,
                difficulty="base",
                cross_cluster=False,
            )


def stream_multi_hop_pairs(cross_pairs: List[CrossClusterPair], chunks: List[Chunk],
                            generator: QAGenerator, chunks_per_side: int = 3) -> Iterator[QAPair]:
    """Cross-cluster / multi-hop generation, streamed. Keeps the existing
    FLAT cross-cluster pairing mechanism (no tree traversal, no
    hierarchy, no GraphRAG/HSG-RAG) but feeds the LLM several relevant
    chunks from each side of the pair instead of a single representative
    chunk, for richer reasoning questions using combined information."""
    clusters = _group_by_cluster(chunks)

    for cp in cross_pairs:
        cluster_a_id = cp.chunk_a.cluster_id
        cluster_b_id = cp.chunk_b.cluster_id

        texts_a, ids_a = _select_relevant_chunks(
            clusters.get(cluster_a_id, [cp.chunk_a]), cp.chunk_a, chunks_per_side
        )
        texts_b, ids_b = _select_relevant_chunks(
            clusters.get(cluster_b_id, [cp.chunk_b]), cp.chunk_b, chunks_per_side
        )

        result = generator.generate_multi_hop(texts_a, texts_b, MULTI_HOP_INSTRUCTION)
        if not result:
            continue

        yield QAPair(
            id=str(uuid.uuid4())[:8],
            question=result["question"],
            reference_answer=result["answer"],
            cluster_ids=sorted(set([cluster_a_id, cluster_b_id])),
            supporting_chunk_ids=list(dict.fromkeys(ids_a + ids_b)),
            question_type="multi_hop",
            difficulty="base",
            cross_cluster=True,
        )


def stream_unanswerable_pairs(chunks: List[Chunk], generator: QAGenerator, count: int) -> Iterator[QAPair]:
    """Negative-question generation, streamed. `count` is an explicit
    number of negatives to produce (pipeline.py computes this from
    however many are still needed toward the target, accounting for any
    already checkpointed from a previous run) rather than a fraction, so
    resuming doesn't regenerate negatives that already exist on disk."""
    if count <= 0 or not chunks:
        return
    sample = random.sample(chunks, min(count, len(chunks)))
    for chunk in sample:
        result = generator.generate_unanswerable(chunk.text, UNANSWERABLE_INSTRUCTION)
        if not result:
            continue
        yield QAPair(
            id=str(uuid.uuid4())[:8],
            question=result["question"],
            reference_answer="NOT_ANSWERABLE",
            cluster_ids=[chunk.cluster_id],
            supporting_chunk_ids=[chunk.chunk_id],
            question_type="unanswerable",
            difficulty="base",
            cross_cluster=False,
            verified=True,
            verification_method="n/a (negative)",
        )


# ---------------------------------------------------------------------
# Batch wrappers (kept for compatibility / offline use / testing —
# pipeline.py itself now drives the streaming generators above directly)
# ---------------------------------------------------------------------

def generate_cluster_based_pairs(chunks: List[Chunk], generator: QAGenerator, distribution: dict,
                                  **kwargs) -> List[QAPair]:
    return list(stream_cluster_based_pairs(chunks, generator, distribution, **kwargs))


def generate_multi_hop_pairs(cross_pairs: List[CrossClusterPair], chunks: List[Chunk],
                              generator: QAGenerator, chunks_per_side: int = 3) -> List[QAPair]:
    return list(stream_multi_hop_pairs(cross_pairs, chunks, generator, chunks_per_side))


def generate_unanswerable_pairs(chunks: List[Chunk], generator: QAGenerator,
                                 fraction: float) -> List[QAPair]:
    n = max(1, int(len(chunks) * fraction)) if chunks else 0
    return list(stream_unanswerable_pairs(chunks, generator, n))


# ---------------------------------------------------------------------
# Streaming-friendly difficulty augmentation
# ---------------------------------------------------------------------

def maybe_paraphrase(pair: QAPair, generator: QAGenerator, fraction: float) -> Optional[QAPair]:
    """Per-pair Bernoulli trial: with probability `fraction`, spin off a
    paraphrased variant of an already-accepted base pair, right now,
    instead of sampling a fraction of a finished batch."""
    if pair.question_type == "unanswerable":
        return None
    if random.random() > fraction:
        return None
    new_q = generator.paraphrase_question(pair.question)
    if not new_q:
        return None
    return QAPair(
        id=str(uuid.uuid4())[:8],
        question=new_q,
        reference_answer=pair.reference_answer,
        cluster_ids=list(pair.cluster_ids),
        supporting_chunk_ids=list(pair.supporting_chunk_ids),
        question_type=pair.question_type,
        difficulty="paraphrased",
        cross_cluster=pair.cross_cluster,
    )


class CompoundPairer:
    """Streaming-friendly compound-question pairer. Buffers eligible
    accepted pairs; the moment a second eligible pair (with a different
    supporting-chunk set) shows up, immediately merges them into one
    compound QAPair instead of waiting for a whole batch to finish and
    sampling pairs out of it afterward."""

    ELIGIBLE_TYPES = ("factual", "definition", "cause_effect", "constraint", "dependency")
    _MAX_BUFFER = 50  # bound memory; oldest unpaired candidate is dropped if exceeded

    def __init__(self, generator: QAGenerator, fraction: float):
        self.generator = generator
        self.fraction = fraction
        self._pending: List[QAPair] = []

    def offer(self, pair: QAPair) -> Optional[QAPair]:
        if pair.question_type not in self.ELIGIBLE_TYPES:
            return None
        if random.random() > self.fraction:
            return None
        for i, candidate in enumerate(self._pending):
            if set(candidate.supporting_chunk_ids) != set(pair.supporting_chunk_ids):
                a = self._pending.pop(i)
                return self._merge(a, pair)
        self._pending.append(pair)
        if len(self._pending) > self._MAX_BUFFER:
            self._pending.pop(0)
        return None

    def _merge(self, a: QAPair, b: QAPair) -> Optional[QAPair]:
        merged = self.generator.merge_into_compound(
            {"question": a.question, "answer": a.reference_answer},
            {"question": b.question, "answer": b.reference_answer},
        )
        if not merged:
            return None
        return QAPair(
            id=str(uuid.uuid4())[:8],
            question=merged["question"],
            reference_answer=merged["answer"],
            cluster_ids=sorted(set(a.cluster_ids + b.cluster_ids)),
            supporting_chunk_ids=list(dict.fromkeys(a.supporting_chunk_ids + b.supporting_chunk_ids)),
            question_type="synthesis",
            difficulty="compound",
            cross_cluster=(a.cross_cluster or b.cross_cluster or set(a.cluster_ids) != set(b.cluster_ids)),
        )


# ---------------------------------------------------------------------
# Batch augmentation wrappers (kept for compatibility / offline use)
# ---------------------------------------------------------------------

def augment_with_paraphrases(pairs: List[QAPair], generator: QAGenerator,
                              fraction: float) -> List[QAPair]:
    new_pairs = []
    for p in pairs:
        para = maybe_paraphrase(p, generator, fraction)
        if para:
            new_pairs.append(para)
    return new_pairs


def augment_with_compounds(pairs: List[QAPair], generator: QAGenerator,
                            fraction: float) -> List[QAPair]:
    pairer = CompoundPairer(generator, fraction)
    new_pairs = []
    for p in pairs:
        compound = pairer.offer(p)
        if compound:
            new_pairs.append(compound)
    return new_pairs


# ---------------------------------------------------------------------
# Incremental (streaming) embedding-based duplicate removal
# ---------------------------------------------------------------------

class StreamingDeduper:
    """Incremental embedding-based duplicate check: compares each new
    question against every question kept so far across the ENTIRE run
    (not just the current round or batch) — including questions loaded
    from a previous, interrupted run's checkpoint file via seed(), so
    resuming never reintroduces a duplicate that's already on disk."""

    def __init__(self, embed_fn, threshold: float = 0.93):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self._embeddings: Optional[np.ndarray] = None

    def seed(self, questions: List[str]) -> None:
        if not questions:
            return
        embs = np.asarray(self.embed_fn(questions))
        self._embeddings = embs if self._embeddings is None else np.vstack([self._embeddings, embs])

    def is_duplicate(self, question: str) -> bool:
        if self._embeddings is None or len(self._embeddings) == 0:
            return False
        emb = np.asarray(self.embed_fn([question]))[0].reshape(1, -1)
        sims = cosine_similarity(emb, self._embeddings)[0]
        return bool(sims.max() >= self.threshold)

    def add(self, question: str) -> None:
        emb = np.asarray(self.embed_fn([question]))[0].reshape(1, -1)
        self._embeddings = emb if self._embeddings is None else np.vstack([self._embeddings, emb])


def deduplicate_pairs(pairs: List[QAPair], embed_fn, threshold: float = 0.93) -> List[QAPair]:
    """Batch convenience wrapper around StreamingDeduper, kept for
    compatibility / offline use."""
    if not pairs:
        return pairs
    deduper = StreamingDeduper(embed_fn, threshold=threshold)
    kept = []
    for pair in pairs:
        if deduper.is_duplicate(pair.question):
            continue
        deduper.add(pair.question)
        kept.append(pair)
    return kept


# ---------------------------------------------------------------------
# Round-trip / quality filtering
# ---------------------------------------------------------------------

def verify_pair(pair: QAPair, chunks: List[Chunk], chunk_embeddings: np.ndarray,
                 embed_fn, generator: QAGenerator, retrieval_top_k: int = 6,
                 similarity_threshold: float = 0.58, borderline_band: float = 0.12,
                 use_llm_judge: bool = True) -> bool:
    """Round-trip verify exactly ONE pair (the unit pipeline.py
    checkpoints on): retrieve top_k chunks for the question, get an
    answer from the LLM constrained to that context, and compare it to
    the reference answer. Sets verified / verification_score /
    verification_method on the pair in place and returns whether it
    passed."""
    if pair.question_type == "unanswerable":
        pair.verified = True
        pair.verification_method = pair.verification_method or "n/a (negative)"
        return True

    # Broken local-model JSON often leaves the schema placeholder, an empty
    # string, or a fragment as the answer. Reject it before spending a
    # retrieval/embedding pass and, crucially, before it can look like a
    # semantic-verification failure.
    reference_answer = pair.reference_answer.strip()
    if len(reference_answer) < 10 or reference_answer in {"...", "}", "{", "null", "None"}:
        pair.verification_score = 0.0
        pair.verification_method = "invalid_reference_answer"
        pair.verified = False
        return False

    q_emb = embed_fn([pair.question])[0].reshape(1, -1)
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    top_idx = np.argsort(-sims)[:retrieval_top_k]
    retrieved_texts = [chunks[i].text for i in top_idx]

    candidate_answer = generator.answer_from_context(pair.question, retrieved_texts)
    if candidate_answer.strip().upper().startswith("NOT_ANSWERABLE"):
        pair.verification_score = 0.0
        pair.verification_method = "embedding"
        pair.verified = False
        return False

    cand_emb = embed_fn([candidate_answer])[0].reshape(1, -1)
    src_emb = embed_fn([reference_answer])[0].reshape(1, -1)
    sim = float(cosine_similarity(cand_emb, src_emb)[0][0])

    decision = sim >= similarity_threshold
    method = "embedding"

    if use_llm_judge and abs(sim - similarity_threshold) <= borderline_band:
        decision = generator.judge_entailment(candidate_answer, pair.reference_answer)
        method = "llm_judge"

    pair.verification_score = sim
    pair.verification_method = method
    pair.verified = decision
    return decision


def round_trip_filter(pairs: List[QAPair], chunks: List[Chunk], chunk_embeddings: np.ndarray,
                       embed_fn, generator: QAGenerator, retrieval_top_k: int = 6,
                       similarity_threshold: float = 0.58, borderline_band: float = 0.12,
                       use_llm_judge: bool = True) -> tuple:
    """Batch convenience wrapper around verify_pair(), kept for
    compatibility / offline use. Returns (accepted, rejected)."""
    accepted, rejected = [], []
    for pair in pairs:
        passed = verify_pair(
            pair, chunks, chunk_embeddings, embed_fn, generator,
            retrieval_top_k=retrieval_top_k, similarity_threshold=similarity_threshold,
            borderline_band=borderline_band, use_llm_judge=use_llm_judge,
        )
        (accepted if passed else rejected).append(pair)
    return accepted, rejected
