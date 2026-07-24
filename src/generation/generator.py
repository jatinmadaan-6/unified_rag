"""
All LLM generation goes through this module: question/answer generation
(now cluster-level, not per-chunk), paraphrasing, compound-question
merging, context-only answering (for the round-trip check), and
entailment judging.

100% free, local, Colab-compatible — runs on HuggingFace `transformers`
and supports the same interchangeable free models as before:

  - "Qwen/Qwen2.5-3B-Instruct"            (default)
  - "mistralai/Mistral-7B-Instruct-v0.3"  (pass load_in_4bit=True on a T4)
  - "google/flan-t5-base"                 (smallest memory footprint)

CHANGE LOG (adaptation for Cluster-RAG benchmark generation):
  - generate_from_cluster() replaces per-chunk generation as the primary
    entry point: the "passage" it receives is a cluster's aggregated
    context, not a single chunk.
  - generate_multi_hop() now takes LISTS of passages from each side of a
    cross-cluster pair (several chunks per cluster) instead of one
    representative chunk per side, for richer cross-topic reasoning.
  - Generation calls now sample (do_sample=True, temperature/top_p) so
    that regenerating over the same cluster/topic across generation
    rounds produces genuinely different, non-template questions, per the
    "avoid repetitive/template-like questions" requirement. Verification
    calls (answer_from_context, judge_entailment) stay deterministic
    (do_sample=False) since those are quality-control checks, not
    creative generation.

Method names/signatures called by qa_pairs.py: generate_from_cluster,
generate_single_hop (kept for compatibility, unused by the cluster
pipeline), generate_multi_hop, generate_unanswerable, paraphrase_question,
merge_into_compound, answer_from_context, judge_entailment.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction. Local models are less reliable than a
    hosted API at strictly following 'JSON only' instructions, so this is
    a little more forgiving: it tries a fenced block first, then the raw
    text, then the first {...} blob it can find."""
    fence = _JSON_FENCE_RE.search(text)
    payload = fence.group(1) if fence else text
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(payload)
        if m:
            return json.loads(m.group(0))
        raise


class QAGenerator:
    """Local, free HuggingFace-backed generator for cluster-level QA
    generation."""

    SEQ2SEQ_MODELS = {"google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-small"}

    def __init__(self, generation_model: str = "Qwen/Qwen2.5-3B-Instruct",
                 judge_model: Optional[str] = None,
                 load_in_4bit: bool = False,
                 max_new_tokens: int = 400,
                 generation_temperature: float = 0.85,
                 generation_top_p: float = 0.95):
        self.generation_model_name = generation_model
        self.judge_model_name = judge_model or generation_model
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading generation model %s on %s ...", generation_model, self.device)
        self._pipe = self._load_pipeline(generation_model, load_in_4bit)

        if self.judge_model_name == self.generation_model_name:
            self._judge_pipe = self._pipe
        else:
            logger.info("Loading judge model %s on %s ...", self.judge_model_name, self.device)
            self._judge_pipe = self._load_pipeline(self.judge_model_name, load_in_4bit)

    # ---------- model loading ----------
    def _load_pipeline(self, model_name: str, load_in_4bit: bool):
        from transformers import pipeline as hf_pipeline

        if model_name in self.SEQ2SEQ_MODELS:
            return hf_pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )

        kwargs = {}
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"

        return hf_pipeline("text-generation", model=model_name, **kwargs)

    # ---------- low-level call ----------
    def _call(self, pipe, system: str, user: str, max_new_tokens: Optional[int] = None,
               do_sample: bool = False, temperature: Optional[float] = None,
               top_p: Optional[float] = None) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.generation_temperature
        top_p = top_p if top_p is not None else self.generation_top_p

        if pipe.task == "text2text-generation":
            prompt = f"{system}\n\n{user}"
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            out = pipe(prompt, **gen_kwargs)
            return out[0]["generated_text"]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        pad_id = pipe.tokenizer.eos_token_id
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "return_full_text": False,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        out = pipe(messages, **gen_kwargs)
        generated = out[0]["generated_text"]
        # chat pipelines can return either a string or the trailing assistant turn
        if isinstance(generated, list):
            return generated[-1]["content"]
        return generated

    # ---------- cluster-level generation (primary entry point) ----------
    def generate_from_cluster(self, cluster_text: str, type_instruction: str) -> Optional[dict]:
        """Generate one question/answer pair grounded in an entire semantic
        cluster's aggregated context (concatenated text of the cluster's
        most informative chunks), not a single chunk."""
        system = (
            "You generate evaluation questions for a RAG (retrieval-augmented generation) "
            "system. The passage below represents ONE semantic topic (a cluster), built by "
            "concatenating the most relevant passages of that topic. Ground the question "
            "strictly in this passage, and write it the way a real user would naturally "
            "phrase it — avoid template-like or repetitive phrasing. Respond with ONLY a "
            "JSON object: {\"question\": \"...\", \"answer\": \"...\"} or {\"skip\": true} "
            "if the instruction cannot be satisfied by this passage. No prose, no markdown."
        )
        user = f"Instruction: {type_instruction}\n\nTopic passage:\n{cluster_text}"
        raw = self._call(self._pipe, system, user, do_sample=True)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse cluster-generation JSON: %r", raw[:200])
            return None
        if data.get("skip"):
            return None
        if "question" in data and "answer" in data:
            return data
        return None

    # ---------- single-hop generation (kept for backward compatibility) ----------
    def generate_single_hop(self, chunk_text: str, type_instruction: str) -> Optional[dict]:
        """Retained for compatibility with any other caller; the cluster
        pipeline itself now goes through generate_from_cluster()."""
        system = (
            "You generate evaluation questions for a RAG (retrieval-augmented generation) "
            "system, strictly grounded in the provided passage. Respond with ONLY a JSON "
            "object: {\"question\": \"...\", \"answer\": \"...\"} or {\"skip\": true} if "
            "the instruction cannot be satisfied by this passage. No prose, no markdown."
        )
        user = f"Instruction: {type_instruction}\n\nPassage:\n{chunk_text}"
        raw = self._call(self._pipe, system, user, do_sample=True)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse generation JSON: %r", raw[:200])
            return None
        if data.get("skip"):
            return None
        if "question" in data and "answer" in data:
            return data
        return None

    # ---------- multi-hop generation (multiple chunks per side) ----------
    def generate_multi_hop(self, texts_a: List[str], texts_b: List[str], instruction: str) -> Optional[dict]:
        """texts_a / texts_b are several relevant chunks pulled from each
        side of a cross-cluster pair (flat clustering, no tree traversal),
        giving the LLM more material to build a genuine reasoning question
        from than a single representative chunk per side."""
        system = (
            "You generate multi-hop evaluation questions for a RAG system. Respond with "
            "ONLY a JSON object: {\"question\": \"...\", \"answer\": \"...\"}. No prose."
        )
        passage_a = "\n\n".join(t for t in texts_a if t.strip())
        passage_b = "\n\n".join(t for t in texts_b if t.strip())
        user = f"{instruction}\n\nTopic A passages:\n{passage_a}\n\nTopic B passages:\n{passage_b}"
        raw = self._call(self._pipe, system, user, do_sample=True)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            return None
        if "question" in data and "answer" in data:
            return data
        return None

    # ---------- unanswerable / negative generation ----------
    def generate_unanswerable(self, chunk_text: str, instruction: str) -> Optional[dict]:
        system = (
            "You write plausible but unanswerable questions to test a RAG system's "
            "refusal behavior. Respond with ONLY JSON: "
            "{\"question\": \"...\", \"answer\": \"NOT_ANSWERABLE\"}. No prose."
        )
        user = f"{instruction}\n\nPassage:\n{chunk_text}"
        raw = self._call(self._pipe, system, user, do_sample=True)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            return None
        return data if "question" in data else None

    # ---------- augmentation ----------
    def paraphrase_question(self, question: str) -> Optional[str]:
        system = (
            "Rewrite the given question so it asks for the exact same answer but uses "
            "different vocabulary and sentence structure than the original (minimize "
            "word overlap). Respond with ONLY JSON: {\"question\": \"...\"}."
        )
        raw = self._call(self._pipe, system, question, max_new_tokens=150, do_sample=True)
        try:
            return _extract_json(raw).get("question")
        except (json.JSONDecodeError, AttributeError):
            return None

    def merge_into_compound(self, qa_a: dict, qa_b: dict) -> Optional[dict]:
        system = (
            "You are given two simple question/answer pairs about related topics. "
            "Combine them into ONE compound question that requires both pieces of "
            "information to answer fully, plus a combined answer. Respond with ONLY "
            "JSON: {\"question\": \"...\", \"answer\": \"...\"}."
        )
        user = (
            f"QA 1: Q: {qa_a['question']} A: {qa_a['answer']}\n"
            f"QA 2: Q: {qa_b['question']} A: {qa_b['answer']}"
        )
        raw = self._call(self._pipe, system, user, max_new_tokens=300, do_sample=True)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            return None
        return data if "question" in data and "answer" in data else None

    # ---------- round-trip check (deterministic — QC, not creative generation) ----------
    def answer_from_context(self, question: str, context_chunks: List[str]) -> str:
        system = (
            "Answer the question using ONLY the provided context. If the context does "
            "not contain the answer, respond exactly with: NOT_ANSWERABLE. Be concise."
        )
        context = "\n\n---\n\n".join(context_chunks)
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return self._call(self._judge_pipe, system, user, max_new_tokens=200, do_sample=False).strip()

    def judge_entailment(self, candidate_answer: str, source_answer: str) -> bool:
        system = (
            "You are a strict evaluator. Given a candidate answer and a reference "
            "answer, decide if the candidate is factually consistent with the reference "
            "(same core facts, minor wording differences are fine). Respond with ONLY "
            "JSON: {\"consistent\": true} or {\"consistent\": false}."
        )
        user = f"Reference answer: {source_answer}\nCandidate answer: {candidate_answer}"
        raw = self._call(self._judge_pipe, system, user, max_new_tokens=50, do_sample=False)
        try:
            return bool(_extract_json(raw).get("consistent", False))
        except (json.JSONDecodeError, AttributeError):
            return False

    # ---------- generic convenience interface ----------
    def generate_question(self, context: str,
                           instruction: str = "Write a single question that can be "
                                               "answered directly from this passage.") -> Optional[str]:
        result = self.generate_from_cluster(context, instruction)
        return result["question"] if result else None

    def generate_answer(self, question: str, context: str) -> str:
        return self.answer_from_context(question, [context])

    def judge_answer(self, candidate_answer: str, reference_answer: str) -> bool:
        return self.judge_entailment(candidate_answer, reference_answer)
