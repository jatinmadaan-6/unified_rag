"""
Question taxonomy — the substitute for graph-centrality-based sampling.

Each type has:
  - a prompt template used by generator.py (written for a CLUSTER-level
    "topic passage" — the concatenated context of a cluster's most
    informative chunks — rather than a single chunk, since Cluster-RAG
    benchmark generation now asks questions per cluster/topic instead of
    per chunk)
  - an `applies(text)` check so we don't ask for e.g. a procedure question
    from a passage with no procedural language in it.

Design note carried over from the original file: types where the
"applicability" is genuinely hard to detect from surface text (factual,
explanation, comparison, scenario, cause_effect, synthesis) are left as
`applies=lambda text: True` and rely on the LLM's own SKIP escape hatch
(see generator.generate_from_cluster) rather than a brittle keyword gate.
Types with a clear textual signal (definition, procedure, constraint,
exception, troubleshooting, dependency) keep a real regex gate, the same
way the original file gated "numerical"/"temporal" on digits/dates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_DEFINITION_KW = re.compile(r"\b(is defined as|refers to|means|known as|definition)\b", re.I)
_PROCEDURE_KW = re.compile(r"\b(step|first|then|next|process|procedure|submit|apply|register)\b", re.I)
_CONSTRAINT_KW = re.compile(r"\b(must|shall|required|eligib\w*|minimum|maximum|mandatory|not allowed)\b", re.I)
_EXCEPTION_KW = re.compile(r"\b(except|unless|however|exception|waived|special case)\b", re.I)
_TROUBLESHOOT_KW = re.compile(r"\b(if you|in case of|contact|issue|problem|grievance|complaint|error)\b", re.I)
_DEPENDENCY_KW = re.compile(r"\b(prerequisite|before|requires?|depends on|only after|must first)\b", re.I)


@dataclass
class QuestionType:
    name: str
    instruction: str
    applies: Callable[[str], bool]


TAXONOMY = {
    "factual": QuestionType(
        name="factual",
        instruction=(
            "Write a single factual/definitional technical question that can be answered directly "
            "and unambiguously from this topic passage (avoid yes/no questions), AND provide a "
            "clear, factual reference answer grounded strictly in the passage."
        ),
        applies=lambda text: True,
    ),
    "definition": QuestionType(
        name="definition",
        instruction=(
            "Write a question asking for the definition or meaning of a key term, "
            "concept, or role introduced in this topic passage, AND provide the complete, "
            "accurate definition as the reference answer."
        ),
        applies=lambda text: bool(_DEFINITION_KW.search(text)),
    ),
    "explanation": QuestionType(
        name="explanation",
        instruction=(
            "Write a 'why' or 'how' question that requires explaining a mechanism, "
            "rationale, or reasoning described in this topic passage, AND provide a detailed "
            "reference answer explaining it directly from the text. If the passage gives "
            "no rationale to explain, respond with SKIP."
        ),
        applies=lambda text: True,  # left to the LLM's SKIP escape hatch
    ),
    "procedure": QuestionType(
        name="procedure",
        instruction=(
            "Write a question about the steps, sequence, or process someone must follow "
            "as described in this topic passage, AND provide the step-by-step procedure as the "
            "reference answer."
        ),
        applies=lambda text: bool(_PROCEDURE_KW.search(text)),
    ),
    "comparison": QuestionType(
        name="comparison",
        instruction=(
            "Write a question that asks the reader to compare or contrast two things, "
            "concepts, or modes mentioned in this topic passage, AND provide a comparative "
            "reference answer based on the passage. Only do this if the passage contains "
            "two or more comparable items; otherwise respond with SKIP."
        ),
        applies=lambda text: True,  # left to the LLM's SKIP escape hatch
    ),
    "constraint": QuestionType(
        name="constraint",
        instruction=(
            "Write a question about a rule, limit, electrical/timing requirement, or eligibility "
            "condition stated in this topic passage (e.g. what is required, a minimum/maximum value), "
            "AND provide the exact limit or requirement as the reference answer."
        ),
        applies=lambda text: bool(_CONSTRAINT_KW.search(text)),
    ),
    "exception": QuestionType(
        name="exception",
        instruction=(
            "Write a question about an exception, special case, or condition under which "
            "the usual rule described in this topic passage does NOT apply, AND provide the "
            "reference answer explaining that exception."
        ),
        applies=lambda text: bool(_EXCEPTION_KW.search(text)),
    ),
    "scenario": QuestionType(
        name="scenario",
        instruction=(
            "Write a realistic scenario-based question describing a concrete situation a "
            "developer or engineer might face, whose resolution depends on this topic "
            "passage, AND provide the correct engineering solution/answer."
        ),
        applies=lambda text: True,
    ),
    "troubleshooting": QuestionType(
        name="troubleshooting",
        instruction=(
            "Write a question about what to do, how to diagnose, or how to resolve an issue "
            "or error based on this topic passage, AND provide the troubleshooting steps or "
            "cause as the reference answer."
        ),
        applies=lambda text: bool(_TROUBLESHOOT_KW.search(text)),
    ),
    "cause_effect": QuestionType(
        name="cause_effect",
        instruction=(
            "Write a 'why' or 'what causes/leads to' question whose answer is a cause, "
            "reason, or consequence explained in this topic passage, AND provide that cause "
            "or consequence as the reference answer. If the passage describes no cause-effect "
            "relationship, respond with SKIP."
        ),
        applies=lambda text: True,  # left to the LLM's SKIP escape hatch, mirrors original "causal"
    ),
    "dependency": QuestionType(
        name="dependency",
        instruction=(
            "Write a question about a dependency or prerequisite relationship — what must "
            "be true, completed, or in place before something else described in this "
            "topic passage can happen, AND provide the required prerequisites as the reference answer."
        ),
        applies=lambda text: bool(_DEPENDENCY_KW.search(text)),
    ),
    "synthesis": QuestionType(
        name="synthesis",
        instruction=(
            "Write a question that requires synthesizing two or more separate pieces of "
            "information from across this topic passage, AND provide the comprehensive, "
            "synthesized reference answer."
        ),
        applies=lambda text: True,
    ),
}

MULTI_HOP_INSTRUCTION = (
    "You are given several passages from Topic A and several passages from Topic B. "
    "Write ONE question that can only be correctly answered by combining information "
    "from BOTH topics — a reader who only saw one side should not be able to answer it "
    "fully. Then give the answer, drawing on both sides."
)

UNANSWERABLE_INSTRUCTION = (
    "You are given a topic passage. Write a plausible-sounding question that is closely "
    "related to the passage's topic but whose specific answer is NOT stated anywhere in "
    "the passage (e.g. ask for a detail, statistic, or fact that is absent). This is used "
    "to test whether a RAG system correctly refuses to answer instead of hallucinating."
)


def pick_applicable_types(text: str, distribution: dict) -> list:
    """Returns taxonomy names whose applicability check passes for this
    passage, restricted to keys present (with positive weight) in the
    configured distribution. Unchanged from the original: no fallback
    here if nothing matches — callers that need one (e.g. cluster
    generation, so a whole cluster doesn't get silently skipped) apply it
    themselves, same as before."""
    return [
        name for name, weight in distribution.items()
        if name in TAXONOMY and TAXONOMY[name].applies(text) and weight > 0
    ]
