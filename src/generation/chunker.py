"""
Structure-aware chunking.

Instead of building a graph to discover relationships between chunks, we
lean on the document's own structure (markdown-style headings, or a
heuristic heading detector for plain-text exports of Google Docs) to give
each chunk a "section_path" — a free hierarchy we get from parsing, not
from an expensive relation-extraction step.

Within a section, long text is split into token-budgeted chunks with a
small overlap so no sentence gets cut mid-thought.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from local_loader import Document

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Heuristic for plain-text exports (e.g. Google Docs export) that have no
# markdown: a short, standalone line with no terminal punctuation, followed
# by a blank line, is treated as a heading.
PLAIN_HEADING_RE = re.compile(r"^[A-Z0-9][^.!?]{0,70}$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: List[str]
    text: str
    position: int  # order within the document
    cluster_id: int = -1  # filled in later by clustering.py


def _approx_token_count(text: str) -> int:
    # Cheap whitespace-based approximation, good enough for chunk sizing.
    return max(1, len(text.split()))


def _parse_sections(text: str):
    """Yields (heading_stack, paragraph_text) tuples in document order."""
    lines = text.split("\n")
    stack: List[str] = []
    buffer: List[str] = []

    yield_list = []

    def flush():
        para = "\n".join(buffer).strip()
        if para:
            yield_list.append((list(stack), para))
        buffer.clear()

    prev_blank = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        md = MD_HEADING_RE.match(stripped)
        if md:
            flush()
            level = len(md.group(1))
            heading = md.group(2).strip()
            stack = stack[: level - 1] + [heading]
            prev_blank = True
            continue
        # heuristic plain-text heading: short line, isolated by blank lines
        next_blank = (i + 1 >= len(lines)) or (lines[i + 1].strip() == "")
        if stripped and prev_blank and next_blank and PLAIN_HEADING_RE.match(stripped) \
                and len(stripped.split()) <= 10:
            flush()
            stack = stack[:0] + [stripped]  # treat as a single top-level heading reset
            prev_blank = True
            continue
        if not stripped:
            flush()
            prev_blank = True
            continue
        buffer.append(line)
        prev_blank = False
    flush()
    return yield_list


def _split_paragraph_to_budget(paragraph: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    words = paragraph.split()
    if len(words) <= max_tokens:
        return [paragraph]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_tokens  # slide with overlap
    return chunks


def chunk_document(doc: Document, max_chunk_tokens: int = 300,
                    chunk_overlap_tokens: int = 40,
                    min_chunk_tokens: int = 40) -> List[Chunk]:
    sections = _parse_sections(doc.text)
    chunks: List[Chunk] = []
    position = 0

    # merge tiny consecutive paragraphs under the same heading before splitting,
    # so we don't emit a flood of very small chunks
    merged: List[tuple] = []
    for stack, para in sections:
        if merged and merged[-1][0] == stack and \
                _approx_token_count(merged[-1][1]) + _approx_token_count(para) <= max_chunk_tokens:
            merged[-1] = (stack, merged[-1][1] + "\n\n" + para)
        else:
            merged.append((stack, para))

    for stack, para in merged:
        if _approx_token_count(para) < min_chunk_tokens and chunks and \
                chunks[-1].section_path == stack:
            # fold very small trailing paragraph into previous chunk
            chunks[-1].text += "\n\n" + para
            continue
        pieces = _split_paragraph_to_budget(para, max_chunk_tokens, chunk_overlap_tokens)
        for piece in pieces:
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4())[:8],
                doc_id=doc.doc_id,
                doc_title=doc.title,
                section_path=stack,
                text=piece,
                position=position,
            ))
            position += 1

    return chunks


def chunk_documents(documents: List[Document], **kwargs) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks
