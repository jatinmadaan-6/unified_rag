"""
Loads documents directly from a local filesystem folder — e.g. a Google
Drive folder mounted inside Colab at /content/drive/MyDrive/RAG_DATASET.

No Google API, no service-account JSON, no OAuth. In Colab, all you do is:

    from google.colab import drive
    drive.mount('/content/drive')

...and then point DATA_FOLDER at any folder under
/content/drive/MyDrive/... (or, outside Colab, at any local directory).

Supported inputs: .pdf, .docx, .txt, .md / .markdown
Anything else is skipped with a warning so the pipeline never crashes on
an unexpected file type in the folder.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

TEXT_EXTS = {".txt", ".md", ".markdown"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}

DEFAULT_SUPPORTED_EXTENSIONS = tuple(TEXT_EXTS | PDF_EXTS | DOCX_EXTS)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    mime_type: str          # kept for interface parity with the rest of the pipeline
    source_path: str        # human-readable breadcrumb / relative path


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx_text(path: str) -> str:
    import docx
    document = docx.Document(path)
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def _extract_text_file(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")


def _list_files(folder: str, recursive: bool, supported_extensions) -> List[Tuple[str, str]]:
    """Returns list of (absolute_path, relative_breadcrumb) tuples."""
    out = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                ext = os.path.splitext(name)[1].lower()
                if ext in supported_extensions:
                    abspath = os.path.join(root, name)
                    rel = os.path.relpath(abspath, folder)
                    out.append((abspath, rel))
    else:
        for name in sorted(os.listdir(folder)):
            abspath = os.path.join(folder, name)
            if os.path.isfile(abspath):
                ext = os.path.splitext(name)[1].lower()
                if ext in supported_extensions:
                    out.append((abspath, name))
    return out


def load_document(path: str, relative_path: str) -> Optional[Document]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in PDF_EXTS:
            text = _extract_pdf_text(path)
            mime = "application/pdf"
        elif ext in DOCX_EXTS:
            text = _extract_docx_text(path)
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext in TEXT_EXTS:
            text = _extract_text_file(path)
            mime = "text/markdown" if ext in (".md", ".markdown") else "text/plain"
        else:
            logger.warning("Skipping unsupported file type: %s", relative_path)
            return None
    except Exception as e:  # noqa: BLE001 - keep the pipeline alive on a bad file
        logger.error("Failed to read %s: %s", relative_path, e)
        return None

    text = text.strip()
    if not text:
        logger.warning("File %s produced empty text, skipping", relative_path)
        return None

    return Document(
        doc_id=str(uuid.uuid4())[:8],
        title=os.path.basename(path),
        text=text,
        mime_type=mime,
        source_path=relative_path,
    )


def load_local_documents(folder: str, recursive: bool = True,
                          supported_extensions=DEFAULT_SUPPORTED_EXTENSIONS) -> List[Document]:
    """Main entry point: local folder path -> list of cleaned Document objects.

    `folder` can be any filesystem path, including a Google-Drive-mounted
    Colab path such as /content/drive/MyDrive/RAG_DATASET.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Data folder not found: {folder}\n"
            f"If you're on Colab, make sure you ran:\n"
            f"  from google.colab import drive\n"
            f"  drive.mount('/content/drive')\n"
            f"and that the path points inside /content/drive/MyDrive/..."
        )

    files = _list_files(folder, recursive, set(supported_extensions))
    logger.info("Found %d candidate files in %s", len(files), folder)

    documents = []
    for abspath, rel in files:
        doc = load_document(abspath, rel)
        if doc:
            documents.append(doc)

    logger.info("Successfully loaded %d/%d documents", len(documents), len(files))
    return documents
