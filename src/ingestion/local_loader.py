import os
from typing import List, Dict


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)

def load_local_documents(folder: str, recursive: bool = True, supported_extensions: List[str] = None) -> List[Dict]:
    """Loads documents from a folder into a list of dicts: {'doc_id': str, 'text': str, 'path': str}"""
    if supported_extensions is None:
        supported_extensions = [".txt", ".md", ".pdf"]
        
    documents = []
    
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist.")
        return documents
        
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                path = os.path.join(root, file)
                doc_id = os.path.splitext(file)[0]
                try:
                    if ext == ".pdf":
                        text = _read_pdf(path)
                    else:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    documents.append({
                        "doc_id": doc_id,
                        "text": text,
                        "path": path
                    })
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    
        if not recursive:
            break
            
    return documents
