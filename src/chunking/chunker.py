from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents: List[Dict], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict]:
    """
    Chunks documents using LangChain's RecursiveCharacterTextSplitter.
    Matches the chunking behavior used in HSG-RAG and Vanilla RAG baselines.
    Returns: List of dicts {'unit_id': str, 'doc_id': str, 'text': str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["text"]
        split_texts = splitter.split_text(text)
        
        for i, c in enumerate(split_texts):
            chunks.append({
                "unit_id": f"{doc_id}::u{i}",
                "doc_id": doc_id,
                "text": c
            })
            
    return chunks
