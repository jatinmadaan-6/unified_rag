from typing import List, Dict

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Compatibility with older LangChain installations.
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

def chunk_documents(documents: List[Dict], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict]:
    """
    Chunks documents using LangChain's RecursiveCharacterTextSplitter.
    Matches the chunking behavior used in HSG-RAG and Vanilla RAG baselines.
    Returns: List of dicts {'unit_id': str, 'doc_id': str, 'text': str}
    """
    splitter = None if RecursiveCharacterTextSplitter is None else RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["text"]
        # A simple overlap-preserving fallback lets the evaluation run even
        # before optional LangChain packages are installed.
        split_texts = splitter.split_text(text) if splitter else [text[i:i + chunk_size] for i in range(0, len(text), max(1, chunk_size - chunk_overlap))]
        
        for i, c in enumerate(split_texts):
            chunks.append({
                "unit_id": f"{doc_id}::u{i}",
                "doc_id": doc_id,
                "text": c
            })
            
    return chunks
