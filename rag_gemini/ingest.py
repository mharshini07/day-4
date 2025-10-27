import os
import pickle
from typing import List, Tuple

import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split a long text into overlapping chunks of approximately chunk_size characters."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]


def build_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, SentenceTransformer]:
    """Return embeddings array and the model instance."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model


def save_faiss_index(embeddings: np.ndarray, texts: List[str], out_dir: str):
    """Save a FAISS index and associated metadata (texts) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    dim = embeddings.shape[1]
    if faiss is None:
        raise RuntimeError("faiss is not available. Please install faiss-cpu.")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    # Save texts
    with open(os.path.join(out_dir, "index_data.pkl"), "wb") as f:
        pickle.dump(texts, f)


def load_faiss_index(out_dir: str):
    """Load FAISS index and texts. Returns (index, texts)."""
    if faiss is None:
        raise RuntimeError("faiss is not available. Please install faiss-cpu.")
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "index_data.pkl"), "rb") as f:
        texts = pickle.load(f)
    return index, texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a PDF and build FAISS index of embeddings")
    parser.add_argument("pdf_path", help="Path to the PDF research paper")
    parser.add_argument("out_dir", help="Directory to write index and metadata")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name for embeddings")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    print(f"Reading PDF: {args.pdf_path}")
    text = extract_text_from_pdf(args.pdf_path)
    print(f"Extracted {len(text)} characters")
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Split into {len(chunks)} chunks")
    print("Building embeddings (this may take a little while)...")
    embeddings, _ = build_embeddings(chunks, model_name=args.model)
    save_faiss_index(embeddings, chunks, args.out_dir)
    print(f"Saved index to {args.out_dir}")
