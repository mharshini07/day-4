import os
import textwrap
from typing import List

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    import faiss
except Exception:
    faiss = None
from ingest import load_faiss_index


def call_gemini(prompt: str, model: str = None, api_key_env: str = "GEMINI_API_KEY") -> str:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Please set the environment variable {api_key_env} with your Gemini API key.")

    model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"
    llm = ChatGoogleGenerativeAI(api_key=api_key, model=model)

    try:
        result = llm.predict(prompt)
    except Exception:
        result = llm(prompt)

    if isinstance(result, str):
        return result
    return str(result)


def retrieve(index_dir: str, query: str, k: int = 3):
    index, texts = load_faiss_index(index_dir)
    if faiss is None:
        raise RuntimeError("faiss not available; install faiss-cpu")
    from sentence_transformers import SentenceTransformer

    embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    embed_model = SentenceTransformer(embed_model_name)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype("float32"), k)
    results = []
    for idx in I[0]:
        results.append(texts[idx])
    return results


def build_prompt(context_chunks: List[str], question: str) -> str:
    wrapped = "\n\n".join([textwrap.fill(c, width=100) for c in context_chunks])
    prompt = f"You are a helpful research assistant. Use the provided context from a research paper to answer the question. If the answer is not contained in the context, say you don't know and cite the context sentences used.\n\nContext:\n{wrapped}\n\nQuestion: {question}\n\nAnswer:"
    return prompt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG with Gemini: query a paper index and call Gemini to answer")
    parser.add_argument("index_dir", help="Directory where the FAISS index and metadata were saved")
    parser.add_argument("question", help="Question to ask about the paper")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved chunks to include in context")
    parser.add_argument("--model", default=None, help="Gemini model name (optional) e.g. models/gemini-1.5-mini)")
    args = parser.parse_args()

    context = retrieve(args.index_dir, args.question, k=args.k)
    prompt = build_prompt(context, args.question)
    print("--- Prompt sent to Gemini (truncated) ---")
    print(prompt[:4000])
    answer = call_gemini(prompt, model=args.model)
    print("\n--- Answer from Gemini ---\n")
    print(answer)
