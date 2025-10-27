Gemini RAG — simple Retrieval-Augmented Generation using a Gemini model

Overview

This small project ingests a research paper (PDF), builds a FAISS vector index using embeddings from SentenceTransformers, and answers questions by retrieving relevant chunks and calling a Gemini model using LangChain's `ChatGoogleGenerativeAI` wrapper.

Security note: Do NOT commit your Gemini API key. Provide it via environment variable `GEMINI_API_KEY`.

Setup

1. Create and activate a Python virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r rag_gemini/requirements.txt

Note: This project uses LangChain's `langchain-google-genai` wrapper to call Gemini. Ensure you have `langchain` and `langchain-google-genai` installed (they're listed in `requirements.txt`).

3. Set your Gemini API key in the environment (example for zsh/macOS):

   export GEMINI_API_KEY="YOUR_API_KEY_HERE"

Optional: You can set `GEMINI_MODEL` to prefer a specific Gemini model name. Default used: `gemini-1.5` (change as needed, e.g. `gemini-2.5-pro`).

Usage

1) Ingest a PDF and build the index:

   python rag_gemini/ingest.py path/to/paper.pdf rag_index_dir

This will create `rag_index_dir/index.faiss` and `rag_index_dir/index_data.pkl`.

2) Query with a question:

   python rag_gemini/rag_gemini.py rag_index_dir "What is the main contribution of the paper?"

This will retrieve the top chunks and call Gemini (via LangChain) to produce an answer.

Notes & Extensions

- The app uses a local sentence-transformers model (`all-MiniLM-L6-v2`) for embeddings by default so you don't need a cloud embedding API. You can change this by setting `EMBED_MODEL` environment variable.
- The Gemini call is performed with the `google-generativeai` client. The exact response shape can vary by library versions; the code attempts to be tolerant.
- For production or larger collections, consider chunking by semantic boundaries, storing chunk metadata (page numbers), and using a more robust vector store like Chroma or Weaviate.

If you want, I can:
- Add page/offset metadata to the saved index
- Add a small web UI (Streamlit/Gradio)
- Add unit tests and a quick smoke test script
