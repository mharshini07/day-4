🧠 Where Does an LLM Store All Its “Memories”?

When we say an agent has memory, it’s important to realize:

The LLM itself (like GPT or Claude) doesn’t store memory between runs.
Memory lives outside the LLM — in external systems or data structures that the agent framework manages.

So, the LLM only uses memory — it doesn’t own it.
Let’s go type by type to see where and how each memory is stored.

⸻

📍 1. Short-Term Memory

Where stored: In the agent’s context window (the conversation history passed in the prompt).
Mechanism: Prompt injection — the history is appended to the LLM input.
Persistence: Lost when the session ends.

Example:

prompt = """
User: How do I create an API?
Assistant: You can start with FastAPI.
User: How do I deploy it?
"""

→ Both lines are passed every time you call the model. That’s short-term memory.

⸻

📍 2. Long-Term Memory

Where stored: In a vector database (vector store) like FAISS, Chroma, Pinecone, or Milvus.
Mechanism: Retrieval-Augmented Generation (RAG) — store and retrieve embeddings of text snippets.
Persistence: Permanent (until deleted).
Purpose: Recall facts, documents, user history, or older conversations.

Pipeline:

Text → Embedding Model → Vector Store (FAISS/Chroma)
          ↓
New Query → Embedding → Similarity Search → Retrieve Top-k → Add to Prompt

So, RAG = Long-Term Memory system.

⸻

📍 3. Episodic Memory

Where stored: In a structured log (JSON, DB, or vector store) of events or interactions.
Each “episode” stores:

{
  "timestamp": "2025-10-25T17:20",
  "action": "called_tool:generate_yaml",
  "result": "success",
  "reflection": "needed to batch runs next time"
}

Retrieval method: semantic search (like long-term), or filtering by metadata (time, type, success/failure).

Frameworks like OpenDevin, MemGPT, and CrewAI do this.

⸻

📍 4. Semantic Memory

Where stored:
	•	In the LLM weights (pretrained world knowledge).
	•	Optionally in a knowledge graph or vector store for domain-specific facts.

Example: GPT knows “Paris is the capital of France” because it’s part of its pretrained parameters.
For new knowledge (e.g., your company’s APIs), agents store it as vectors and recall via RAG.

⸻

📍 5. Procedural Memory

Where stored:
	•	In scripts, policies, or tool definitions (code-based memory).
	•	Or in fine-tuned weights if you train the LLM to perform tasks.

Example:
	•	“How to call Slack API” → stored as a tool description.
	•	“How to deploy model” → stored as a workflow or function chain.

Agents like OpenDevin and AutoGPT use YAML or JSON policies for this.

⸻

📍 6. Reflective Memory

Where stored:
In summary documents, notes, or JSON logs — often written by the agent itself after reflecting.

Example:
After finishing a task, the agent writes:

Note: I wasted tokens retrying the API. Next time, validate inputs first.

This reflection can then be embedded and stored in FAISS for recall during planning.

Used in: MemGPT, OpenDevin, LangGraph nodes with reflection.

⸻

📍 7. Cache / Tool Memory

Where stored:
	•	In local caches (like Redis, SQLite, or in-memory dicts).
	•	Used to avoid recomputation (like re-querying an API or re-embedding same text).

Example:

cache["get_weather:London"] = "25°C, sunny"

When same request comes again, it’s reused.

⸻

📍 8. Intent / Goal Memory

Where stored:
In the agent’s internal state or goal stack, usually an in-memory data structure.

Example:

goals = [
  {"goal": "summarize 50 convos", "status": "in_progress"},
  {"goal": "reduce TPM usage", "status": "pending"}
]

Used to decide what to prioritize next.

⸻

🧩 Summary Table

Memory Type	Stored In	Used By	Persistence
Short-Term	Prompt / Context window	LLM	Ephemeral
Long-Term	Vector DB (FAISS, Pinecone, etc.)	Retriever	Persistent
Episodic	Logs / JSON / DB	Planner, Reflector	Persistent
Semantic	LLM weights / Knowledge graph	LLM itself	Persistent
Procedural	Code / Tool registry	Action Executor	Persistent
Reflective	Notes / Summaries (vector store)	Self-improver	Persistent
Cache	Redis / Dict	Tool layer	Temporary
Intent / Goal	In-memory stack / DB	Planner	Session-based


⸻

💬 In Agentic Framework Terms

Framework	Memory System
LangChain	Conversation buffer + FAISS retriever
LangGraph	Node-level state memory
CrewAI	Reflection + episodic logs
OpenDevin	Vector + event memory
MemGPT	Long-term + working + reflection layers


⸻
