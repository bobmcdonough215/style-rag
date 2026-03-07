# Style RAG — Advanced Retrieval Patterns

Four architectural improvements beyond basic top-K retrieval, ordered by implementation complexity. Each builds on the previous. All are within scope of the current stack — no new models or external APIs required unless noted.

---

## Build Progression

```
Layer 1 (MVP)     → Basic top-K similarity search
Layer 2a          → Priority-weighted reranker
Layer 2b          → Parent Document Retrieval
Layer 2c          → Contextual Compression
Layer 3           → Semantic Cache
```

Each stage produces a working system. Don't skip ahead — validate each layer before adding the next.

---

## Layer 2a — Priority-Weighted Reranker

### The Problem
Standard vector search ranks chunks by semantic similarity only. It has no concept of authority or precedence. If AP Style (Priority 3) and House Style (Priority 1) both contain a rule about date formatting, vector search may return the AP chunk first simply because its wording happened to be closer to the query. Qwen then follows the first rule it sees.

### The Fix
After retrieving the top K chunks and before passing them to Qwen, sort the results by `priority` metadata. Priority 1 (house style) always appears first in the prompt context — Qwen is significantly more likely to follow rules that appear earlier in the context window.

### Implementation — `rag.py`

```python
def rerank_by_priority(docs: list) -> list:
    """
    Sort retrieved chunks by metadata priority (ascending).
    Priority 1 = highest authority (house style) — appears first in context.
    Priority 4 = lowest authority (external standards) — appears last.
    Chunks without a priority tag are sorted to the end.
    """
    return sorted(docs, key=lambda d: d.metadata.get("priority", 99))
```

Call this immediately after retrieval and before building the prompt context:

```python
retrieved_docs = retriever.get_relevant_documents(query)
ranked_docs = rerank_by_priority(retrieved_docs)
# Pass ranked_docs to the LLM, not retrieved_docs
```

### Why This Works
LLMs exhibit "primacy bias" — they tend to weight information that appears earlier in the context more heavily than information that appears later. By placing high-priority sources first, you're working with this tendency rather than against it.

### Cost
Five lines of Python. No additional API calls. No latency impact. Implement this first.

---

## Layer 2b — Parent Document Retrieval

### The Problem
A 500-token chunk is optimized for retrieval precision — small enough that its embedding is specific to one topic. But style rules are sometimes split across chunk boundaries. If a rule and its exception live in adjacent chunks, the LLM only sees half the context and may give an incomplete or misleading answer.

### The Fix
Use LangChain's `ParentDocumentRetriever`. Store small "child" chunks (200 tokens) for high-accuracy similarity search, but when a match is found, retrieve the full "parent" chunk (800 tokens) to give Qwen complete context.

```
Ingest:    Full document → split into parent chunks (800 tokens) → stored in docstore
                        → split parents into child chunks (200 tokens) → embedded in Chroma

Retrieval: Query → similarity search against child chunks (precise)
                → fetch corresponding parent chunks (complete context)
                → pass parent chunks to Qwen
```

### Implementation — `ingest.py` changes

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Two splitters: small for search, large for context
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)

# Docstore holds parent chunks in memory (persists for session)
# For persistent storage across sessions, swap InMemoryStore
# for a SQLite-backed store
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents — ParentDocumentRetriever handles both splits internally
retriever.add_documents(documents)
```

### The Gotcha — InMemoryStore Does Not Persist
`InMemoryStore` is lost when the app restarts. For a portfolio project this is acceptable — re-run `ingest.py` to rebuild it. For a production tool, replace `InMemoryStore` with a persistent SQLite store:

```python
from langchain.storage import LocalFileStore

docstore = LocalFileStore("./docstore")  # Persists to disk
```

Document this tradeoff explicitly in your README — it shows you understand the production implications of your implementation choice.

### Impact
- Retrieval precision stays high (small child chunks)
- LLM context quality improves (large parent chunks)
- No additional API calls, no new models
- Small change to `ingest.py`, transparent to `app.py`

---

## Layer 2c — Contextual Compression

### The Problem
Retrieving 4 chunks of 500 tokens each sends ~2,000 tokens of context to Qwen. In practice, only 50–100 of those tokens contain the actual answer. The rest is surrounding text that:
- Consumes your context window budget
- Occupies KV cache RAM on your M4 Pro
- Can "distract" the LLM with tangentially related content

### The Fix
Wrap your retriever in a `ContextualCompressionRetriever`. Before the retrieved chunks are passed to Qwen, a compression step strips out sentences that don't relate to the user's query.

### Two Options — Know the Tradeoff

**Option A: LLMChainFilter (uses Qwen)**
Uses Qwen itself to filter irrelevant sentences from each chunk. Higher quality filtering but adds a second LLM call per query, which increases latency.

```python
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

**Option B: EmbeddingsFilter (no extra LLM call)**
Uses embedding similarity to filter sentences. Faster, no extra LLM call, slightly lower quality than Option A.

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)
```

### Recommendation
Start with **Option B** (EmbeddingsFilter). It adds no latency, requires no extra LLM call, and the RAM savings on your M4 Pro are immediate. Upgrade to Option A only if you find filtering quality is insufficient.

### PyTorch Note
If you use a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM` from Hugging Face) as the compressor, PyTorch comes back into the picture — the model runs locally via the Transformers library. This is the one place in the pipeline where PyTorch becomes relevant. It's a small model, runs fast on the M4 Pro, and the quality improvement is significant. Worth considering in Layer 3.

### Impact on M4 Pro
Compressed chunks reduce KV cache memory usage per query. On a 24GB machine this isn't a crisis, but it translates to faster Time to First Token and more headroom for longer conversations with memory enabled.

---

## Layer 3 — Semantic Cache

### The Problem
Every query runs the full pipeline: embed the question, search Chroma, retrieve chunks, generate a response. For repeated or similar questions — especially during testing or in a small team using the same tool — this is redundant work. Qwen generates the same answer twice while your Mac burns CPU and RAM doing it.

### The Fix
Before calling the full RAG pipeline, check a local cache for semantically similar previous queries. If a match above a similarity threshold (0.95) is found, return the cached answer immediately. The LLM is never called.

```
Query → Cache check (< 5ms)
    → Hit (similarity > 0.95): return cached answer instantly
    → Miss: run full RAG pipeline → store result in cache → return answer
```

### Implementation — `rag.py`

```python
import sqlite3
import json
import numpy as np
from langchain_openai import OpenAIEmbeddings
from config import SIMILARITY_THRESHOLD_CACHE

CACHE_DB = "./cache/semantic_cache.db"

def init_cache():
    os.makedirs("./cache", exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            embedding TEXT,
            response TEXT,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_cache(query_embedding: list, threshold: float = 0.95):
    """
    Check if a semantically similar query has been cached.
    Returns (response, sources) if found, None otherwise.
    """
    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute("SELECT query, embedding, response, sources FROM query_cache").fetchall()
    conn.close()

    for row in rows:
        cached_embedding = json.loads(row[1])
        similarity = cosine_similarity(query_embedding, cached_embedding)
        if similarity >= threshold:
            print(f"Cache hit! Similarity: {similarity:.3f}")
            return row[2], json.loads(row[3])  # response, sources

    return None, None

def store_in_cache(query: str, query_embedding: list, response: str, sources: list):
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "INSERT INTO query_cache (query, embedding, response, sources) VALUES (?, ?, ?, ?)",
        (query, json.dumps(query_embedding), response, json.dumps(sources))
    )
    conn.commit()
    conn.close()
```

### Updated query flow with cache in `rag.py`:

```python
def query(user_input: str) -> tuple[str, list, str]:
    """
    Returns (response, sources, cache_status)
    cache_status is "hit" or "miss" — used by Streamlit sidebar display
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    query_embedding = embeddings.embed_query(user_input)

    # Check cache first
    cached_response, cached_sources = check_cache(query_embedding)
    if cached_response:
        return cached_response, cached_sources, "hit"

    # Cache miss — run full pipeline
    docs, scope = get_resilient_context(user_input, vectorstore, llm, metadata_field_info)
    ranked_docs = rerank_by_priority(docs)
    response = generate_response(user_input, ranked_docs)
    sources = build_citations(ranked_docs)

    # Store result for future queries
    store_in_cache(user_input, query_embedding, response, sources)

    return response, sources, "miss"
```

### Streamlit Sidebar Display

```python
# In app.py
response, sources, cache_status = rag.query(user_input)

cache_label = "⚡ Cache Hit" if cache_status == "hit" else "🔍 Full Pipeline"
st.sidebar.markdown(f"**Response Source:** {cache_label}")
```

### Threshold Selection
- `0.95` — only return cached answer for near-identical queries. Safe default.
- `0.90` — slightly more aggressive; may occasionally return a cached answer for a related-but-different question. Test carefully before lowering.
- Never go below `0.85` for a compliance tool — the risk of returning a cached answer to a subtly different question is too high.

### Add to `config.py`

```python
SIMILARITY_THRESHOLD_CACHE = 0.95   # Cache hit threshold — keep high for compliance tools
CACHE_DB_PATH = "./cache/semantic_cache.db"
```

---

## Updated Build Order (Revised)

```
Phase 1 — MVP
  1. Ollama + Qwen3.5 9B setup
  2. Python environment + dependencies
  3. ingest.py — basic PDF + web ingestion with metadata
  4. rag.py — basic top-K retrieval + generation
  5. app.py — Streamlit UI, question mode + review mode

Phase 2 — Resilient Retrieval
  6. Priority-weighted reranker (sort by metadata priority)
  7. Self-querying with ambiguity fallback
  8. Parent Document Retrieval (child search, parent context)
  9. Contextual Compression — EmbeddingsFilter first

Phase 3 — Performance & Polish
  10. Semantic cache (SQLite + cosine similarity)
  11. Document upload UI (drag-and-drop ingest from Streamlit)
  12. Conversation memory (sliding window)
  13. Strict mode toggle in UI
  14. Source Scope + Cache Status sidebar indicators
```

---

## How These Four Improvements Tell a Portfolio Story

Each layer addresses a real, specific problem:

| Layer | Problem Solved | Technique | Complexity |
|---|---|---|---|
| Priority Reranker | Authority ignored by vector search | Metadata sort | Trivial |
| Parent Doc Retrieval | Context split at chunk boundaries | Two-level index | Low |
| Contextual Compression | Noisy context wastes RAM and attention | Embedding filter | Low-Medium |
| Semantic Cache | Redundant LLM calls waste compute | SQLite + cosine sim | Medium |

Together they show a progression from "I followed a tutorial" to "I identified specific failure modes and addressed them deliberately." That's the difference between a demo and an architecture.

In an interview, you can walk through this progression in 90 seconds:
> "Basic retrieval works but ignores document authority, so I added a priority sort. Chunk boundaries were cutting rules in half, so I implemented parent document retrieval. The context window was noisy, so I added embedding-based compression. And repeated queries were burning unnecessary compute, so I added a semantic cache with a 0.95 similarity threshold to avoid false hits in a compliance context."

Each sentence is a problem-solution pair. That's what senior-level engineers sound like.
