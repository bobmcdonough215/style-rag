# Style RAG — Developer Guide

Comprehensive handoff document for the next developer or agent picking up this project. Covers architecture, every file in detail, roadblocks encountered and how they were resolved, what's left to build, and critical gotchas.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File-by-File Reference](#file-by-file-reference)
4. [Data Flow](#data-flow)
5. [Hardware & Environment](#hardware--environment)
6. [Roadblocks Encountered & Fixes](#roadblocks-encountered--fixes)
7. [What's Built (Current State)](#whats-built-current-state)
8. [What's Left to Build](#whats-left-to-build)
9. [Critical Gotchas](#critical-gotchas)
10. [Dependency Versions](#dependency-versions)
11. [Git History](#git-history)

---

## Project Overview

Style RAG is a local-first RAG pipeline serving two audiences:

- **Editorial teams** — checking AP style, APA, Chicago, and internal house style rules
- **Web content teams** — looking up CMS workflows (Sitecore), HTML component standards, content governance, and distribution requirements

All generation runs locally via Ollama (Qwen 3.5 9B). Sensitive document content never leaves the machine. Only embeddings use a cloud API (OpenAI `text-embedding-3-small`).

The app has two modes:
- **Question Mode** — ask a question, get a grounded answer with citations
- **Review Mode** — paste content, get AP style violations flagged with corrections and a fully corrected version

---

## Architecture

```
                           +-------------------+
                           |    Streamlit UI   |
                           |     (app.py)      |
                           +--------+----------+
                                    |
                    +---------------+---------------+
                    |                               |
             Cache Check                    Full RAG Pipeline
             (cache.py)                         (rag.py)
                    |                               |
            SQLite cosine sim              +--------+--------+
            >= 0.95 threshold              |                 |
                    |                  Retrieve          Generate
              Cache Hit?             (Chroma)        (Ollama/Qwen)
               /      \                |                 |
             Yes       No         OpenAI embed      Local LLM
              |         |          + threshold       reasoning=False
         Instant     Full pipeline  + rerank         num_predict=1200
         response    (see right)    by priority
                                        |
                                  Top 4 chunks
                                  priority-sorted
                                        |
                                  Prompt Assembly
                                  (system + context + question)
                                        |
                                  Stream to UI
                                  with _filter_thinking()
                                        |
                                  Programmatic Citations
                                  (from chunk metadata, never LLM)
```

### Priority System

The reranker sorts retrieved chunks by `priority` metadata before they enter the context window. Lower number = higher authority = appears first. LLMs exhibit primacy bias — they weight earlier context more heavily.

```
Priority 1 — House style (docs/house-style/)     → Always wins
Priority 2 — Internal process docs (docs/web-content/)  → Sitecore how-tos
Priority 3 — External style guides (docs/editorial/)    → AP, APA, Chicago
Priority 4 — External standards (web sources)            → WCAG, Plain Language
```

### Hybrid Architecture

OpenAI and Ollama never interact. LangChain orchestrates them independently:

```
Ingest:  PDF/Web → OpenAI embeds chunks → Chroma stores vectors + metadata
Query:   Question → OpenAI embeds query → Chroma similarity search → Ollama generates answer
```

---

## File-by-File Reference

### `config.py` — Central Configuration

Every tunable parameter lives here. Nothing is hardcoded in other files.

**Key sections:**
- **Model & Hardware** — `LLM_MODEL = "qwen3.5:9b"`, `EMBEDDING_MODEL = "text-embedding-3-small"`
- **Paths** — vectorstore, cache DB, three document folders (house-style, editorial, web-content)
- **Generation** — `TEMPERATURE = 0.1` (near-deterministic for compliance), `ENABLE_THINKING = False` (critical — see Roadblocks)
- **Retrieval** — `TOP_K_CHUNKS = 4`, `SIMILARITY_THRESHOLD = 0.30`, `FETCH_K_MULTIPLIER = 3`
- **Chunking** — four doc-type-specific configs (style: 500/50, process: 800/100, standards: 500/75, template: 600/50)
- **Prompts** — all system prompts for all modes live here, isolated from logic

**Why `SIMILARITY_THRESHOLD = 0.30`:** Chroma + OpenAI embeddings produce relevance scores in the 0.35-0.45 range for clearly relevant content. The original 0.7 threshold filtered out everything. We benchmarked across multiple query types to find this value. See Roadblocks section.

**Why `ENABLE_THINKING = False`:** Maps to `reasoning=False` on OllamaLLM, which sends `think: false` at the top level of the Ollama API. This is critical — without it, Qwen 3.5 spends 80+ seconds on hidden thinking tokens before generating visible output. See Roadblocks section.

---

### `ingest.py` — Document Ingestion Pipeline

Run with `python ingest.py` (incremental) or `python ingest.py --rebuild` (full rebuild).

**What it does:**
1. Validates every source entry has required metadata keys (`source`, `audience`, `priority`, `doc_type`)
2. Loads checksum manifest — skips unchanged PDFs (no re-embedding cost)
3. Clears stale chunks from Chroma for any changed sources
4. Loads PDFs via `PyPDFLoader`, web pages via `WebBaseLoader` with `SoupStrainer` targeting `<article>` and `<main>` tags
5. Tags every page with source metadata before splitting
6. Applies doc-type-specific chunking (style vs process vs standards vs template)
7. Deduplicates chunks by content hash
8. Indexes into Chroma via `Chroma.from_documents()`
9. Saves checksum manifest only after successful indexing

**Source configuration:**
- `SOURCES` list at the top defines all explicit sources with curated metadata
- `_scan_directory()` auto-discovers PDFs in `docs/web-content/` — drop a new PDF in, re-run ingest, no config changes needed
- House style docs (Priority 1) and editorial docs (Priority 3) are listed explicitly because they need curated source names and different priority levels
- Web content docs (Priority 2) use auto-discovery because they share metadata

**Web source behavior:** `compute_checksum()` returns `""` for URLs, so `source_has_changed()` always returns `True` for web sources. This means web content is always re-fetched on ingest. This is intentional — remote content can change without notice.

---

### `rag.py` — Retrieval & Generation

Core RAG pipeline. Contains initialization helpers, retrieval, reranking, prompt assembly, citation building, and the programmatic query function.

**`init_llm()` — lines 61-69:**
```python
OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=TEMPERATURE,
    repeat_penalty=REPEAT_PENALTY,
    num_predict=MAX_TOKENS_REVIEW,  # 1200 — highest cap, model stops naturally
    reasoning=ENABLE_THINKING,      # False — disables thinking mode
)
```
- `num_predict=1200` is set to the review mode cap (highest). The model stops naturally for shorter answers. We do NOT use `.bind()` to change this per-mode because `.bind(options=...)` overrides `model_kwargs` entirely, which would lose `reasoning=False`.
- `reasoning=False` is the correct parameter name on `langchain_ollama==0.3.10`. This maps to the top-level `think: false` in the Ollama API. Previous attempts (`think=`, `model_kwargs.options.think`) failed silently. See Roadblocks.

**`retrieve()` — lines 74-97:**
- Fetches `TOP_K * FETCH_K_MULTIPLIER` (4 * 3 = 12) candidates
- Filters by `SIMILARITY_THRESHOLD` (0.30) — prevents hallucination from irrelevant chunks
- Returns top `TOP_K_CHUNKS` (4) after filtering
- `skip_threshold=True` bypasses filtering for review mode (prose inputs score too low against style guide chunks)

**`rerank_by_priority()` — lines 100-107:**
- Sorts chunks by `priority` metadata ascending (1 first, 4 last)
- Exploits LLM primacy bias — house style appears first in context

**`build_citations()` / `format_citations()` — lines 133-154:**
- Citations are NEVER generated by the LLM (it will hallucinate them)
- Extracted programmatically from chunk metadata
- Deduplicated by label

**`query()` — lines 168-224:**
- Programmatic entry point (used for testing, not by app.py which streams)
- app.py replicates this flow with streaming support

---

### `cache.py` — Semantic Cache

SQLite-backed cache that checks for semantically similar previous queries before calling the LLM.

**How it works:**
1. `init_cache()` creates the SQLite table if it doesn't exist
2. `check_cache()` loads all rows matching the current `mode` + `strict` setting, computes cosine similarity against each cached embedding, returns the first match >= 0.95
3. `store_in_cache()` saves the query, embedding, answer, and citations after a successful generation

**Design decisions:**
- **0.95 threshold** — only near-identical questions return cached answers. Safe for a compliance tool. Tested: rephrased questions ("How should I capitalize..." vs "What is AP style for capitalizing...") correctly miss the cache.
- **Mode/strict-aware** — a strict-mode answer won't be returned for a relaxed-mode query
- **Review mode skipped** — paste inputs are unique, caching them wastes storage
- **Linear scan** — `check_cache()` iterates all matching rows in Python. Fine for a local single-user tool. At 1000+ cached queries, consider adding a SQLite index or vector search extension.
- **Cache lives at `./cache/semantic_cache.db`** — gitignored

---

### `app.py` — Streamlit UI

Two-panel interface with question mode (chat input) and review mode (text area form).

**Startup sequence (lines 10-43):**
1. Check Ollama is running (`rag.check_ollama()`)
2. Check vectorstore exists
3. Load embeddings, vectorstore, LLM via `@st.cache_resource` (persists across reruns)
4. Initialize semantic cache

**Embeddings are exposed separately (lines 28-42):**
```python
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
```
The `_embeddings` parameter in `load_vectorstore(_embeddings)` uses the underscore prefix to tell Streamlit not to hash it (OpenAIEmbeddings isn't hashable). Embeddings are exposed as a separate cached resource so the cache can use `embeddings.embed_query()` independently of the vectorstore.

**`_filter_thinking()` — lines 116-134:**
Generator that intercepts `<think>` tags from the stream before they reach `st.write_stream()`. Even with `reasoning=False`, this is a safety net — Qwen 3.5 may occasionally emit thinking tags under certain prompt conditions.

**Query flow (lines 139-237):**
1. Embed query and check cache (question mode only)
2. If cache hit: display instantly, no LLM call
3. If cache miss: retrieve chunks, rerank by priority, stream response, build programmatic citations
4. Store result in cache (question mode only)
5. Append to conversation history, enforce sliding window (6 turns)
6. Update sidebar diagnostics (cache status, chunks used, citations)

**`st.stop()` in the except block (line 202):** Halts the entire app on generation failure. Fine for single-user local tool. For multi-user deployment, replace with early return logic.

---

## Data Flow

### Ingestion (runs once or on document change)

```
PDF / URL
  -> Load raw text (PyPDFLoader or WebBaseLoader + SoupStrainer)
  -> Tag every page with metadata (source, audience, priority, doc_type)
  -> Detect scanned pages (< 50 chars warning)
  -> Split into chunks (doc-type-specific sizes)
  -> Deduplicate by content hash
  -> Embed via OpenAI text-embedding-3-small
  -> Store in Chroma (vectors + metadata)
  -> Save checksum manifest
```

### Query (runs on every user input)

```
User input
  -> Embed query via OpenAI
  -> Check semantic cache (cosine sim >= 0.95)
  -> [HIT] Return cached answer instantly
  -> [MISS] Chroma similarity search (fetch 12, threshold 0.30, keep top 4)
  -> Rerank by priority metadata (house style first)
  -> Assemble prompt (system + context + question)
  -> Stream through Ollama/Qwen 3.5 (reasoning=False, num_predict=1200)
  -> Filter thinking tags from stream
  -> Display response + programmatic citations
  -> Store in cache for future queries
```

---

## Hardware & Environment

- **Machine:** MacBook Pro M4 with 24GB unified RAM
- **GPU config:** `sudo sysctl iogpu.wired_limit_mb=20480` (grants 20GB to GPU)
- **Model:** Qwen 3.5 9B via Ollama (~8.6GB when loaded)
- **Memory profile while running:** ~18.7GB used, green memory pressure, ~1.5GB swap
- **Comfortable headroom:** ~5GB free with the app running + a few browser tabs
- **Avoid:** Running Cursor + multiple AI chat tabs simultaneously while the model is loaded

**Python:** 3.9.6 (system Python on macOS 15.5)
**Note:** Python 3.9 does not support `dict | None` union syntax — use `Optional[dict]` from typing instead.

---

## Roadblocks Encountered & Fixes

### 1. Similarity Threshold Too High (0.7 -> 0.55 -> 0.30)

**Symptom:** Every query returned 0 chunks. The system always responded "This topic is not covered."

**Root cause:** Chroma's `similarity_search_with_relevance_scores()` with OpenAI embeddings produces scores in the 0.35-0.45 range for clearly relevant content. The initial 0.7 threshold (and later 0.55) filtered everything out.

**Debug process:** Ran raw similarity searches and printed scores:
```python
results = vs.similarity_search_with_relevance_scores('AP style dates', k=5)
for doc, score in results:
    print(f'{score:.3f} | {doc.metadata.get("source")}')
```
Top hit scored 0.366 — relevant content but far below 0.55.

**Fix:** Lowered `SIMILARITY_THRESHOLD` to 0.30 in config.py. Benchmarked across multiple query types: relevant hits 0.34+, tangential 0.27-0.33, noise below 0.25.

**Lesson:** Always benchmark your actual similarity score distribution before setting thresholds. Don't assume a "reasonable" value.

---

### 2. Qwen 3.5 Thinking Mode — 25x Slowdown (86s -> 3.6s)

**Symptom:** Queries took 60-90 seconds despite 100% GPU utilization.

**Root cause:** Qwen 3.5 was spending 80+ seconds generating hidden thinking tokens before producing visible output. The `think: false` flag was being placed inside `options` in the Ollama API request, where Ollama silently ignores it. It needs to be a **top-level** parameter.

**Three failed attempts:**
1. `think=ENABLE_THINKING` as a direct OllamaLLM parameter — silently ignored
2. `model_kwargs={"options": {"think": False}}` — placed inside options, Ollama ignores it
3. `.bind(options={"think": False})` — overrides model_kwargs entirely, loses other settings

**The fix:** `reasoning=False` on OllamaLLM (langchain_ollama 0.3.10). This is a first-class parameter that maps to the top-level `think: false` in the Ollama API.

**How we confirmed it:** Benchmarked raw Ollama API:
```bash
# Inside options — BROKEN (86 seconds)
curl -d '{"model":"qwen3.5:9b","prompt":"test","options":{"num_predict":50,"think":false}}'

# Top-level — WORKS (3 seconds)
curl -d '{"model":"qwen3.5:9b","prompt":"test","think":false,"options":{"num_predict":50}}'
```
The Ollama response JSON shows `eval_duration` of ~2 seconds for 50 tokens, but `total_duration` was 86 seconds when thinking was active — the difference was hidden thinking tokens.

**Lesson:** When an Ollama parameter appears to be ignored, check whether it's a top-level API parameter vs an `options` sub-parameter. The Ollama API documentation distinguishes these, but LangChain abstractions can obscure the mapping.

---

### 3. `.bind(options=...)` Overrides model_kwargs

**Symptom:** Empty responses after adding `.bind(options={"num_predict": N})` to the chain.

**Root cause:** LangChain's `.bind(options={...})` replaces the entire options dict from `model_kwargs`, not merges. This lost `think: False`, causing the model to spend all tokens on thinking with nothing left for visible output.

**Fix:** Removed `.bind()` entirely. Set `num_predict=MAX_TOKENS_REVIEW` (1200) at init time. The model stops naturally for shorter answers — no per-mode token cap needed.

---

### 4. Review Mode Retrieval Failure

**Symptom:** Review mode always returned "This topic is not covered." The pasted prose scored 0.13 against style guide chunks — well below the 0.30 threshold.

**Root cause:** Similarity search compares the semantic meaning of the query against chunk embeddings. A sentence like "On September 3rd, Vice President John Davis announced..." doesn't semantically match "AP style abbreviates months with six or more letters." The content is *about* a topic the style guide covers, but it doesn't *resemble* the style guide text.

**Fix:** Added `skip_threshold` parameter to `retrieve()`. Review mode passes `skip_threshold=True`, which returns the top K chunks regardless of similarity score. The model needs broad style coverage for reviews, not narrow matching.

---

### 5. WCAG URL Returned 0 Chunks

**Symptom:** The WCAG 2.2 source at `w3.org/TR/WCAG22/` returned no usable content.

**Root cause:** The W3C spec page structure didn't match the `SoupStrainer(["article", "main"])` targets.

**Fix:** Switched to the Quick Reference URL: `w3.org/WAI/WCAG22/quickref/`. This returned 321 chunks.

---

### 6. Python 3.9 Type Syntax

**Symptom:** `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` on `def check_cache(...) -> dict | None:`

**Root cause:** The `X | Y` union syntax requires Python 3.10+. The system Python is 3.9.6.

**Fix:** Import `Optional` from `typing` and use `Optional[dict]` instead.

---

## What's Built (Current State)

### Phase 1 — MVP (Complete)
- [x] Ollama + Qwen 3.5 9B local inference
- [x] `ingest.py` — PDF + web ingestion with metadata tagging, checksum skip, dedup, schema validation, `--rebuild` flag, auto-discovery for web-content PDFs
- [x] `rag.py` — retrieval with fetch-and-filter pattern, priority reranker, programmatic citations, prompt assembly
- [x] `app.py` — Streamlit UI with question + review modes, streaming with thinking filter, sliding window memory, sidebar diagnostics
- [x] `config.py` — all parameters, prompts, and paths centralized
- [x] Strict mode toggle (answers only from indexed docs vs supplementing with general knowledge)

### Phase 2 — Retrieval Improvements (Partial)
- [x] Priority-weighted reranker (Layer 2a)
- [x] Review mode skip-threshold fix
- [ ] Self-querying with metadata filtering (Layer 2 — see below)
- [ ] Parent Document Retrieval (Layer 2b — see below)
- [ ] Contextual Compression (Layer 2c — see below)

### Phase 3 — Performance & Polish (Partial)
- [x] Semantic cache with SQLite + cosine similarity (Layer 3)
- [x] Cache status in sidebar diagnostics
- [x] Conversation memory (sliding window, 6 turns)
- [ ] Document upload UI (drag-and-drop ingest from Streamlit)
- [ ] Source Scope sidebar indicator

### Additional Completed Work
- [x] House style documents (Priority 1) — "Meridian Partners LLP" fictional firm
- [x] Three house style PDFs: Web Content Format Standards, Insights & Distribution Standards, HTML Component Reference
- [x] Document folder reorganization: `docs/house-style/`, `docs/editorial/`, `docs/web-content/`
- [x] `cache.py` extracted as separate module (was briefly in rag.py, refactored for clean separation)
- [x] `_filter_thinking()` stream interceptor as safety net
- [x] `strip_thinking()` post-processing for non-streamed responses

---

## What's Left to Build

### Self-Querying with Metadata Filtering (High Impact)

**Problem:** If someone asks a Sitecore CMS question, the retriever might pull AP style chunks alongside Sitecore docs because they happened to score similarly. The context window gets polluted with irrelevant content.

**Solution:** Use the LLM to analyze the user's intent and apply metadata filters *before* the vector search. If the query is about CMS workflows, filter to `audience: "web-content"` first. If it's about style rules, filter to `audience: "editorial"`.

**Implementation approach:** LangChain's `SelfQueryRetriever` with a fallback to unfiltered search if the self-query returns 0 results (the "resilient retriever" pattern from the planning docs).

**Reference:** `style-rag-plan-v2.md` section "Pre-Filtering Query Logic (Self-Querying)" and `style-rag-advanced-retrieval.md`.

---

### Parent Document Retrieval (Medium Impact)

**Problem:** A 500-token chunk is optimized for retrieval precision, but style rules sometimes get split across chunk boundaries. The LLM sees half the context.

**Solution:** Store small "child" chunks (200 tokens) for high-accuracy search, but when a match is found, retrieve the full "parent" chunk (800 tokens) for complete context.

**Implementation:** LangChain's `ParentDocumentRetriever` with a persistent docstore (SQLite or `LocalFileStore`). Requires changes to `ingest.py` (two-level splitting) and a full re-ingest.

**Gotcha:** `InMemoryStore` does not persist across restarts. Use `LocalFileStore("./docstore")` for persistence.

**Reference:** `style-rag-advanced-retrieval.md` Layer 2b.

---

### Contextual Compression (Medium Impact)

**Problem:** 4 chunks x 500 tokens = 2,000 tokens of context. Only 50-100 contain the actual answer. The rest wastes context window and KV cache RAM.

**Solution:** Wrap the retriever in a `ContextualCompressionRetriever` with `EmbeddingsFilter`. Before chunks reach the LLM, irrelevant sentences are stripped.

**Recommendation:** Start with `EmbeddingsFilter` (no extra LLM call, uses OpenAI embeddings). Only upgrade to `LLMChainFilter` if quality is insufficient.

**Reference:** `style-rag-advanced-retrieval.md` Layer 2c.

---

### Document Upload UI (Low Impact, High Polish)

**Problem:** Adding new documents requires terminal access to run `ingest.py`.

**Solution:** Streamlit file uploader widget that saves the PDF to the appropriate folder and triggers ingestion. Would need to determine metadata (audience, priority, doc_type) either from user input or folder selection.

---

## Critical Gotchas

### DO NOT use `.bind()` on the LLM chain
`.bind(options={...})` replaces `model_kwargs` entirely. This loses `reasoning=False` and causes 80+ second delays or empty responses. Set everything at init time.

### DO NOT raise the similarity threshold above 0.35
Chroma + OpenAI embeddings score relevant content at 0.35-0.45. A threshold above 0.35 will filter out valid results. If retrieval quality seems poor, debug by printing raw scores — don't blindly raise the threshold.

### The `reasoning` parameter is version-specific
`reasoning=False` works on `langchain_ollama==0.3.10`. If you upgrade the package, verify this parameter still maps correctly. The Ollama API parameter is `think` (top-level, boolean).

### Web sources are always re-fetched
`compute_checksum()` returns `""` for URLs, so they're never skipped by the checksum system. This is intentional but means a full ingest always re-fetches and re-embeds web content (costs OpenAI credits).

### Python 3.9 compatibility
No `X | Y` union types. No `match` statements. Use `Optional[X]` from typing and `if/elif` chains.

### Review mode skips similarity threshold
This is intentional. Prose inputs don't semantically match style guide chunks. If you modify `retrieve()`, preserve the `skip_threshold` parameter.

### Checksums save only after successful indexing
`save_checksums()` is called in `main()` after `Chroma.from_documents()` succeeds. If indexing fails mid-run, no checksums are updated, so the next run will retry all changed sources.

### House style docs are fictional
The "Meridian Partners LLP" firm is made up. The documents contain realistic but fictional formatting rules for blogs, expert analysis, white papers, insights, and outside publications. They exist to demonstrate the priority system and provide rich content for the RAG to query.

---

## Dependency Versions

Exact versions from the working environment:

```
Python                    3.9.6
macOS                     15.5
streamlit                 1.50.0
langchain                 0.3.28
langchain-chroma          0.2.6
langchain-community       0.3.31
langchain-core            0.3.83
langchain-ollama          0.3.10
langchain-openai          0.3.35
langchain-text-splitters  0.3.11
chromadb                  1.5.2
openai                    2.26.0
ollama                    0.6.1
httpx                     0.28.1
beautifulsoup4            4.14.3
numpy                     2.0.2
fpdf2                     2.8.4 (dev only — used to generate house style PDFs)
```

---

## Git History

```
18def81 Add house style documents and reorganize doc structure
5182547 Add semantic cache for near-instant repeated query responses
d7f4018 Fix 25x inference slowdown and tune retrieval for both modes
3d4f660 Fix retrieval and generation pipeline
5984c44 Initial commit — Phase 1 MVP
```

---

## Planning Documents

These files exist in the project root and contain the original architecture decisions, prompt iteration logs, and advanced retrieval patterns. They are read-only reference — do not modify.

- `style-rag-plan-v2.md` — Full project plan, feature set, build order, tech stack
- `style-rag-advanced-retrieval.md` — Priority reranker, parent doc retrieval, contextual compression, semantic cache
- `style-rag-prompts.md` — All system prompts with documentation and iteration log
- `style-rag-inference-and-roadblocks.md` — Model parameters, failure modes, tuning notes

---

## Quick Start for the Next Developer

```bash
# 1. Install Ollama and pull the model
ollama pull qwen3.5:9b

# 2. Grant GPU memory (M4 Pro with 24GB)
sudo sysctl iogpu.wired_limit_mb=20480

# 3. Set up environment
cd /path/to/RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. Ingest documents
python ingest.py --rebuild

# 6. Start the app
streamlit run app.py

# App runs at http://localhost:8501
```
