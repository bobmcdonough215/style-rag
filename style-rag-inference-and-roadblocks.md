# Style RAG — Inference Parameters, Tuning & Roadblocks

A thorough pre-build reference covering model behavior configuration, retrieval tuning, known failure modes, and how to address them before they become problems mid-build.

---

## 1. Qwen3.5 Thinking Mode

Qwen3.5 9B has "thinking mode" enabled by default. Before responding, it generates an internal reasoning chain wrapped in `<think>...</think>` tags. This is powerful for complex reasoning tasks but is almost always counterproductive for a style guide RAG tool where answers are factual lookups, not multi-step logic problems.

### The Problem
- Thinking mode adds 10–45 seconds of latency before the first token of the actual response appears
- For a question like "is it 'website' or 'Website'?" the model does not need to reason — it needs to retrieve and report
- Thinking tokens also consume context window space, leaving less room for retrieved chunks
- The `<think>` content will appear in the raw output if not stripped, breaking your UI

### The Solution
Disable thinking mode by default. Expose it as an optional toggle for complex queries if you want, but off should be the default state.

**In Ollama via Modelfile:**
```
PARAMETER num_predict 600
SYSTEM "Do not use extended reasoning. Answer directly and concisely based only on the provided context."
```

**In LangChain via model parameters:**
```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="qwen3.5:9b",
    temperature=0.1,
    num_predict=600,
    system="Answer directly without extended reasoning."
)
```

**Stripping think tags as a safety net:**
Even with thinking disabled, add a post-processing step to strip any `<think>` content from responses before they reach the UI:
```python
import re

def strip_thinking(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
```

Add this to `rag.py` as a utility function called on every response before it's returned.

---

## 2. Full config.py Reference

Everything tunable lives here. Never hardcode model parameters in `rag.py` or `app.py`.

```python
# ─── Model ────────────────────────────────────────────────────────────────────
LLM_MODEL = "qwen3.5:9b"
EMBEDDING_MODEL = "text-embedding-3-small"
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Thinking Control ─────────────────────────────────────────────────────────
ENABLE_THINKING = False        # Disable Qwen3.5 thinking mode
THINKING_BUDGET_TOKENS = 0     # 0 = off; set to n to cap thinking token spend

# ─── Generation ───────────────────────────────────────────────────────────────
TEMPERATURE = 0.1              # Low = deterministic; high = creative/risky
MAX_TOKENS_QUESTION = 600      # Cap for question mode responses
MAX_TOKENS_REVIEW = 1200       # Cap for content review mode (needs more room)
MAX_TOKENS_FORMAT = 800        # Cap for format generation mode
REPEAT_PENALTY = 1.1           # Penalize repetitive output; 1.0 = off

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_CHUNKS = 4               # How many chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.7     # Minimum similarity score (0–1); below = discard
FETCH_K_MULTIPLIER = 3         # Fetch TOP_K * this, then re-rank; improves quality

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE_STYLE = 500
CHUNK_OVERLAP_STYLE = 50
CHUNK_SIZE_PROCESS = 800
CHUNK_OVERLAP_PROCESS = 100
CHUNK_SIZE_STANDARDS = 500
CHUNK_OVERLAP_STANDARDS = 75
CHUNK_SIZE_TEMPLATE = 600
CHUNK_OVERLAP_TEMPLATE = 50

# ─── Behavior ─────────────────────────────────────────────────────────────────
STRICT_MODE_DEFAULT = True     # Answer only from indexed docs by default
MAX_CONVERSATION_TURNS = 6     # How many turns to keep in memory
SOURCE_CITATION_ENABLED = True # Always show source attribution

# ─── Paths ────────────────────────────────────────────────────────────────────
VECTORSTORE_PATH = "./vectorstore"
DOCS_PATH_EDITORIAL = "./docs/editorial"
DOCS_PATH_WEB_CONTENT = "./docs/web-content"

# ─── Priority Levels ──────────────────────────────────────────────────────────
PRIORITY_HOUSE_STYLE = 1
PRIORITY_INTERNAL_PROCESS = 2
PRIORITY_EXTERNAL_STYLE = 3
PRIORITY_EXTERNAL_STANDARDS = 4
```

---

## 3. Temperature — Why It Matters More in RAG Than You Think

Temperature controls randomness in generation. Most tutorials leave it at default (0.7–1.0), which is fine for creative tasks but actively harmful for a compliance tool.

| Temperature | Behavior | Use Case |
|---|---|---|
| `0.0` | Fully deterministic | Brittle; identical input always gives identical output |
| `0.1` | Near-deterministic | **Recommended for style/process RAG** |
| `0.3` | Slight variation | Acceptable for format generation |
| `0.7+` | Creative, variable | Good for writing assistants, bad for rule lookups |

**The specific risk at high temperature:** The model knows AP style from training data. At high temperature it may blend its training knowledge with retrieved content and produce a confident answer that's slightly wrong — a rule from AP 2018 when your guide is AP 2024, for example. At 0.1 it sticks closer to what was actually retrieved.

**Set it to 0.1 and leave it there until you have a specific reason to change it.**

---

## 4. Self-Querying — Scoping Retrieval by Audience

Since the pipeline serves two distinct audiences (editorial and web content), your metadata strategy can do more than just tag documents — it can actively narrow the search space before similarity scoring runs.

**The problem without self-querying:** A web content team member asks "how do I add an alternating list in Sitecore?" Chroma searches all chunks — including AP Stylebook, APA, and Chicago Manual entries — before returning the 4 most similar results. The editorial chunks dilute retrieval quality and fill your context window with irrelevant content.

**Self-querying fixes this:** LangChain's `SelfQueryRetriever` can parse the intent of a query and automatically apply metadata filters before similarity search. If the query is clearly a CMS question, it scopes retrieval to `audience == "web-content"` only. If it's a style question, it scopes to `audience == "editorial"`. This keeps context clean and answers precise.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="audience", description="Either 'editorial' or 'web-content' or 'both'", type="string"),
    AttributeInfo(name="doc_type", description="One of: style, process, template, standards", type="string"),
    AttributeInfo(name="priority", description="Integer 1-4, where 1 is highest priority", type="integer"),
]

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    "Style guides and web content standards documentation",
    metadata_field_info,
)
```

**Why this matters for your Mac:** Filtering by audience before similarity search means fewer chunks are scored, less content is loaded into RAM, and your context window stays clean. On a 24GB machine this is a performance win as well as a quality win.

**Add this in Layer 2** — build basic retrieval first, then upgrade to self-querying once the pipeline is stable.

### The Ambiguity Fallback Pattern

Self-querying relies on the LLM correctly classifying the query's intent. Ambiguous queries that straddle both audiences — for example, "How do I format an H3 for the blog?" — may be classified too narrowly, scoping retrieval to `web-content` only and missing a house style capitalization rule that lives in the editorial docs.

This is not a hypothetical edge case. It will happen in real use. Plan for it explicitly.

**The resilient retriever pattern — try filtered, fallback to global:**

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from config import TOP_K_CHUNKS, SIMILARITY_THRESHOLD

def get_resilient_context(query, vectorstore, llm, metadata_field_info):
    """
    Attempts a filtered search first to save RAM and context window space.
    Falls back to global unfiltered search if filtered results are insufficient.
    """

    # 1. Try: filtered search via self-querying
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        "Style guides and web operations documentation",
        metadata_field_info,
        verbose=True
    )
    initial_docs = retriever.get_relevant_documents(query)

    # 2. Check: validate result quality against similarity threshold
    # NOTE: Use similarity_search_with_relevance_scores to get actual scores
    # get_relevant_documents does NOT return scores by default in Chroma
    valid_docs = [
        doc for doc in initial_docs
        if doc.metadata.get("score", 0.0) >= SIMILARITY_THRESHOLD  # fail closed
    ]

    if len(valid_docs) >= 2:
        return valid_docs, "filtered"

    # 3. Fallback: unfiltered global similarity search
    print("Self-querying returned insufficient results. Falling back to global search...")
    fallback_docs = vectorstore.similarity_search(query, k=TOP_K_CHUNKS)
    return fallback_docs, "global"
```

### Chroma Score Caveat — Critical Implementation Note

Chroma's `get_relevant_documents()` does **not** return similarity scores by default. The `doc.metadata.get("score")` check above will return `None` for every chunk unless you explicitly use the scored method:

```python
# Use this instead of similarity_search when you need scores
docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
    query, k=TOP_K_CHUNKS
)

# Returns list of (Document, float) tuples
valid_docs = [
    doc for doc, score in docs_with_scores
    if score >= SIMILARITY_THRESHOLD
]
```

Update your retrieval logic to use `similarity_search_with_relevance_scores` wherever threshold filtering is applied. Without this, your threshold check is silently doing nothing — every chunk passes because the default score is missing, not because it's actually relevant.

### Portfolio Narrative for This Pattern

When documenting this in your README or discussing it in an interview, frame it as a deliberate architectural tradeoff:

> "The pipeline uses a two-step retrieval strategy: it first attempts a metadata-filtered search scoped to the detected audience (editorial or web content) to minimize RAM usage and keep the context window focused. If the filtered search returns fewer than two chunks above the similarity threshold — indicating an ambiguous or cross-domain query — it falls back to a global unfiltered search to maximize recall. This balances computational efficiency with retrieval completeness."

This framing demonstrates you understand that RAG is a probabilistic system, not a deterministic lookup — which is the difference between a junior demo and a production-minded implementation.

### Streamlit Logging — Show Your Work

Add a small "Source Scope" indicator to the Streamlit sidebar showing whether the current response used filtered or global retrieval. This makes the system's decision-making visible to users and to anyone you demo the project to:

```python
# In app.py — after calling get_resilient_context
docs, scope = get_resilient_context(query, vectorstore, llm, metadata_field_info)

# Display in sidebar
scope_label = "🎯 Filtered" if scope == "filtered" else "🌐 Global Fallback"
st.sidebar.markdown(f"**Source Scope:** {scope_label}")
```

Small detail, high signal. A recruiter or hiring manager watching a demo will notice that the app is transparent about how it's reasoning — that's a product instinct, not just a technical one.

---

## 5. Retrieval Tuning — The Most Common RAG Failure Point

Most RAG bugs are not LLM bugs. They're retrieval bugs. The model can only answer well if the right chunks are retrieved. If the wrong chunks come back, the best LLM in the world will give you a bad answer.

### Chunk Size Calibration
The most important pre-build decision after model selection.

**Too small (< 200 tokens):**
- Retrieves fragments without enough context to be useful
- A rule about Oxford commas might be split across two chunks, neither of which is complete
- Citation quality suffers — "from page 4" without surrounding context

**Too large (> 1000 tokens):**
- Retrieved chunks contain the right answer buried in irrelevant surrounding content
- More tokens passed to the model = slower generation
- Similarity scores become less precise because the chunk covers too many topics

**The sweet spot for style guides is 400–600 tokens.** Style rules are typically 2–5 sentences. A 500 token chunk captures the rule, its exception, and one example — exactly what you need.

**Process docs are different.** A Sitecore how-to step has dependencies on the previous step. 800 tokens with 100 token overlap ensures step 3 always has context from step 2's ending.

### The Fetch-and-Rerank Pattern
Instead of retrieving exactly TOP_K chunks, retrieve `TOP_K * 3` chunks and then rerank them by relevance before passing the top K to the model. This dramatically improves retrieval quality at minimal cost.

```python
# In rag.py
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": TOP_K_CHUNKS * FETCH_K_MULTIPLIER,  # Fetch more
    }
)
# Then rerank and slice to TOP_K before passing to LLM
```

LangChain has a `ContextualCompressionRetriever` with reranking built in. Worth adding in Layer 2.

### Similarity Threshold
Without a threshold, Chroma returns the top K chunks regardless of how relevant they are. If you ask "how do I add an alternating list in Sitecore?" and none of your indexed docs cover that yet, Chroma will still return 4 chunks — just the least-wrong ones. The LLM will then hallucinate an answer from those irrelevant chunks.

A similarity threshold of 0.7 means: only return chunks that are at least 70% similar to the query. If nothing clears 0.7, return nothing and let strict mode handle it gracefully ("Not covered in your indexed guides").

**Start at 0.7, tune down to 0.65 if you're getting too many "not found" responses on questions you know are covered.**

---

## 6. Context Window Budget

With Qwen3.5 9B on 24GB RAM via Ollama, your practical context window is large but not unlimited. Here's how to think about the budget:

```
System prompt:         ~200  tokens
Retrieved chunks (4x): ~2000 tokens (4 × 500 avg)
Conversation history:  ~800  tokens (last 3 turns)
User query:            ~50   tokens
─────────────────────────────────────
Total input:           ~3050 tokens

Reserved for output:   600   tokens (question mode)
─────────────────────────────────────
Total per query:       ~3650 tokens
```

At 3,650 tokens per query you are well within safe limits. The risk comes from:
- Review mode where the user pastes a long document (could be 1,000+ tokens of input)
- Conversation memory accumulating over many turns
- Retrieving 6+ large chunks

**Safeguards to build in:**
- Cap pasted content at 1,500 tokens in review mode; truncate with a warning if exceeded
- Cap conversation memory at 6 turns; drop oldest turns first (sliding window)
- Monitor total context length in `rag.py` and log a warning if it exceeds 6,000 tokens

---

## 7. System Prompt Design

The system prompt is the most important prompt engineering decision in the project. It sets the model's behavior for every single query.

### Base System Prompt (Strict Mode On)
```
You are a style and web content standards assistant for an editorial and web content team.

Answer questions based ONLY on the context provided below. Do not use general knowledge or information from your training data.

If the answer is not present in the provided context, respond exactly with:
"This topic is not covered in your indexed guides. Consider adding documentation for it."

Always cite your source at the end of your response in this format:
*Source: [Document Name], [Section or Page if available]*

If multiple sources are relevant, cite all of them.

Be concise. Do not pad responses. Do not editorialize.
```

### Base System Prompt (Strict Mode Off)
```
You are a style and web content standards assistant for an editorial and web content team.

Prioritize the context provided below when answering. If the context covers the topic, answer from it and cite your source. If the context does not fully cover the topic, you may supplement with general knowledge but clearly indicate which parts of your answer come from indexed guides and which come from general knowledge.

Always cite your source for context-derived information:
*Source: [Document Name], [Section or Page if available]*

Be concise. Do not pad responses.
```

### Review Mode System Prompt
```
You are a copy editor and web standards reviewer.

Review the content provided by the user for style, grammar, and web standards issues. Base your review ONLY on the context provided below.

Format your response as:
1. A numbered list of issues found, each with:
   - The original text
   - The issue and the rule it violates
   - The corrected version
2. A fully corrected version of the entire input at the end

If no issues are found, say so explicitly.
Cite the source of each rule applied.
```

**Keep system prompts in `config.py` as constants, not hardcoded in `rag.py`.** You will iterate on them constantly.

---

## 8. Known Failure Modes & How to Handle Them

### Failure: Model ignores retrieved context and answers from training data
**Symptom:** Response is confident but doesn't cite any source, or cites something not in your docs.
**Cause:** Temperature too high, or system prompt not strong enough.
**Fix:** Lower temperature to 0.1. Strengthen system prompt with explicit "ONLY use the context below" language. Add strict mode by default.

### Failure: "Not found" responses on questions that are definitely covered
**Symptom:** Model says it can't find the answer but you know the doc is indexed.
**Cause:** Similarity threshold too high, or the question phrasing is too different from the indexed text.
**Fix:** Lower threshold from 0.7 to 0.65. Check if the chunk containing the answer is actually in Chroma with a debug query. Consider re-phrasing the indexed document or adding synonym handling.

### Failure: Slow responses (10+ seconds per query)
**Symptom:** Acceptable during development, unacceptable in a demo.
**Cause 1:** Thinking mode is on. **Fix:** Disable it.
**Cause 2:** MAX_TOKENS too high. **Fix:** Cap at 600 for question mode.
**Cause 3:** Too many chunks being retrieved. **Fix:** Drop TOP_K from 6 to 4.
**Cause 4:** RAM pressure from other apps. **Fix:** Close Chrome tabs. Seriously — each tab can use 200–500MB on macOS.

### Failure: Responses get progressively slower in a long session
**Symptom:** First query takes 3s, tenth query takes 12s.
**Cause:** Conversation memory is accumulating context. The model is processing an increasingly long input.
**Fix:** Implement sliding window memory — keep only the last N turns. Set MAX_CONVERSATION_TURNS = 6.

### Failure: Hallucinated citations
**Symptom:** Model cites "AP Stylebook, p. 47" but p. 47 doesn't say that, or the page doesn't exist.
**Cause:** Model is generating plausible-sounding citations from training data rather than from retrieved chunks.
**Fix:** Never ask the model to generate citations. Instead, extract source metadata from the retrieved chunks in `rag.py` and append them programmatically. The model only generates the answer text; your code supplies the citations.

```python
# In rag.py — build citations from chunk metadata, not LLM output
def format_citations(docs):
    seen = set()
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen:
            seen.add(source)
            citations.append(f"*Source: {source}*")
    return "\n".join(citations)
```

### Failure: PDF text extraction is garbled or empty
**Symptom:** Chunks from a PDF contain gibberish, symbols, or are empty entirely.
**Cause:** The PDF is scanned (image-based), not text-based. PyPDFLoader cannot extract text from image PDFs.
**Fix:** Add a check during ingest that validates extracted text length. If a page returns fewer than 50 characters, flag it as a potential scan. For scanned PDFs you'll need OCR — `pytesseract` or `pdfplumber` with OCR fallback. Add this to the ingest pipeline as a fallback handler.

```python
# In ingest.py — basic scan detection
def is_likely_scanned(text: str) -> bool:
    return len(text.strip()) < 50
```

### Failure: Web scraping returns navigation, ads, or boilerplate instead of content
**Symptom:** Chunks contain "Subscribe now", "Home | About | Contact", cookie consent text.
**Cause:** WebBaseLoader pulls the full page HTML including nav and footer.
**Fix:** Use `bs4_strainer` to target only the main content area. For AP-style sites this usually means targeting `<article>`, `<main>`, or a specific class.

```python
from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer

loader = WebBaseLoader(
    url,
    bs_kwargs={"parse_only": SoupStrainer("article")}
)
```

### Failure: Duplicate content inflates retrieval results
**Symptom:** All 4 retrieved chunks come from the same document section; other relevant sources are crowded out.
**Cause:** The same content was indexed twice (e.g., PDF and scraped version of the same guide), or chunk overlap is creating near-duplicate chunks.
**Fix:** Add a deduplication check at ingest time using a hash of the chunk content. If the same hash already exists in Chroma, skip it.

```python
import hashlib

def chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
```

### Failure: Chroma vectorstore grows stale after documents are updated
**Symptom:** Model cites outdated rules from an old version of your house style guide.
**Cause:** You updated the PDF but didn't re-ingest it. Old chunks are still in Chroma alongside new ones.
**Fix:** On re-ingest, delete all chunks from that source before re-adding. Track document version in metadata.

```python
# In ingest.py — delete by source before re-indexing
vectorstore.delete(where={"source": "Acme Corp House Style Guide"})
```

---

## 9. Ollama-Specific Considerations

### First-run model load time
The first query after starting Ollama loads the model into memory. This takes 10–20 seconds on the 9B. After that it stays loaded and subsequent queries are fast. Build a loading state into your Streamlit UI so users know to wait on first load.

### Ollama must be running before the app starts
If Ollama isn't running, LangChain will throw a connection error. Add a health check at app startup:

```python
import httpx

def check_ollama():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except:
        return False
```

Show a clear error message in the UI if Ollama isn't running rather than letting the app crash silently.

### Model updates via Ollama
Running `ollama pull qwen3.5:9b` again will update the model if a new version is available. Behavior can change between versions. Pin your model version in config once you have a working setup:

```python
LLM_MODEL = "qwen3.5:9b"  # Pin this — note the exact version tag when you pull
```

---

## 10. OpenAI Embeddings Considerations

### Cost
`text-embedding-3-small` costs $0.02 per million tokens. For a style guide RAG project with a few hundred pages of documents, you'll spend pennies on the initial ingest. Ongoing query embedding is negligible.

### Embeddings are computed at ingest AND at query time
Every user query is also embedded via OpenAI before retrieval. This means:
- An internet connection is required for the app to function (even though the LLM is local)
- If OpenAI has an outage, your retrieval breaks even though Qwen is running fine locally

**Plan for this:** Add a try/except around the embedding call with a clear user-facing error message. Consider caching frequently-asked query embeddings locally.

### Don't mix embedding models
If you ingest documents with `text-embedding-3-small` and then later switch to a different model, the stored vectors are incompatible. You must re-ingest everything. Document your embedding model choice in `config.py` and treat it as a schema decision — don't change it without a full re-ingest.

---

## 11. Streamlit-Specific Considerations

### Streamlit reruns the entire script on every interaction
This is Streamlit's core behavior and it will bite you. If you initialize the Chroma vectorstore at the top of `app.py`, it will reconnect to it on every button click. Use `st.session_state` and `@st.cache_resource` to persist expensive objects across reruns:

```python
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL, temperature=TEMPERATURE)
```

Without this, every interaction triggers a fresh Chroma connection and a model reload attempt. Your app will be slow and unstable.

### Streaming responses
Qwen3.5 via Ollama supports token streaming — displaying words as they're generated rather than waiting for the full response. This dramatically improves perceived performance. LangChain supports streaming with a callback handler. Add this from the start, not as an afterthought:

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(
    model=LLM_MODEL,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True
)
```

In Streamlit use `st.write_stream()` for streaming output.

### Session state for conversation memory
Streamlit has no built-in memory between interactions. Store conversation history in `st.session_state`:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Pair this with MAX_CONVERSATION_TURNS to prevent unbounded memory growth.

---

## 12. Environment & Dependency Management

### Use a virtual environment
Never install project dependencies globally. Create a dedicated environment:

```bash
python -m venv venv
source venv/bin/activate       # On Mac
pip install -r requirements.txt
```

### requirements.txt (starting point)
```
langchain
langchain-community
langchain-openai
chromadb
ollama
streamlit
pypdf
beautifulsoup4
python-dotenv
httpx
```

Pin versions once you have a working setup. Unpinned dependencies will break your project when packages update.

### .env file
```
OPENAI_API_KEY=sk-...
```

Never commit this to GitHub. Add `.env` to `.gitignore` immediately. Use `python-dotenv` to load it:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 13. Pre-Build Checklist

Before writing a single line of application code, verify:

- [ ] Ollama installed and running (`ollama serve`)
- [ ] Qwen3.5 9B pulled (`ollama pull qwen3.5:9b`)
- [ ] Test query returns a fast response in Ollama CLI (`ollama run qwen3.5:9b`)
- [ ] Python virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] OpenAI API key in `.env` and loading correctly
- [ ] Test embedding call returns a vector (sanity check before full ingest)
- [ ] At least one PDF and one scraped URL ready to test ingest with
- [ ] `.gitignore` includes `.env` and `vectorstore/`
- [ ] `config.py` created with all parameters defined

---

## 14. Things That Will Surprise You

- **Chroma stores embeddings on disk but loads them into memory at runtime.** A large vectorstore can use several hundred MB of RAM. Not a problem on 24GB, but worth knowing.
- **LangChain version compatibility is fragile.** The library moves fast and APIs change between minor versions. If you find a tutorial and it doesn't work, check the LangChain version it was written for.
- **Ollama's API is OpenAI-compatible.** This means you can swap Ollama for the OpenAI API with almost no code changes — useful if you ever want to test your pipeline against GPT-4o.
- **Your first ingest will probably fail.** A PDF will have encoding issues, a URL will block the scraper, or a chunk will be empty. This is normal. Build ingest as a resilient script with per-file error handling and logging from the start.
- **Retrieval quality varies dramatically with query phrasing.** "H3 for Insights" and "should the Insights header be an H3?" may return completely different chunks. This is a fundamental RAG challenge. Testing with varied phrasings early reveals weak spots in your chunking strategy.
- **The model will sometimes refuse to say "I don't know."** Even with strict mode and a strong system prompt, LLMs are trained to be helpful and will sometimes fill gaps. Regular testing with out-of-scope questions is how you catch this.
