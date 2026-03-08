import re
import httpx
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTORSTORE_PATH,
    TEMPERATURE,
    MAX_TOKENS_REVIEW,
    ENABLE_THINKING,
    REPEAT_PENALTY,
    TOP_K_CHUNKS,
    SIMILARITY_THRESHOLD,
    FETCH_K_MULTIPLIER,
    STRICT_MODE_DEFAULT,
    QUESTION_MODE_PROMPT_STRICT,
    QUESTION_MODE_PROMPT_RELAXED,
    REVIEW_MODE_PROMPT,
    REVIEW_MODE_FEW_SHOT_EXAMPLE,
    REVIEW_QUERY_EXTRACTION_PROMPT,
    WEB_CONTENT_MODE_PROMPT,
    FORMAT_GENERATION_PROMPT,
    CONFLICT_RESOLUTION_PROMPT,
)


# ─── Ollama Health Check ─────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ─── Post-Processing ─────────────────────────────────────────────────────────

def strip_thinking(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


# ─── Initialization Helpers ──────────────────────────────────────────────────
# Designed to be wrapped with @st.cache_resource in app.py

def init_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def init_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )


def init_llm() -> OllamaLLM:
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        repeat_penalty=REPEAT_PENALTY,
        num_predict=MAX_TOKENS_REVIEW,  # Highest cap; model stops naturally for shorter answers
        reasoning=ENABLE_THINKING,  # False disables Qwen3.5 thinking mode — critical for speed
    )


# ─── Review Mode Query Reformulation ─────────────────────────────────────────
# Raw prose scores ~0.05 against style guide chunks — semantically unrelated.
# This extracts the style topics the prose touches so retrieval finds the
# right rules. See EVAL_REPORT.md Issue #2 for the full analysis.

def build_review_query(user_input: str, llm: OllamaLLM) -> str:
    """
    Extract style-relevant topics from review prose to produce a search
    query that matches style guide chunks. Bridges the semantic gap between
    'Vice President John Davis' and 'AP style capitalizes titles before names'.
    """
    prompt = REVIEW_QUERY_EXTRACTION_PROMPT.format(text=user_input[:1000])
    raw = llm.invoke(prompt)
    query = strip_thinking(raw).strip()
    return query if query else user_input


# ─── Retrieval ───────────────────────────────────────────────────────────────

def retrieve(query: str, vectorstore: Chroma, skip_threshold: bool = False) -> list:
    """
    Fetch TOP_K * FETCH_K_MULTIPLIER candidate chunks with relevance scores,
    discard any below SIMILARITY_THRESHOLD, then keep the top TOP_K_CHUNKS.
    Over-fetching before filtering gives the priority reranker better
    candidates to work with.

    skip_threshold: When True (review mode), skip similarity filtering and
    return the top K chunks regardless of score. Review inputs are prose,
    not questions, so they score low against style guide chunks.
    """
    fetch_k = TOP_K_CHUNKS * FETCH_K_MULTIPLIER
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query, k=fetch_k
    )
    if skip_threshold:
        return [doc for doc, score in docs_with_scores[:TOP_K_CHUNKS]]
    # Filter by similarity threshold — prevents hallucination from irrelevant chunks
    valid = [
        (doc, score) for doc, score in docs_with_scores
        if score >= SIMILARITY_THRESHOLD
    ]
    # Keep only the top K after filtering
    return [doc for doc, score in valid[:TOP_K_CHUNKS]]


def retrieve_with_scores(query: str, vectorstore: Chroma, skip_threshold: bool = False) -> list:
    """
    Same as retrieve() but returns (doc, score) tuples.
    Used by the eval suite to analyze score distributions and filtering behavior.
    """
    fetch_k = TOP_K_CHUNKS * FETCH_K_MULTIPLIER
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query, k=fetch_k
    )
    if skip_threshold:
        return docs_with_scores[:TOP_K_CHUNKS]
    valid = [
        (doc, score) for doc, score in docs_with_scores
        if score >= SIMILARITY_THRESHOLD
    ]
    return valid[:TOP_K_CHUNKS]


def rerank_by_priority(docs: list) -> list:
    """
    Sort retrieved chunks by metadata priority (ascending).
    Priority 1 = highest authority (house style) — appears first in context.
    Chunks without a priority tag sort to the end.
    Leverages LLM primacy bias: earlier context = stronger influence.
    """
    return sorted(docs, key=lambda d: d.metadata.get("priority", 99))


# ─── Prompt Assembly ─────────────────────────────────────────────────────────

def get_system_prompt(mode: str, strict: bool = STRICT_MODE_DEFAULT) -> str:
    if mode == "review":
        return REVIEW_MODE_PROMPT + "\n" + REVIEW_MODE_FEW_SHOT_EXAMPLE
    elif mode == "web-content":
        return WEB_CONTENT_MODE_PROMPT
    elif mode == "format":
        return FORMAT_GENERATION_PROMPT
    elif mode == "conflict":
        return CONFLICT_RESOLUTION_PROMPT
    else:
        return QUESTION_MODE_PROMPT_STRICT if strict else QUESTION_MODE_PROMPT_RELAXED


def format_context(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ─── Programmatic Citations ──────────────────────────────────────────────────
# Never ask the LLM to generate citations — it will hallucinate them.
# Extract source metadata from retrieved chunks and append directly.

def build_citations(docs: list) -> list[dict]:
    """
    Build deduplicated citation list from chunk metadata.
    Returns list of dicts for flexible rendering in the UI.
    """
    seen = set()
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        label = f"{source}, p. {page}" if page else source
        if label not in seen:
            seen.add(label)
            citations.append({"source": source, "page": page, "label": label})
    return citations


def format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = [f"*Source: {c['label']}*" for c in citations]
    return "\n".join(lines)


# ─── Core Query Function ─────────────────────────────────────────────────────

PROMPT_TEMPLATE = PromptTemplate.from_template(
    "{system_prompt}\n\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "--- END CONTEXT ---\n\n"
    "User: {question}"
)


def query(
    user_input: str,
    vectorstore: Chroma,
    llm: OllamaLLM,
    mode: str = "question",
    strict: bool = STRICT_MODE_DEFAULT,
) -> dict:
    """
    Main RAG pipeline entry point.

    Returns dict with:
      - answer: str (cleaned LLM response)
      - citations: list[dict] (source metadata from retrieved chunks)
      - citations_formatted: str (ready-to-display citation block)
      - num_chunks: int (how many chunks were used)
    """
    # 1. Retrieve and rerank
    #    Review mode: reformulate prose into style topics for better retrieval
    if mode == "review":
        retrieval_query = build_review_query(user_input, llm)
    else:
        retrieval_query = user_input
    docs = retrieve(retrieval_query, vectorstore, skip_threshold=(mode == "review"))
    ranked_docs = rerank_by_priority(docs)

    # 2. Handle no relevant chunks found
    if not ranked_docs:
        return {
            "answer": "This topic is not covered in your indexed guides. "
                      "Consider adding documentation for it.",
            "citations": [],
            "citations_formatted": "",
            "num_chunks": 0,
        }

    # 3. Assemble prompt
    system_prompt = get_system_prompt(mode, strict)
    context = format_context(ranked_docs)

    # 4. Generate — num_predict set at init to max (review mode cap).
    #    Model stops naturally for shorter answers; no .bind() needed.
    #    (.bind(options=...) overrides model_kwargs and loses think=False)
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    raw_response = chain.invoke({
        "system_prompt": system_prompt,
        "context": context,
        "question": user_input,
    })

    # 5. Clean response
    answer = strip_thinking(raw_response)

    # 6. Build programmatic citations
    citations = build_citations(ranked_docs)
    citations_formatted = format_citations(citations)

    return {
        "answer": answer,
        "citations": citations,
        "citations_formatted": citations_formatted,
        "num_chunks": len(ranked_docs),
    }
