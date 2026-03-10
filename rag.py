import logging
import re
import time
import httpx
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Raised when a pipeline stage (retrieval, generation) fails."""


from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTORSTORE_PATH,
    ENABLE_THINKING,
    QUESTION_TEMPERATURE,
    QUESTION_TOP_K,
    QUESTION_TOP_P,
    QUESTION_REPEAT_PENALTY,
    MAX_TOKENS_QUESTION,
    REVIEW_TEMPERATURE,
    REVIEW_TOP_K,
    REVIEW_TOP_P,
    REVIEW_REPEAT_PENALTY,
    MAX_TOKENS_REVIEW,
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
    """Question/general mode LLM — low temperature for factual accuracy."""
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=QUESTION_TEMPERATURE,
        top_k=QUESTION_TOP_K,
        top_p=QUESTION_TOP_P,
        repeat_penalty=QUESTION_REPEAT_PENALTY,
        num_predict=MAX_TOKENS_QUESTION,
        reasoning=ENABLE_THINKING,
    )


def init_review_llm() -> OllamaLLM:
    """Review mode LLM — higher temperature to explore more issues,
    presence_penalty to prevent deliberation loops."""
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=REVIEW_TEMPERATURE,
        top_k=REVIEW_TOP_K,
        top_p=REVIEW_TOP_P,
        repeat_penalty=REVIEW_REPEAT_PENALTY,
        num_predict=MAX_TOKENS_REVIEW,
        reasoning=ENABLE_THINKING,
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
    try:
        raw = llm.invoke(prompt)
    except Exception as e:
        logger.warning("Review query reformulation failed, using raw input: %s", e)
        return user_input
    query = strip_thinking(raw).strip()
    return query if query else user_input


# ─── Retrieval ───────────────────────────────────────────────────────────────

def retrieve(query: str, vectorstore: Chroma, skip_threshold: bool = False,
             with_scores: bool = False) -> list:
    """
    Fetch TOP_K * FETCH_K_MULTIPLIER candidate chunks with relevance scores,
    discard any below SIMILARITY_THRESHOLD, then keep the top TOP_K_CHUNKS.
    Over-fetching before filtering gives the priority reranker better
    candidates to work with.

    skip_threshold: When True (review mode), skip similarity filtering and
    return the top K chunks regardless of score. Review inputs are prose,
    not questions, so they score low against style guide chunks.

    with_scores: When True, return (doc, score) tuples instead of bare docs.
    Used by the eval suite to analyze score distributions.
    """
    fetch_k = TOP_K_CHUNKS * FETCH_K_MULTIPLIER
    try:
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k
        )
    except Exception as e:
        logger.error("Vectorstore retrieval failed: %s", e)
        raise RAGPipelineError(f"Retrieval failed: {e}") from e
    if skip_threshold:
        pairs = docs_with_scores[:TOP_K_CHUNKS]
    else:
        # Filter by similarity threshold — prevents hallucination from irrelevant chunks
        valid = [
            (doc, score) for doc, score in docs_with_scores
            if score >= SIMILARITY_THRESHOLD
        ]
        pairs = valid[:TOP_K_CHUNKS]
    if pairs:
        top_score = pairs[0][1]
        logger.info("Retrieval: %d chunks, top_score=%.4f, skip_threshold=%s",
                     len(pairs), top_score, skip_threshold)
    else:
        logger.info("Retrieval: 0 chunks (all below threshold %.2f)", SIMILARITY_THRESHOLD)
    if with_scores:
        return pairs
    return [doc for doc, score in pairs]


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
    review_llm: OllamaLLM = None,
    stream: bool = False,
) -> dict:
    """
    Main RAG pipeline entry point.

    Args:
      review_llm: Separate LLM instance tuned for review mode. Uses different
                  Qwen3.5 parameters (higher temp, presence_penalty) to prevent
                  deliberation loops. Falls back to llm if not provided.
      stream: When True, returns a stream iterator instead of invoking.
              Caller must join chunks and call finalize_response() afterward.

    Returns dict with:
      - answer: str (cleaned LLM response)  [non-streaming]
      - stream: iterator, docs: list        [streaming]
      - citations: list[dict] (source metadata from retrieved chunks)
      - citations_formatted: str (ready-to-display citation block)
      - num_chunks: int (how many chunks were used)
    """
    t_start = time.perf_counter()

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
        logger.info("Query [%s] refused — no chunks above threshold (%.2fs)",
                     mode, time.perf_counter() - t_start)
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

    # 4. Generate — select the right LLM for the mode
    active_llm = review_llm if (mode == "review" and review_llm) else llm
    chain = PROMPT_TEMPLATE | active_llm | StrOutputParser()
    prompt_vars = {
        "system_prompt": system_prompt,
        "context": context,
        "question": user_input,
    }

    if stream:
        # Streaming path — returns an iterator for the UI to consume.
        # Caller is responsible for joining chunks and passing the full
        # text back through finalize_response() afterward.
        logger.info("Query [%s] streaming — %d chunks (%.2fs to retrieval)",
                     mode, len(ranked_docs), time.perf_counter() - t_start)
        return {
            "stream": chain.stream(prompt_vars),
            "docs": ranked_docs,
        }

    try:
        raw_response = chain.invoke(prompt_vars)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise RAGPipelineError(f"Generation failed: {e}") from e

    elapsed = time.perf_counter() - t_start

    # 5. Clean response
    answer = strip_thinking(raw_response)

    # 6. Build programmatic citations
    citations = build_citations(ranked_docs)
    citations_formatted = format_citations(citations)

    logger.info("Query [%s] complete — %d chunks, %d chars, %.2fs",
                 mode, len(ranked_docs), len(answer), elapsed)

    return {
        "answer": answer,
        "citations": citations,
        "citations_formatted": citations_formatted,
        "num_chunks": len(ranked_docs),
    }


def finalize_response(raw_response: str, docs: list) -> dict:
    """
    Post-process a streamed response: strip thinking tags, build citations.
    Called by the UI after streaming completes.
    """
    answer = strip_thinking(raw_response)
    citations = build_citations(docs)
    citations_formatted = format_citations(citations)
    return {
        "answer": answer,
        "citations": citations,
        "citations_formatted": citations_formatted,
        "num_chunks": len(docs),
    }
