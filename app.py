import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
import rag
from config import STRICT_MODE_DEFAULT, VECTORSTORE_PATH, MAX_CONVERSATION_TURNS

st.set_page_config(page_title="Style RAG", layout="wide")

# ─── Startup Checks ─────────────────────────────────────────────────────────

if not rag.check_ollama():
    st.error(
        "Ollama is not running. Start it with `ollama serve` in your terminal, "
        "then refresh this page."
    )
    st.stop()

if not os.path.exists(VECTORSTORE_PATH):
    st.warning(
        "No vectorstore found. Run `python ingest.py` to index your documents first."
    )
    st.stop()


# ─── Cached Resources ───────────────────────────────────────────────────────

@st.cache_resource
def load_vectorstore():
    embeddings = rag.init_embeddings()
    return rag.init_vectorstore(embeddings)

@st.cache_resource
def load_llm():
    return rag.init_llm()

vectorstore = load_vectorstore()
llm = load_llm()


# ─── Session State ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Style RAG")
    st.caption("Editorial & Web Content Assistant")

    st.divider()

    mode = st.radio("Mode", ["Question", "Review"])
    strict = st.toggle(
        "Strict Mode",
        value=STRICT_MODE_DEFAULT,
        help="When on, answers come only from indexed documents. "
             "When off, the model may supplement with general knowledge.",
    )

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Diagnostics — shown after first query
    if "last_meta" in st.session_state:
        st.divider()
        st.caption("Last Query")
        meta = st.session_state.last_meta
        st.markdown(f"**Chunks used:** {meta['num_chunks']}")
        if meta.get("citations"):
            for c in meta["citations"]:
                st.markdown(f"- {c['label']}")


# ─── Chat History ────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── Input ───────────────────────────────────────────────────────────────────

mode_key = "review" if mode == "Review" else "question"

if mode == "Review":
    with st.form("review_form", clear_on_submit=True):
        user_input = st.text_area(
            "Paste content for style review:",
            height=200,
            placeholder="Paste the text you want reviewed for style and web standards issues...",
        )
        submitted = st.form_submit_button("Review Content", type="primary")
    if not submitted or not user_input:
        user_input = None
else:
    user_input = st.chat_input("Ask a style or web content question...")


# ─── Stream Filter ───────────────────────────────────────────────────────────
# Intercepts <think> tags before they reach st.write_stream().
# think=False in Ollama should prevent them, but this is the safety net.

def _filter_thinking(stream):
    buffer = ""
    in_think = False
    for chunk in stream:
        buffer += chunk
        if "<think>" in buffer:
            in_think = True
        if in_think:
            if "</think>" in buffer:
                buffer = buffer.split("</think>", 1)[-1]
                in_think = False
            else:
                continue
        if buffer and not in_think:
            yield buffer
            buffer = ""
    # Flush any remaining content after stream ends
    if buffer and not in_think:
        yield buffer


# ─── Process Query ───────────────────────────────────────────────────────────

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve and rerank
    docs = rag.retrieve(user_input, vectorstore)
    ranked_docs = rag.rerank_by_priority(docs)

    with st.chat_message("assistant"):
        if not ranked_docs:
            answer = (
                "This topic is not covered in your indexed guides. "
                "Consider adding documentation for it."
            )
            st.markdown(answer)
            citations = []
            citations_formatted = ""
        else:
            # Build chain and stream the response
            system_prompt = rag.get_system_prompt(mode_key, strict)
            context = rag.format_context(ranked_docs)
            chain = (
                rag.PROMPT_TEMPLATE
                | llm.bind(num_predict=rag.get_max_tokens(mode_key))
                | StrOutputParser()
            )

            try:
                stream = chain.stream({
                    "system_prompt": system_prompt,
                    "context": context,
                    "question": user_input,
                })
                raw_response = st.write_stream(
                    _filter_thinking(stream)
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                # st.stop() halts the entire app — fine for single-user local tool.
                # For multi-user deployment, replace with early return logic.
                st.stop()

            answer = rag.strip_thinking(raw_response)

            # Programmatic citations — appended below the streamed response
            citations = rag.build_citations(ranked_docs)
            citations_formatted = rag.format_citations(citations)
            if citations_formatted:
                st.divider()
                st.markdown(citations_formatted)

    # Store in history
    full_response = (
        f"{answer}\n\n{citations_formatted}" if citations_formatted else answer
    )
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Enforce sliding window — prevents unbounded context growth
    max_messages = MAX_CONVERSATION_TURNS * 2  # each turn = user + assistant
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]

    # Update sidebar diagnostics on next rerun
    st.session_state.last_meta = {
        "num_chunks": len(ranked_docs),
        "citations": citations,
    }
