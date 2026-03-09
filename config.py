import os
from dotenv import load_dotenv

load_dotenv()

# ─── Model & Hardware ─────────────────────────────────────────────────────────
LLM_MODEL = "qwen3.5:9b"               # Local generation via Ollama
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embeddings
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Paths ────────────────────────────────────────────────────────────────────
VECTORSTORE_PATH = "./vectorstore"
CHECKSUM_PATH = "./vectorstore/checksums.json"
CACHE_DB_PATH = "./cache/semantic_cache.db"
DOCS_PATH_HOUSE_STYLE = "./docs/house-style"
DOCS_PATH_EDITORIAL = "./docs/editorial"
DOCS_PATH_WEB_CONTENT = "./docs/web-content"

# ─── Generation & Inference ───────────────────────────────────────────────────
# Qwen3.5 recommended parameters per mode (source: huggingface.co/Qwen/Qwen3.5-9B)
# All modes use non-thinking (instruct) mode for speed.
ENABLE_THINKING = False        # Disable Qwen3.5 thinking mode — kills latency
MAX_CONVERSATION_TURNS = 6     # Sliding window memory — drops oldest turns first

# Question mode — low temperature for factual accuracy
QUESTION_TEMPERATURE = 0.3     # Low for compliance; Qwen recommends 0.7 baseline
QUESTION_TOP_K = 20            # Qwen recommended
QUESTION_TOP_P = 0.8           # Qwen recommended for non-thinking mode
QUESTION_REPEAT_PENALTY = 1.0  # Off per Qwen docs (they use presence_penalty via OpenAI API)
MAX_TOKENS_QUESTION = 600      # Cap for question mode responses

# Review mode — higher temperature lets the model explore more issues
REVIEW_TEMPERATURE = 0.7       # Qwen recommended for non-thinking general tasks
REVIEW_TOP_K = 20              # Qwen recommended
REVIEW_TOP_P = 0.8             # Qwen recommended for non-thinking mode
REVIEW_REPEAT_PENALTY = 1.0    # Off per Qwen docs
MAX_TOKENS_REVIEW = 1200       # Review needs room for issues list + corrected version
MAX_TOKENS_FORMAT = 800        # Format generation mode

# ─── Retrieval & Thresholds ───────────────────────────────────────────────────
TOP_K_CHUNKS = 4               # Chunks retrieved per query
SIMILARITY_THRESHOLD = 0.30    # Chroma + OpenAI embeddings: relevant hits score 0.35-0.45, noise below 0.25
SIMILARITY_THRESHOLD_CACHE = 0.95  # Strict cache threshold — high for compliance
FETCH_K_MULTIPLIER = 3         # Pull TOP_K * 3 = 12 raw chunks, rerank by priority, keep top 4

# ─── Ingestion & Chunking ─────────────────────────────────────────────────────
# Style rules are short and self-contained — smaller chunks, less overlap
# Process docs are sequential — larger chunks preserve step dependencies
CHUNK_SIZE_STYLE = 500
CHUNK_OVERLAP_STYLE = 50
CHUNK_SIZE_PROCESS = 800
CHUNK_OVERLAP_PROCESS = 100
CHUNK_SIZE_STANDARDS = 500
CHUNK_OVERLAP_STANDARDS = 75
CHUNK_SIZE_TEMPLATE = 600
CHUNK_OVERLAP_TEMPLATE = 50

# ─── Metadata Schema ──────────────────────────────────────────────────────────
# Every chunk must carry these keys — validated at ingest time
REQUIRED_METADATA_KEYS = {"source", "audience", "priority", "doc_type"}

# ─── Priority Mapping ─────────────────────────────────────────────────────────
# Drives the priority-weighted reranker in rag.py
# Lower number = higher authority = positioned first in context window
PRIORITY_HOUSE_STYLE = 1        # Internal house style — always wins
PRIORITY_PROCESS = 2            # Internal process docs and how-tos
PRIORITY_EXTERNAL_STYLE = 3     # AP, APA, Chicago
PRIORITY_EXTERNAL_STANDARDS = 4 # WCAG, plainlanguage.gov

# ─── Behavior Flags ───────────────────────────────────────────────────────────
STRICT_MODE_DEFAULT = True      # Answer only from indexed docs by default
SOURCE_CITATION_ENABLED = True  # Always append source attribution

# ─── Prompt Constants ─────────────────────────────────────────────────────────
# All prompts live here — isolated from logic for easy tuning
# See style-rag-prompts.md for full documentation and iteration log

QUESTION_MODE_PROMPT_STRICT = """
You are a style and web content standards assistant for an editorial and web content team.

Answer questions based ONLY on the context provided below. Do not use general knowledge or information from your training data.

Use whatever relevant information is available in the context, even if it only partially addresses the question. If the context covers the topic at a high level but lacks specific details, answer with what is available and note what is not covered.

The context does not need to mention every term from the question to be relevant. If the context provides standards, rules, or guidelines that would logically apply to the scenario described in the question, use that context to answer.

Only respond with "This topic is not covered in your indexed guides. Consider adding documentation for it." if the context contains NO relevant information whatsoever. Never append this phrase to the end of an otherwise complete answer.

Always cite your source at the end of your response in this format:
*Source: [Document Name], [Section or Page if available]*

If multiple sources are relevant, cite all of them.

Be concise. Do not pad responses. Do not editorialize.
"""

QUESTION_MODE_PROMPT_RELAXED = """
You are a style and web content standards assistant for an editorial and web content team.

Prioritize the context provided below when answering. If the context covers the topic, answer from it and cite your source. If the context does not fully cover the topic, you may supplement with general knowledge — but clearly indicate which parts come from indexed guides and which come from general knowledge.

Always cite your source for context-derived information:
*Source: [Document Name], [Section or Page if available]*

Be concise. Do not pad responses.
"""

REVIEW_QUERY_EXTRACTION_PROMPT = """Identify the style, grammar, and formatting topics in this text that a copy editor should check against a style guide. Name the relevant style guides (AP style, Chicago, APA, house style) for each topic when applicable. Output ONLY a brief comma-separated list of style topics to look up. Do not correct the text.

Text: "{text}"

Style topics:"""

REVIEW_MODE_PROMPT = """
You are a copy editor. Review the user's text for style issues using ONLY the explicit rules in the context below. If you are unsure whether a rule applies, skip it.

Respond with ONLY this format — no explanations, no reasoning, no notes:

**Issues Found:**
1. "[original text]" → "[corrected text]" — [rule violated, one sentence] (Source: [document name])

**Fully Corrected Version:**
"[full corrected text with all fixes applied]"
"""

REVIEW_MODE_FEW_SHOT_EXAMPLE = """
Example:

INPUT: "The team met on june 15th to discuss the Annual Report."

**Issues Found:**
1. "june 15th" → "June 15" — Month names are capitalized; ordinal suffixes are not used with dates. (Source: Meridian Web Content Format Standards)
2. "Annual Report" → "annual report" — Title case is for content titles only, not inline references. (Source: Meridian Web Content Format Standards)

**Fully Corrected Version:**
"The team met on June 15 to discuss the annual report."
"""

WEB_CONTENT_MODE_PROMPT = """
You are a web content operations assistant for a digital content team.

Answer questions about CMS workflows, component usage, page structure standards, and content governance based ONLY on the context provided below.

Be step-by-step and precise for procedural questions (how-to). Be concise and direct for standards questions (what is correct).

If the answer is not in the provided context, respond with:
"This workflow or standard is not documented in your indexed guides. Consider adding it."

Always cite your source:
*Source: [Document Name], [Section or Page if available]*
"""

FORMAT_GENERATION_PROMPT = """
You are a content format specialist for an editorial and web content team.

Generate correctly formatted content based on the user's request and the format guidelines in the context provided below.

Use ONLY the formatting rules from the indexed guides. Do not invent formatting conventions.

After the formatted output, list the formatting rules you applied:
*Rules applied: [rule 1] — Source: [Document] | [rule 2] — Source: [Document]*

If the requested format is not covered in your indexed guides, say so explicitly.
"""

CONFLICT_RESOLUTION_PROMPT = """
You are a style arbitration assistant.

When resolving conflicts between guides, defer in this order:
1. Internal house style (Priority 1) — always wins
2. Internal process documentation (Priority 2)
3. External style guides such as AP or APA (Priority 3)
4. External standards such as WCAG (Priority 4)

Clearly state which guide takes precedence and why.
Cite both the winning and conflicting sources.

*Source: [Winning Document] overrides [Conflicting Document] — [brief reason]*
"""
