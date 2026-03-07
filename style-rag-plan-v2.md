# Style RAG — Project Plan (v2)

A local-first RAG pipeline serving two overlapping audiences: **editorial teams** checking style and grammar rules before publishing, and **web content teams** looking up CMS workflows, component standards, and content governance rules. Built with LangChain, Qwen3.5 9B via Ollama, and Streamlit.

---

## The Problem This Solves

Two types of institutional knowledge are almost never written down in one place:

1. **Editorial style rules** — AP, APA, Chicago, internal house style. Teams waste time manually searching dense guides or asking colleagues for answers that exist in a document somewhere.
2. **Web content operations knowledge** — CMS how-tos, component usage rules, content governance decisions, HTML standards. This knowledge lives in Slack threads, someone's brain, or a SharePoint doc nobody can find.

This tool makes both instantly queryable — either by asking a direct question or pasting content for review.

---

## Two Audiences, Two Query Types

### Editorial Teams
- Style and grammar questions
- Content review and correction
- Format generation (press releases, e-vites, web articles)
- Cross-guide conflict resolution

### Web Content Teams
- CMS workflow questions ("how do I add an alternating list in Sitecore Experience Editor?")
- Component usage rules ("do we use H3 for Insights? Only bold for blogs?")
- Content governance ("do we leave resource page tiles up during a tile change?")
- Page structure standards ("how many carousels is too many on a bio page?")
- SEO, accessibility, and URL conventions

---

## Feature Set

### MVP (Core)
- Ingest PDFs (house style guides, how-tos, process docs, CMS documentation) and scraped web content (AP, WCAG, etc.)
- Ask a question, get a grounded answer with citations
- Paste raw content, get style issues flagged with corrections
- Source attribution on every response (which guide, which section)
- Streamlit chat UI

### Layer 2 (Makes it genuinely useful)
- Multi-document awareness — knows which guide to pull from based on context
- House style takes precedence over external guides when there's a conflict
- Output format selector — user specifies "press release", "web article", "e-vite", "blog" for context-aware answers
- Corrected version output — don't just flag issues, rewrite the corrected copy
- Query type awareness — routes style questions vs process/CMS questions to the most relevant document sources

### Layer 3 (Portfolio differentiators)
- Document management UI — upload new PDFs directly in the app without touching the terminal
- Conversation memory — follow-up questions work ("what about for email subject lines?")
- Strict mode toggle — answers only from indexed documents, no LLM general knowledge filling gaps

### Out of Scope (note but don't build yet)
- User accounts / team access
- Edit history / version tracking
- Live CMS integration (e.g. querying Sitecore directly)
- Slack bot interface

---

## Document Categories

### Editorial
| Document | Type | Source |
|---|---|---|
| AP Stylebook | Style guide | Scraped web |
| APA 7th Edition | Style guide | Scraped web |
| Chicago Manual of Style excerpts | Style guide | Scraped web |
| Acme Corp House Style Guide | Style guide | PDF (internal) |
| Plain Language Guidelines | Style guide | plainlanguage.gov (scraped) |
| Press release / e-vite templates | Template | PDF (internal) |

### Web Content & CMS
| Document | Type | Source |
|---|---|---|
| HTML component standards | Standards doc | PDF (internal) |
| Sitecore Experience Editor how-tos | Process doc | PDF (internal) |
| Component usage rules | Standards doc | PDF (internal) |
| Content governance guidelines | Process doc | PDF (internal) |
| Page structure standards | Standards doc | PDF (internal) |
| SEO guidelines | Standards doc | PDF (internal) |
| WCAG accessibility guidelines | Standards doc | w3.org (scraped) |
| URL slug and taxonomy conventions | Standards doc | PDF (internal) |
| Internal naming conventions | Standards doc | PDF (internal) |

---

## Tech Stack

| Role | Tool | Why |
|---|---|---|
| LLM | Qwen3.5 9B via Ollama | Local, fast on M4 Pro |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap, reliable, no setup |
| Vector store | Chroma | Persists to disk, simple API |
| PDF ingestion | LangChain `PyPDFLoader` | Built in, handles multi-page |
| Web scraping | LangChain `WebBaseLoader` | Built in, handles AP-style pages |
| Orchestration | LangChain | Ties everything together |
| UI | Streamlit | Fast to build, looks decent |
| Language | Python 3.11+ | Standard |

### M4 Pro GPU Memory Optimization

Before running the app for the first time, run this command in your terminal to grant macOS's GPU access to the majority of your 24GB unified RAM:

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

This sets the wired memory limit to 20GB, ensuring Ollama and Qwen3.5 can fully leverage the M4 Pro's GPU rather than being constrained by macOS's default conservative memory allocation. Without this, the model may fall back to slower CPU inference under memory pressure. Add this to your README as a required setup step.

---

## Project File Structure

```
style-rag/
├── app.py                  # Streamlit UI — the front door
├── ingest.py               # One-time indexing script
├── rag.py                  # Retrieval + generation logic
├── config.py               # Settings, source priorities, model names
├── sources.py              # List of URLs to scrape + PDF paths
│
├── docs/
│   ├── editorial/          # Style guides, templates
│   │   ├── house-style.pdf
│   │   └── press-release-template.pdf
│   └── web-content/        # CMS how-tos, component standards
│       ├── sitecore-how-tos.pdf
│       ├── component-standards.pdf
│       └── content-governance.pdf
│
├── vectorstore/            # Chroma auto-creates this
│
├── requirements.txt
└── .env                    # OpenAI API key
```

---

## Data Flow

### Ingest Flow
Runs once, or whenever new documents are added.

```
PDF / URL
   → Load raw text
   → Split into chunks (500–800 tokens with overlap)
   → Tag each chunk with metadata (source, priority, doc type, audience)
   → Embed each chunk via OpenAI text-embedding-3-small
   → Store embeddings + metadata in Chroma
```

### Query Flow
Runs on every user input.

```
User input
   → Embed the question via OpenAI
   → Chroma finds most similar chunks (top 4–6)
   → Chunks + question passed to Qwen3.5 as context
   → Qwen3.5 generates answer grounded in those chunks
   → Response + source citations returned to UI
```

> **Key insight:** OpenAI only handles embedding in both flows. Qwen3.5 only handles generation in the query flow. They never interact — LangChain orchestrates them independently.

---

## Chunking Strategy

| Document Type | Chunk Size | Overlap | Reason |
|---|---|---|---|
| Style guides | 500 tokens | 50 tokens | Rules are short and self-contained |
| Process docs / how-tos | 800 tokens | 100 tokens | Steps are sequential and need surrounding context |
| Component standards | 500 tokens | 75 tokens | Rule-based but often have multiple related sub-rules |
| Templates | 600 tokens | 50 tokens | Structured but benefit from seeing surrounding context |

Document type is set at ingest time via metadata tagging, and the appropriate chunking config is applied per type.

---

## Metadata Strategy

Every chunk stored in Chroma gets metadata tags. This enables precedence logic, source citations, and audience-aware routing.

```python
{
  "source": "Acme Corp House Style Guide",
  "priority": 1,             # 1 = highest; house style always wins
  "doc_type": "style",       # "style", "process", "template", "standards"
  "audience": "editorial",   # "editorial", "web-content", "both"
  "format_context": ["press_release", "web", "email", "blog"]
}
```

**Priority levels:**
- `1` — Internal house style / company standards (always wins)
- `2` — Internal process docs and how-tos
- `3` — External style guides (AP, APA, Chicago)
- `4` — External standards (WCAG, plainlanguage.gov)

### Pre-Filtering Query Logic (Self-Querying)

Implement a self-querying layer in `rag.py` that uses the LLM to analyze the user's intent and apply metadata filters *before* performing the vector similarity search. This prevents cross-contamination between editorial rules and CMS workflows.

| Approach | How It Works | Impact on M4 Pro |
|---|---|---|
| Standard RAG | Searches all 10+ guides for every query | Slower; context window fills with irrelevant rules |
| Self-Querying RAG | LLM filters by `audience` first, then searches only relevant docs | Faster; keeps context window focused and precise |

If a user asks a CMS question, Chroma only searches chunks where `audience == "web-content"`. Editorial chunks are never loaded into the context window. Build this in Layer 2 once basic retrieval is stable — it is a meaningful quality and performance upgrade.

### Post-Processing Safety Net

Explicitly document and implement a `strip_thinking()` utility function in `rag.py`. Even with thinking mode disabled in config, Qwen3.5 may occasionally emit `<think>` tags under certain prompt conditions. This function ensures they are always stripped before the response reaches the UI:

```python
import re

def strip_thinking(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
```

Call this on every response before returning it to `app.py`. It is a one-line safeguard that prevents a confusing user experience.

---

## Input Mode Implementation

Both modes use the same RAG pipeline — only the prompt wrapper differs.

**Question mode**
User types a question → standard RAG query → grounded answer with citation.

**Review mode**
User pastes a block of text → system prompt prepends:
> *"Review the following content for style and web standards issues based only on the indexed guides. List each issue, explain the rule, and provide a corrected version where applicable."*

Handled in the UI with a simple radio button toggle. No backend changes.

---

## Example Interactions

### Editorial — Question mode

**Q:** Do we capitalize "Director of Communications" in a press release?

**A:** Capitalize formal titles when they appear directly before a name (e.g., "Director of Communications Jane Smith") but lowercase when used generically or after a name (e.g., "Jane Smith, director of communications").
*Source: AP Stylebook, Titles — Acme Corp House Style Guide, p. 4*

---

**Q:** What's the correct date format for a web article?

**A:** Use "Jan. 5, 2025" for web articles. AP style abbreviates months with six or more letters when paired with a specific date. Never use ordinal suffixes (1st, 2nd, 3rd).
*Source: AP Stylebook, Dates*

---

### Editorial — Review mode

**Input:**
> "On September 3rd, 2025, Vice President of Marketing, John Davis, announced the launch of our new Website."

**Output:**
- **"September 3rd"** → "Sept. 3" — AP abbreviates months; no ordinal suffix
- **"Vice President of Marketing, John Davis"** → remove comma; AP style
- **"Website"** → "website" — lowercase per AP 2019 update

**Corrected:** "On Sept. 3, 2025, Vice President of Marketing John Davis announced the launch of our new website."
*Sources: AP Stylebook / Acme Corp House Style Guide*

---

### Web Content — CMS question

**Q:** How do I add an alternating list inside Sitecore Experience Editor?

**A:** In Experience Editor, navigate to the component panel and select the Alternating List component. Insert it into the designated placeholder. Each row accepts a headline, body copy, and optional image. Alternate the image position (left/right) using the "Flip Layout" toggle in the component properties panel. Save and publish when complete.
*Source: Sitecore How-To Guide, p. 8*

---

### Web Content — Component standards

**Q:** Do we use H3 for the Insights section header, or just bold text?

**A:** Use H3 for Insights section headers on article and landing pages. Bold text is reserved for blog post subheadings only. Maintaining heading hierarchy (H1 → H2 → H3) is required for accessibility compliance.
*Source: HTML Component Standards Doc — WCAG 2.1 Section 1.3.1*

---

### Web Content — Governance question

**Q:** When we're updating tiles, do we leave the existing resource page tiles live?

**A:** Yes — existing resource page tiles remain live during a tile update until the replacement is published and QA'd. Never unpublish a tile before its replacement is ready. If a tile must be removed without a replacement, escalate to the content lead before taking action.
*Source: Content Governance Guidelines, p. 3*

---

### Web Content — Page structure

**Q:** How many carousels is too many on a bio page?

**A:** Bio pages should have no more than one carousel. Bio pages follow a defined component hierarchy: hero, summary, expertise list, related content. A second carousel creates visual competition and degrades mobile performance. Use a static content grid instead if additional content needs to be surfaced.
*Source: Component Standards Doc, Bio Page Guidelines*

---

### Cross-guide conflict resolution

**Q:** Our house guide says to use Oxford commas but AP says don't — which do we follow?

**A:** Your internal house style guide takes precedence. Acme Corp style specifies the Oxford comma for all web and print content. AP's omission applies to wire copy and does not override your house rules.
*Source: Acme Corp House Style Guide, p. 2 — "In cases of conflict, defer to this guide over AP."*

---

## What You'll Learn Building This

- **Chunking tradeoffs** — why chunk size matters and how to tune it per document type
- **Embedding model selection** — why OpenAI over a local model for this use case
- **Vector similarity search** — how Chroma finds relevant chunks
- **Prompt engineering** — structuring system prompts to keep answers grounded
- **Metadata filtering** — how to prioritize sources and route by audience at retrieval time
- **LangChain abstractions** — chains, retrievers, document loaders
- **Local LLM tradeoffs** — latency, privacy, cost vs API-based inference
- **Multi-corpus RAG** — managing multiple document types with different structures and priorities

---

## Portfolio Narrative

> "Built a local-first RAG pipeline serving editorial and web content teams, indexing style guides, CMS how-tos, and content governance documentation. Uses LangChain for orchestration, Qwen3.5 9B running locally via Ollama on Apple Silicon for generation, OpenAI embeddings, and Chroma for vector storage. Features two input modes (question and content review), source-attributed responses, and a metadata-driven precedence system that ensures internal house style always overrides external guides. Built with a Streamlit UI."

---

## Build Order

1. Install Ollama + pull Qwen3.5 9B — verify local inference works
2. Set up Python environment — install all dependencies
3. Build `ingest.py` — PDF + web ingestion pipeline with metadata tagging
4. Build `rag.py` — retrieval + generation chain
5. Build `app.py` — Streamlit UI with question and review modes
6. Add metadata + precedence logic
7. Add document upload UI (Layer 2)
8. Add conversation memory (Layer 2)
9. Add strict mode toggle (Layer 3)
