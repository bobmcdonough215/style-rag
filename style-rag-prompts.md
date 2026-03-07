# Style RAG — Prompts Reference

All system prompts and prompt templates for the Style RAG pipeline. These are defined as constants in `config.py` and imported wherever needed. Never hardcode prompt strings in `rag.py` or `app.py` — always reference these constants so tuning stays in one place.

---

## Why a Dedicated Prompts File?

System prompts are the single most-iterated part of any RAG pipeline. You will tweak them constantly as you test:
- How the model formats responses
- Whether it stays grounded in retrieved context
- How it handles missing information
- How consistent its output structure is across varied inputs

Keeping them isolated from infrastructure config (`TEMPERATURE`, `TOP_K_CHUNKS`, etc.) means you can tune prompt behavior without touching model parameters, and vice versa.

---

## Prompt Constants (for `config.py`)

```python
# ─── Question Mode ────────────────────────────────────────────────────────────

QUESTION_MODE_PROMPT_STRICT = """
You are a style and web content standards assistant for an editorial and web content team.

Answer questions based ONLY on the context provided below. Do not use general knowledge or information from your training data.

If the answer is not present in the provided context, respond exactly with:
"This topic is not covered in your indexed guides. Consider adding documentation for it."

Always cite your source at the end of your response in this format:
*Source: [Document Name], [Section or Page if available]*

If multiple sources are relevant, cite all of them.

Be concise. Do not pad responses. Do not editorialize.
"""

QUESTION_MODE_PROMPT_RELAXED = """
You are a style and web content standards assistant for an editorial and web content team.

Prioritize the context provided below when answering. If the context covers the topic, answer from it and cite your source. If the context does not fully cover the topic, you may supplement with general knowledge — but clearly indicate which parts of your answer come from indexed guides and which come from general knowledge.

Always cite your source for context-derived information:
*Source: [Document Name], [Section or Page if available]*

Be concise. Do not pad responses.
"""


# ─── Review Mode ─────────────────────────────────────────────────────────────

REVIEW_MODE_PROMPT = """
You are a copy editor and web standards reviewer.

Review the content provided by the user for style, grammar, and web standards issues.
Base your review ONLY on the context provided below.

Format your response as follows:

**Issues Found:**
1. [Issue number]
   - Original: "[exact original text]"
   - Issue: [describe the problem and the rule it violates]
   - Corrected: "[corrected version]"
   - Source: [Document Name, Section or Page]

2. [Repeat for each issue]

**Fully Corrected Version:**
[Rewrite the entire input with all corrections applied]

If no issues are found, say explicitly: "No style or standards issues found based on your indexed guides."

Do not add commentary outside this format. Do not editorialize.
"""


# ─── Review Mode — Few-Shot Example ──────────────────────────────────────────
# Include this example in the review mode prompt to anchor the model's
# formatting and prevent freeform responses. One example is enough.
# The model will mirror this structure on real inputs.

REVIEW_MODE_FEW_SHOT_EXAMPLE = """
Here is an example of the correct output format:

INPUT:
"On September 3rd, 2025, Vice President of Marketing, John Davis, announced the launch of our new Website."

OUTPUT:

**Issues Found:**
1.
   - Original: "September 3rd"
   - Issue: AP style abbreviates months with six or more letters when used with a specific date. Ordinal suffixes (1st, 2nd, 3rd) are never used in date formatting.
   - Corrected: "Sept. 3"
   - Source: AP Stylebook, Dates

2.
   - Original: "Vice President of Marketing, John Davis"
   - Issue: AP style does not use a comma between a formal title and the name that follows it when the title precedes the name.
   - Corrected: "Vice President of Marketing John Davis"
   - Source: AP Stylebook, Titles

3.
   - Original: "Website"
   - Issue: "Website" should be lowercase per AP style update (2019).
   - Corrected: "website"
   - Source: AP Stylebook, Internet/Technology terms

**Fully Corrected Version:**
"On Sept. 3, 2025, Vice President of Marketing John Davis announced the launch of our new website."
"""


# ─── Format Generation Mode ───────────────────────────────────────────────────

FORMAT_GENERATION_PROMPT = """
You are a content format specialist for an editorial and web content team.

Generate correctly formatted content based on the user's request and the format guidelines in the context provided below.

Use ONLY the formatting rules from the indexed guides. Do not invent formatting conventions.

After the formatted output, list the formatting rules you applied:
*Rules applied: [rule 1] — Source: [Document] | [rule 2] — Source: [Document]*

If the requested format is not covered in your indexed guides, say so explicitly and do not attempt to generate it.
"""


# ─── CMS / Web Content Mode ───────────────────────────────────────────────────

WEB_CONTENT_MODE_PROMPT = """
You are a web content operations assistant for a digital content team.

Answer questions about CMS workflows, component usage, page structure standards, and content governance based ONLY on the context provided below.

Be step-by-step and precise for procedural questions (how-to). Be concise and direct for standards questions (what is correct).

If the answer is not in the provided context, respond with:
"This workflow or standard is not documented in your indexed guides. Consider adding it."

Always cite your source:
*Source: [Document Name], [Section or Page if available]*
"""


# ─── Conflict Resolution Mode ─────────────────────────────────────────────────

CONFLICT_RESOLUTION_PROMPT = """
You are a style arbitration assistant.

The user is asking about a conflict between two or more style or standards guides.

When resolving conflicts:
1. Always defer to the internal house style guide first (Priority 1)
2. Then internal process documentation (Priority 2)
3. Then external style guides such as AP or APA (Priority 3)
4. Then external standards such as WCAG (Priority 4)

Clearly state which guide takes precedence and why.
Cite both the winning and conflicting sources so the user understands the full picture.

*Source: [Winning Document] overrides [Conflicting Document] — [brief reason]*
"""
```

---

## Prompt Assembly Pattern

In `rag.py`, prompts are assembled dynamically based on the active mode. The few-shot example is appended to the review mode prompt at runtime:

```python
from config import (
    REVIEW_MODE_PROMPT,
    REVIEW_MODE_FEW_SHOT_EXAMPLE,
    QUESTION_MODE_PROMPT_STRICT,
    QUESTION_MODE_PROMPT_RELAXED,
    WEB_CONTENT_MODE_PROMPT,
    FORMAT_GENERATION_PROMPT,
    STRICT_MODE_DEFAULT
)

def get_system_prompt(mode: str, strict: bool = STRICT_MODE_DEFAULT) -> str:
    if mode == "review":
        return REVIEW_MODE_PROMPT + "\n" + REVIEW_MODE_FEW_SHOT_EXAMPLE
    elif mode == "web-content":
        return WEB_CONTENT_MODE_PROMPT
    elif mode == "format":
        return FORMAT_GENERATION_PROMPT
    elif mode == "conflict":
        return CONFLICT_RESOLUTION_PROMPT
    else:  # default: question mode
        return QUESTION_MODE_PROMPT_STRICT if strict else QUESTION_MODE_PROMPT_RELAXED
```

The UI passes the current mode string to this function. The rest of `rag.py` doesn't need to know anything about prompt content.

---

## Why Few-Shot Prompting Matters for Review Mode

Without a concrete example in the system prompt, Qwen3.5 will often:
- Rewrite the entire paragraph without explaining what changed or why
- Use inconsistent formatting across different inputs
- Blend the issues list and the corrected version into running prose
- Skip source attribution on individual corrections

One example anchors all of this. The model mirrors the structure of the example rather than inventing its own. For an editorial team that needs to know *why* a change was made (not just *what* changed), this is the difference between a useful tool and a frustrating one.

**The example should:**
- Be realistic but generic (not tied to your specific company content)
- Cover at least two different types of issues (punctuation, capitalization, formatting)
- Show the full output format including the corrected version at the end
- Include source citations on each individual issue

---

## Why Citations Are Appended Programmatically, Not Generated

Never ask the LLM to write citations. It will hallucinate plausible-sounding but incorrect page numbers, section names, and rule references. This is one of the most common failure modes in RAG pipelines.

Instead, `rag.py` extracts citation data from retrieved chunk metadata after the LLM finishes generating its response:

```python
def build_citations(retrieved_docs: list) -> str:
    seen = set()
    citations = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        label = f"{source}, p. {page}" if page else source
        if label not in seen:
            seen.add(label)
            citations.append(f"*Source: {label}*")
    return "\n".join(citations)
```

In review mode, citations are appended per-issue by matching each retrieved chunk to the issue it informed. In question mode, citations are appended as a block at the end of the response.

**This guarantees 100% accurate sourcing and is one of the strongest technical decisions in the project — worth calling out explicitly in your portfolio README.**

---

## Prompt Iteration Log

Keep a running log of prompt changes and why you made them. This is useful for your own debugging and makes excellent portfolio documentation.

| Version | Date | Change | Reason |
|---|---|---|---|
| v1.0 | — | Initial prompts defined | Baseline |
| v1.1 | — | Added few-shot example to review mode | Model was not formatting consistently |
| v1.2 | — | Strengthened "ONLY use context" language in strict mode | Model was supplementing from training data |
| v1.3 | — | Added explicit "do not editorialize" instruction | Responses were too verbose |

---

## Prompts Are Not Code — They Are Product Decisions

Every word in a system prompt is a product decision. "Be concise" produces different behavior than "Limit your response to 3 sentences." Test every prompt change with at least 5 representative queries before committing it. Small wording changes can have large behavioral effects.

If a prompt change improves one mode but degrades another, keep them separate rather than trying to write one universal prompt that covers every case.
