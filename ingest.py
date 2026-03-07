import os
import json
import hashlib
import argparse
import shutil
from bs4 import SoupStrainer
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import glob as globmod
from config import (
    VECTORSTORE_PATH,
    CHECKSUM_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE_STYLE, CHUNK_OVERLAP_STYLE,
    CHUNK_SIZE_PROCESS, CHUNK_OVERLAP_PROCESS,
    CHUNK_SIZE_STANDARDS, CHUNK_OVERLAP_STANDARDS,
    CHUNK_SIZE_TEMPLATE, CHUNK_OVERLAP_TEMPLATE,
    REQUIRED_METADATA_KEYS,
    DOCS_PATH_EDITORIAL, DOCS_PATH_WEB_CONTENT,
)

# ─── Source Configuration ─────────────────────────────────────────────────────
# Explicit sources for PDFs with unique metadata, plus web sources.
# Web-content PDFs are auto-discovered from DOCS_PATH_WEB_CONTENT below.

def _scan_directory(directory: str, metadata: dict) -> list:
    """
    Auto-discover all PDFs in a directory and generate SOURCES entries
    with shared metadata. The source name is derived from the filename.
    Drop a new PDF in the folder, re-run ingest — no config changes needed.
    """
    entries = []
    for filepath in sorted(globmod.glob(os.path.join(directory, "*.pdf"))):
        # Derive a human-readable source name from the filename
        name = os.path.splitext(os.path.basename(filepath))[0]
        name = name.replace("-", " ").replace("_", " ").replace("  ", " ").strip()
        entry_metadata = {**metadata, "source": name}
        entries.append({"path": filepath, "type": "pdf", "metadata": entry_metadata})
    return entries


SOURCES = [
    # ─── Editorial PDFs (explicit — each has a curated source name) ───────────
    {
        "path": f"{DOCS_PATH_EDITORIAL}/ap_style_guide-detailed.pdf",
        "type": "pdf",
        "metadata": {
            "source": "AP Stylebook Guide (Detailed)",
            "audience": "editorial",
            "priority": 3,
            "doc_type": "style"
        }
    },
    {
        "path": f"{DOCS_PATH_EDITORIAL}/AP-Style.pdf",
        "type": "pdf",
        "metadata": {
            "source": "AP Style Formatting & Guidelines",
            "audience": "editorial",
            "priority": 3,
            "doc_type": "style"
        }
    },
    {
        "path": f"{DOCS_PATH_EDITORIAL}/APA-referencing-guide.pdf",
        "type": "pdf",
        "metadata": {
            "source": "APA 7th Edition Reference Guide",
            "audience": "editorial",
            "priority": 3,
            "doc_type": "style"
        }
    },
    {
        "path": f"{DOCS_PATH_EDITORIAL}/chicago-manual-quick-guide.pdf",
        "type": "pdf",
        "metadata": {
            "source": "Chicago Manual of Style Quick Guide",
            "audience": "editorial",
            "priority": 3,
            "doc_type": "style"
        }
    },
    # ─── Web Sources ─────────────────────────────────────────────────────────
    {
        "path": "https://www.plainlanguage.gov/guidelines/",
        "type": "web",
        "metadata": {
            "source": "Plain Language Guidelines",
            "audience": "editorial",
            "priority": 3,
            "doc_type": "style"
        }
    },
    {
        "path": "https://www.w3.org/TR/WCAG22/",
        "type": "web",
        "metadata": {
            "source": "WCAG 2.2 Accessibility Guidelines",
            "audience": "web-content",
            "priority": 4,
            "doc_type": "standards"
        }
    },
    {
        "path": "https://www.umaryland.edu/cpa/website-manual/prepare/editorial/",
        "type": "web",
        "metadata": {
            "source": "UMD Editorial Style Guide",
            "audience": "editorial",
            "priority": 3,
            "doc_type": "style"
        }
    },
# ─── Web Content PDFs (auto-discovered from docs/web-content/) ──────────
] + _scan_directory(DOCS_PATH_WEB_CONTENT, {
    "audience": "web-content",
    "priority": 2,
    "doc_type": "process",
})


# ─── Metadata Schema Validation ───────────────────────────────────────────────

def validate_sources(sources: list) -> bool:
    """
    Ensure every entry in SOURCES has all required metadata keys.
    Fails loud and early rather than producing unchunkable or unfilterable docs.
    """
    valid = True
    for i, src in enumerate(sources):
        missing = REQUIRED_METADATA_KEYS - set(src.get("metadata", {}).keys())
        if missing:
            print(f"  SCHEMA ERROR: SOURCES[{i}] ('{src.get('path', 'unknown')}') "
                  f"is missing required metadata keys: {missing}")
            valid = False
    return valid


# ─── Checksum Manifest ────────────────────────────────────────────────────────

def load_checksums() -> dict:
    """Load the checksum manifest from disk. Returns empty dict if none exists."""
    if os.path.exists(CHECKSUM_PATH):
        with open(CHECKSUM_PATH, "r") as f:
            return json.load(f)
    return {}

def save_checksums(checksums: dict):
    """Persist the checksum manifest to disk alongside the vectorstore."""
    os.makedirs(os.path.dirname(CHECKSUM_PATH), exist_ok=True)
    with open(CHECKSUM_PATH, "w") as f:
        json.dump(checksums, f, indent=2)

def compute_checksum(path: str) -> str:
    """
    Compute an MD5 hash of a local file's contents.
    Web sources always return "" so source_has_changed() is always True —
    web content is always re-fetched since remote content can change without notice.
    """
    if path.startswith("http"):
        return ""  # Fix: always re-fetch web sources — stable URL hash was a bug
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def source_has_changed(path: str, checksums: dict) -> bool:
    """Return True if the source is new or its content has changed since last ingest."""
    current = compute_checksum(path)
    return checksums.get(path) != current


# ─── Load & Tag ───────────────────────────────────────────────────────────────

def load_and_tag_documents(checksums: dict) -> tuple[list, dict]:
    """
    Load all sources, tag every page/chunk with metadata, and apply
    doc-type-specific chunking strategy before returning all chunks.

    Skips any PDF source whose checksum matches the last ingest run.
    Web sources are always re-fetched.
    Returns updated checksums alongside the loaded documents.
    """
    all_docs = []
    updated_checksums = checksums.copy()

    for src in SOURCES:
        path = src["path"]

        # Skip unchanged sources — no embedding cost, no re-indexing
        if not source_has_changed(path, checksums):
            print(f"\nSkipping (unchanged): {path}")
            continue

        print(f"\nProcessing: {path}...")

        try:
            if src["type"] == "pdf":
                loader = PyPDFLoader(path)
                raw_docs = loader.load()
            else:
                # SoupStrainer targets main content only — prevents nav/footer pollution
                loader = WebBaseLoader(
                    path,
                    bs_kwargs={"parse_only": SoupStrainer(["article", "main"])}
                )
                raw_docs = loader.load()

        except Exception as e:
            print(f"  ERROR loading {path}: {e} — skipping.")
            continue

        if not raw_docs:
            print(f"  WARNING: No content loaded from {path} — skipping.")
            continue

        # Tag every page with source metadata before splitting
        # This ensures every chunk inherits audience, priority, and doc_type
        for doc in raw_docs:
            doc.metadata.update(src["metadata"])

            # Scan detection — flag pages with suspiciously little text
            if len(doc.page_content.strip()) < 50:
                print(f"  WARNING: Possible scanned page in {path} "
                      f"(page {doc.metadata.get('page', '?')}) — text may be missing.")

        # Select chunking strategy based on doc_type
        # Fix: all four doc_types now have explicit branches — no silent fallthrough
        doc_type = src["metadata"]["doc_type"]
        if doc_type == "process":
            c_size, c_overlap = CHUNK_SIZE_PROCESS, CHUNK_OVERLAP_PROCESS
        elif doc_type == "standards":
            c_size, c_overlap = CHUNK_SIZE_STANDARDS, CHUNK_OVERLAP_STANDARDS
        elif doc_type == "template":
            c_size, c_overlap = CHUNK_SIZE_TEMPLATE, CHUNK_OVERLAP_TEMPLATE
        else:
            # "style" and any future types default to style config
            c_size, c_overlap = CHUNK_SIZE_STYLE, CHUNK_OVERLAP_STYLE

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"  → {len(chunks)} chunks ({doc_type}, {c_size} tokens).")
        all_docs.extend(chunks)

        # Update checksum — only written to disk after successful indexing
        # in main(), so a failed run won't mark a source as clean
        updated_checksums[path] = compute_checksum(path)

    return all_docs, updated_checksums


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate(docs):
    """
    Remove duplicate chunks based on content hash.
    Prevents double-indexing if the same source appears in multiple formats.
    """
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    removed = len(docs) - len(unique)
    if removed > 0:
        print(f"\nDeduplication: removed {removed} duplicate chunks.")
    return unique


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 0. Parse CLI flags ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Style RAG ingestion pipeline")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing vectorstore and rebuild from scratch"
    )
    args = parser.parse_args()

    # ── 1. Handle --rebuild flag ──────────────────────────────────────────────
    if args.rebuild and os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        print("Vectorstore deleted — rebuilding from scratch.\n")

    # ── 2. Validate SOURCES schema before doing any work ─────────────────────
    print("Validating source configuration...")
    if not validate_sources(SOURCES):
        print("\nAborting: fix the metadata errors above before re-running.")
        return
    print("  ✓ All sources valid.\n")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # ── 3. Load checksum manifest ─────────────────────────────────────────────
    checksums = load_checksums()
    if checksums:
        print(f"Loaded checksum manifest ({len(checksums)} entries).")
    else:
        print("No checksum manifest found — all sources will be indexed.")

    # ── 4. Clear stale chunks for sources that have changed ───────────────────
    if os.path.exists(VECTORSTORE_PATH):
        changed_sources = [
            src for src in SOURCES
            if source_has_changed(src["path"], checksums)
        ]
        if changed_sources:
            print("\nClearing stale chunks for changed sources...")
            vectorstore = Chroma(
                persist_directory=VECTORSTORE_PATH,
                embedding_function=embeddings
            )
            for src in changed_sources:
                source_name = src["metadata"]["source"]
                try:
                    vectorstore.delete(where={"source": source_name})
                    print(f"  Cleared: {source_name}")
                except Exception as e:
                    print(f"  Could not clear {source_name}: {e}")
        else:
            print("\nAll sources unchanged — nothing to re-index.")
            return

    # ── 5. Load, tag, chunk, and deduplicate ──────────────────────────────────
    documents, updated_checksums = load_and_tag_documents(checksums)
    documents = deduplicate(documents)

    if not documents:
        print("\nNo new documents to index.")
        return

    print(f"\nTotal chunks to index: {len(documents)}")

    # ── 6. Index into Chroma ──────────────────────────────────────────────────
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    # ── 7. Persist updated checksums — only after successful indexing ─────────
    save_checksums(updated_checksums)

    print(f"\nSuccess! Vectorstore updated at '{VECTORSTORE_PATH}'")
    print(f"Chunks indexed this run: {len(documents)}")
    print(f"Checksum manifest saved to '{CHECKSUM_PATH}'")
    print(f"\nTip: To rebuild from scratch next time, run: python ingest.py --rebuild")


if __name__ == "__main__":
    main()
