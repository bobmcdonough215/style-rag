"""
Unit tests for rag.py core functions.
Tests pure logic only — no LLM calls, no vectorstore, no network.
"""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import rag


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_doc(content, source="Test Doc", page=None, priority=1):
    """Create a mock document with metadata."""
    meta = {"source": source, "priority": priority}
    if page is not None:
        meta["page"] = page
    return SimpleNamespace(page_content=content, metadata=meta)


# ─── strip_thinking ──────────────────────────────────────────────────────────

class TestStripThinking:
    def test_removes_think_tags(self):
        text = "<think>some reasoning</think>The answer is 42."
        assert rag.strip_thinking(text) == "The answer is 42."

    def test_removes_multiline_think(self):
        text = "<think>\nline 1\nline 2\n</think>\nClean output."
        assert rag.strip_thinking(text) == "Clean output."

    def test_no_think_tags_unchanged(self):
        text = "Just a normal response."
        assert rag.strip_thinking(text) == "Just a normal response."

    def test_empty_string(self):
        assert rag.strip_thinking("") == ""

    def test_only_think_tags(self):
        text = "<think>all thinking no answer</think>"
        assert rag.strip_thinking(text) == ""

    def test_multiple_think_blocks(self):
        text = "<think>first</think>Hello <think>second</think>world."
        assert rag.strip_thinking(text) == "Hello world."


# ─── rerank_by_priority ──────────────────────────────────────────────────────

class TestRerankByPriority:
    def test_sorts_ascending(self):
        docs = [
            _make_doc("c", priority=3),
            _make_doc("a", priority=1),
            _make_doc("b", priority=2),
        ]
        result = rag.rerank_by_priority(docs)
        assert [d.metadata["priority"] for d in result] == [1, 2, 3]

    def test_missing_priority_sorts_last(self):
        doc_with = _make_doc("a", priority=1)
        doc_without = SimpleNamespace(
            page_content="b", metadata={"source": "X"}
        )
        result = rag.rerank_by_priority([doc_without, doc_with])
        assert result[0].metadata["priority"] == 1
        assert result[1].metadata.get("priority") is None

    def test_same_priority_preserves_order(self):
        docs = [
            _make_doc("first", priority=1),
            _make_doc("second", priority=1),
        ]
        result = rag.rerank_by_priority(docs)
        assert result[0].page_content == "first"
        assert result[1].page_content == "second"

    def test_empty_list(self):
        assert rag.rerank_by_priority([]) == []

    def test_single_doc(self):
        docs = [_make_doc("only", priority=5)]
        result = rag.rerank_by_priority(docs)
        assert len(result) == 1


# ─── build_citations ─────────────────────────────────────────────────────────

class TestBuildCitations:
    def test_basic_citation(self):
        docs = [_make_doc("content", source="AP Stylebook", page=7)]
        result = rag.build_citations(docs)
        assert len(result) == 1
        assert result[0]["source"] == "AP Stylebook"
        assert result[0]["page"] == 7
        assert result[0]["label"] == "AP Stylebook, p. 7"

    def test_no_page_number(self):
        docs = [_make_doc("content", source="House Style Guide")]
        result = rag.build_citations(docs)
        assert result[0]["label"] == "House Style Guide"
        assert result[0]["page"] is None

    def test_deduplication(self):
        docs = [
            _make_doc("chunk 1", source="AP Stylebook", page=7),
            _make_doc("chunk 2", source="AP Stylebook", page=7),
            _make_doc("chunk 3", source="AP Stylebook", page=12),
        ]
        result = rag.build_citations(docs)
        assert len(result) == 2
        labels = {c["label"] for c in result}
        assert "AP Stylebook, p. 7" in labels
        assert "AP Stylebook, p. 12" in labels

    def test_empty_docs(self):
        assert rag.build_citations([]) == []

    def test_missing_source_metadata(self):
        doc = SimpleNamespace(page_content="x", metadata={})
        result = rag.build_citations([doc])
        assert result[0]["source"] == "Unknown source"


# ─── format_citations ────────────────────────────────────────────────────────

class TestFormatCitations:
    def test_formats_single(self):
        citations = [{"source": "AP", "page": 1, "label": "AP, p. 1"}]
        result = rag.format_citations(citations)
        assert result == "*Source: AP, p. 1*"

    def test_formats_multiple(self):
        citations = [
            {"source": "A", "page": None, "label": "A"},
            {"source": "B", "page": 2, "label": "B, p. 2"},
        ]
        result = rag.format_citations(citations)
        assert "*Source: A*" in result
        assert "*Source: B, p. 2*" in result
        assert result.count("\n") == 1

    def test_empty_returns_empty(self):
        assert rag.format_citations([]) == ""


# ─── format_context ──────────────────────────────────────────────────────────

class TestFormatContext:
    def test_joins_with_separator(self):
        docs = [_make_doc("First chunk"), _make_doc("Second chunk")]
        result = rag.format_context(docs)
        assert "First chunk" in result
        assert "Second chunk" in result
        assert "\n\n---\n\n" in result

    def test_single_doc_no_separator(self):
        docs = [_make_doc("Only chunk")]
        result = rag.format_context(docs)
        assert result == "Only chunk"
        assert "---" not in result

    def test_empty_list(self):
        assert rag.format_context([]) == ""


# ─── get_system_prompt ────────────────────────────────────────────────────────

class TestGetSystemPrompt:
    def test_question_strict(self):
        prompt = rag.get_system_prompt("question", strict=True)
        assert "ONLY" in prompt
        assert "context" in prompt.lower()

    def test_question_relaxed(self):
        prompt = rag.get_system_prompt("question", strict=False)
        assert "supplement" in prompt.lower() or "general knowledge" in prompt.lower()

    def test_review_includes_few_shot(self):
        prompt = rag.get_system_prompt("review")
        assert "Issues Found" in prompt
        assert "Fully Corrected Version" in prompt

    def test_web_content_mode(self):
        prompt = rag.get_system_prompt("web-content")
        assert "CMS" in prompt or "web content" in prompt.lower()

    def test_format_mode(self):
        prompt = rag.get_system_prompt("format")
        assert "format" in prompt.lower()

    def test_conflict_mode(self):
        prompt = rag.get_system_prompt("conflict")
        assert "precedence" in prompt.lower() or "conflict" in prompt.lower()


# ─── retrieve (with mock vectorstore) ────────────────────────────────────────

class TestRetrieve:
    def _mock_vectorstore(self, scores):
        """Create a mock vectorstore returning docs with given scores."""
        vs = MagicMock()
        results = [
            (_make_doc(f"chunk {i}", priority=1), score)
            for i, score in enumerate(scores)
        ]
        vs.similarity_search_with_relevance_scores.return_value = results
        return vs

    def test_filters_below_threshold(self):
        vs = self._mock_vectorstore([0.5, 0.4, 0.2, 0.1])
        docs = rag.retrieve("test", vs)
        # Only 0.5 and 0.4 are above 0.30 threshold
        assert len(docs) == 2

    def test_skip_threshold_returns_all(self):
        vs = self._mock_vectorstore([0.5, 0.1, 0.05, 0.01])
        docs = rag.retrieve("test", vs, skip_threshold=True)
        assert len(docs) == 4

    def test_with_scores_returns_tuples(self):
        vs = self._mock_vectorstore([0.5, 0.4])
        result = rag.retrieve("test", vs, with_scores=True)
        assert len(result) == 2
        assert isinstance(result[0], tuple)
        doc, score = result[0]
        assert score == 0.5

    def test_without_scores_returns_docs(self):
        vs = self._mock_vectorstore([0.5, 0.4])
        result = rag.retrieve("test", vs, with_scores=False)
        assert len(result) == 2
        assert hasattr(result[0], "page_content")

    def test_empty_results(self):
        vs = self._mock_vectorstore([])
        assert rag.retrieve("test", vs) == []

    def test_all_below_threshold(self):
        vs = self._mock_vectorstore([0.1, 0.05, 0.01])
        assert rag.retrieve("test", vs) == []

    def test_respects_top_k_limit(self):
        # Generate more docs than TOP_K_CHUNKS
        scores = [0.9 - i * 0.05 for i in range(20)]
        vs = self._mock_vectorstore(scores)
        from config import TOP_K_CHUNKS
        docs = rag.retrieve("test", vs)
        assert len(docs) <= TOP_K_CHUNKS


# ─── finalize_response ───────────────────────────────────────────────────────

class TestFinalizeResponse:
    def test_strips_thinking_and_builds_citations(self):
        docs = [_make_doc("content", source="AP Style", page=3)]
        result = rag.finalize_response(
            "<think>reasoning</think>The answer.", docs
        )
        assert result["answer"] == "The answer."
        assert len(result["citations"]) == 1
        assert result["num_chunks"] == 1

    def test_empty_docs(self):
        result = rag.finalize_response("Some answer.", [])
        assert result["answer"] == "Some answer."
        assert result["citations"] == []
        assert result["citations_formatted"] == ""
        assert result["num_chunks"] == 0


# ─── RAGPipelineError ────────────────────────────────────────────────────────

class TestRAGPipelineError:
    def test_is_exception(self):
        assert issubclass(rag.RAGPipelineError, Exception)

    def test_retrieve_raises_on_vectorstore_failure(self):
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.side_effect = ConnectionError("db down")
        with pytest.raises(rag.RAGPipelineError, match="Retrieval failed"):
            rag.retrieve("test", vs)


# ─── Config Validation ───────────────────────────────────────────────────────

class TestConfigValidation:
    def test_current_config_is_valid(self):
        """Importing config should not raise — validates current values."""
        import config  # noqa: F401

    def test_validate_catches_bad_threshold(self):
        import config
        original = config.SIMILARITY_THRESHOLD
        try:
            config.SIMILARITY_THRESHOLD = 2.0
            with pytest.raises(ValueError, match="SIMILARITY_THRESHOLD"):
                config._validate()
        finally:
            config.SIMILARITY_THRESHOLD = original

    def test_validate_catches_bad_top_k(self):
        import config
        original = config.TOP_K_CHUNKS
        try:
            config.TOP_K_CHUNKS = -1
            with pytest.raises(ValueError, match="TOP_K_CHUNKS"):
                config._validate()
        finally:
            config.TOP_K_CHUNKS = original
