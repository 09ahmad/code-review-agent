"""
Tests for agent/graph.py's error-handling contract.

The whole design premise of this agent is "one broken tool never crashes
the whole review" -- every node catches exceptions and returns a safe
fallback plus a note in tool_errors. These tests break each tool on purpose
(via monkeypatch) and check the node survives and reports it honestly,
rather than trusting that the try/except blocks stay correct over time.
"""

from __future__ import annotations

import agent.graph as graph_module


def test_static_analysis_node_survives_tool_exception(monkeypatch):
    def boom(diff):
        raise RuntimeError("ruff exploded")

    monkeypatch.setattr(graph_module, "run_static_analysis", boom)
    result = graph_module.node_static_analysis({"diff": "x = 1"})

    assert result["static_analysis_issues"] == []
    assert "static_analysis" in result["tool_errors"]
    assert "ruff exploded" in result["tool_errors"]["static_analysis"]


def test_review_model_node_survives_tool_exception(monkeypatch):
    def boom(diff):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(graph_module, "review_with_finetuned_model", boom)
    # This path also writes a ground-truth entry to the real debug_log.txt
    # on disk -- mock that out so running the test suite doesn't pollute
    # the actual project's debug log with synthetic test failures.
    monkeypatch.setattr(graph_module, "_write_graph_debug_log", lambda *a, **k: None)
    result = graph_module.node_review_model({"diff": "x = 1"})

    assert result["review_model_issues"] == []
    assert result["review_model_available"] is False
    assert "review_model" in result["tool_errors"]


def test_review_model_node_error_path_does_not_write_real_debug_log(monkeypatch, tmp_path):
    # Regression test for a real issue: this node's exception handler
    # writes ground-truth failure detail straight to the project's
    # debug_log.txt on disk. Left un-mocked, running this test suite
    # silently appends synthetic "model exploded" entries to that real
    # file every time. Confirm _write_graph_debug_log is the only thing
    # that touches disk here, so mocking it is sufficient and sufficient
    # is actually happening.
    calls = []
    monkeypatch.setattr(
        graph_module, "_write_graph_debug_log", lambda status, detail: calls.append((status, detail))
    )

    def boom(diff):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(graph_module, "review_with_finetuned_model", boom)
    graph_module.node_review_model({"diff": "x = 1"})

    assert len(calls) == 1
    assert "simulated failure" in calls[0][1]
    def boom(diff):
        raise RuntimeError("faiss exploded")

    monkeypatch.setattr(graph_module, "retrieve_similar_patterns", boom)
    result = graph_module.node_retrieval({"diff": "x = 1"})

    assert result["retrieved_context"] == []
    assert "retrieval" in result["tool_errors"]


def test_test_runner_node_survives_tool_exception(monkeypatch):
    def boom():
        raise RuntimeError("pytest exploded")

    monkeypatch.setattr(graph_module, "run_tests", boom)
    result = graph_module.node_test_runner({"diff": "x = 1"})

    # fail-open: an unrelated crash here should not block the review
    assert result["test_results"]["passed"] is True
    assert "test_runner" in result["tool_errors"]


def test_aggregate_and_score_node_defaults_to_escalate_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(graph_module, "score_confidence", boom)
    result = graph_module.node_aggregate_and_score(
        {"static_analysis_issues": [], "review_model_issues": []}
    )

    # Safest possible stance when scoring itself breaks: zero confidence,
    # always ask a human, never silently look fine.
    assert result["confidence"] == 0.0
    assert result["needs_human_review"] is True
    assert "aggregate_and_score" in result["tool_errors"]


def test_review_diff_never_raises_even_on_total_pipeline_failure(monkeypatch):
    def boom():
        raise RuntimeError("graph build exploded")

    monkeypatch.setattr(graph_module, "build_graph", boom)
    result = graph_module.review_diff("x = 1")

    assert result["needs_human_review"] is True
    assert result["confidence"] == 0.0
    assert "Something went wrong" in result["final_review"]
    assert result["fatal_error"] is not None


def test_review_diff_handles_empty_diff_without_running_the_graph():
    result = graph_module.review_diff("   ")
    assert "No diff was provided" in result["final_review"]
    assert result["needs_human_review"] is False
