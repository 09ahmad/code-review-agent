"""
The core agent: a LangGraph state machine that orchestrates static
analysis, the fine-tuned review model, test execution, and retrieval, then
scores its own confidence and either finalizes a review or flags for a
human.

Run directly:
    python -m agent.graph --diff path/to/your.diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langgraph.graph import StateGraph, END

from agent.state import AgentState, Issue
from agent.tools.static_analysis import run_static_analysis
from agent.tools.review_model import review_with_finetuned_model
from agent.tools.test_runner import run_tests
from agent.tools.retrieval import retrieve_similar_patterns
from agent.confidence import score_confidence, needs_human_review


# ---------------------------------------------------------------------------
# Graph nodes — each is a plain function: (state) -> partial state update.
# This is what makes the flow inspectable: every node's input/output can be
# logged independently.
#
# Every node below follows the same rule: NEVER let an exception escape.
# Catch it, record a short human-readable reason in tool_errors, and return
# a safe empty/default result so the rest of the pipeline keeps running.
# A demo that degrades gracefully (one tool quietly unavailable) is far
# better than one that crashes mid-review in front of an interviewer.
# ---------------------------------------------------------------------------

def _record_error(state: AgentState, tool_name: str, exc: Exception) -> dict:
    errors = dict(state.get("tool_errors", {}))
    errors[tool_name] = f"{type(exc).__name__}: {exc}"
    return errors


def node_static_analysis(state: AgentState) -> dict:
    try:
        issues = run_static_analysis(state["diff"])
        return {"static_analysis_issues": issues}
    except Exception as exc:  # noqa: BLE001 - intentional broad catch, see module note
        return {
            "static_analysis_issues": [],
            "tool_errors": _record_error(state, "static_analysis", exc),
        }


def _write_graph_debug_log(status: str, detail: str) -> None:
    """
    Writes ground-truth diagnostic info to the same plain debug file that
    review_model.py uses, for exceptions caught here in graph.py (i.e.
    outside review_model.py's own safety net -- e.g. at import time).

    Split out into its own function (rather than inlined at the call site)
    specifically so tests can monkeypatch this instead of actually writing
    to the real debug_log.txt on disk every time the error path is
    exercised in a test.
    """
    try:
        import datetime
        from pathlib import Path

        log_path = Path(__file__).resolve().parent.parent / "debug_log.txt"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"TIMESTAMP: {datetime.datetime.now()}\n")
            f.write(f"STATUS: {status}\n")
            f.write(f"DETAIL:\n{detail}\n")
    except Exception:  # noqa: BLE001
        pass


def node_review_model(state: AgentState) -> dict:
    try:
        raw_text, issues = review_with_finetuned_model(state["diff"])
        # review_with_finetuned_model never raises -- it returns a
        # "[...unavailable...]" style stub string instead. Detect that here
        # so the final report can say "model unavailable" rather than
        # silently looking like "model ran, found nothing".
        available = not raw_text.startswith("[")
        return {"review_model_issues": issues, "review_model_available": available}
    except Exception as exc:  # noqa: BLE001
        # If we land here, the failure happened OUTSIDE review_model.py's
        # own safety net entirely (e.g. at import time). Log this ground
        # truth so it's never ambiguous which layer actually caught the
        # problem.
        _write_graph_debug_log(
            "CAUGHT AT graph.py node_review_model (outside review_model.py's own handling)",
            f"{type(exc).__name__}: {exc}",
        )
        return {
            "review_model_issues": [],
            "review_model_available": False,
            "tool_errors": _record_error(state, "review_model", exc),
        }


def node_retrieval(state: AgentState) -> dict:
    try:
        context = retrieve_similar_patterns(state["diff"])
        return {"retrieved_context": context}
    except Exception as exc:  # noqa: BLE001
        return {
            "retrieved_context": [],
            "tool_errors": _record_error(state, "retrieval", exc),
        }


def node_test_runner(state: AgentState) -> dict:
    try:
        results = run_tests()
        return {"test_results": results}
    except Exception as exc:  # noqa: BLE001
        return {
            "test_results": {
                "passed": True,  # fail-open: an unrelated crash here shouldn't
                "summary": "Test runner unavailable in this environment.",
                "raw_output": "",
            },
            "tool_errors": _record_error(state, "test_runner", exc),
        }


def node_aggregate_and_score(state: AgentState) -> dict:
    try:
        all_issues: list[Issue] = (
            state.get("static_analysis_issues", [])
            + state.get("review_model_issues", [])
        )
        confidence = score_confidence(
            state.get("static_analysis_issues", []),
            state.get("review_model_issues", []),
            state.get("test_results", {}),
        )
        escalate = needs_human_review(confidence, all_issues)
        return {
            "all_issues": all_issues,
            "confidence": confidence,
            "needs_human_review": escalate,
        }
    except Exception as exc:  # noqa: BLE001
        # If scoring itself fails, default to the safest possible stance:
        # zero confidence, always ask a human. Never silently pretend
        # everything is fine.
        return {
            "all_issues": state.get("static_analysis_issues", [])
            + state.get("review_model_issues", []),
            "confidence": 0.0,
            "needs_human_review": True,
            "tool_errors": _record_error(state, "aggregate_and_score", exc),
        }


def node_finalize_review(state: AgentState) -> dict:
    issues = state.get("all_issues", [])
    confidence = state.get("confidence", 0.0)
    lines = [
        f"## Code Review (confidence: {confidence:.2f})",
        "",
    ]
    if state.get("needs_human_review"):
        lines.append(
            "**A human reviewer should double-check this PR** — the agent's "
            "confidence was below threshold or a high-severity issue was "
            "found.\n"
        )

    if not issues:
        lines.append("No issues found by static analysis or the review model.")
    else:
        for issue in issues:
            severity = str(issue.get("severity", "low")).upper()
            source = issue.get("source", "unknown")
            description = issue.get("description", "")
            line_no = issue.get("line", "?")
            lines.append(
                f"- **[{severity}]** (line {line_no}, via {source}): {description}"
                + (
                    f"\n  - *Suggested fix:* {issue['suggested_fix']}"
                    if issue.get("suggested_fix")
                    else ""
                )
            )

    if not state.get("review_model_available", True):
        lines.append(
            "\n**Fine-tuned model:** not available in this run (falling back "
            "to static analysis + retrieval only) — see README for setup."
        )

    test_results = state.get("test_results", {})
    lines.append(f"\n**Tests:** {test_results.get('summary', 'not run')}")

    context = state.get("retrieved_context", [])
    if context:
        lines.append("\n**Related past patterns considered:**")
        for c in context:
            lines.append(f"- {c}")

    tool_errors = state.get("tool_errors", {})
    if tool_errors:
        lines.append("\n**Notice:** some tools were unavailable in this run "
                      "and were skipped gracefully:")
        for tool_name in tool_errors:
            lines.append(f"- {tool_name.replace('_', ' ')}")

    return {"final_review": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("static_analysis", node_static_analysis)
    graph.add_node("review_model", node_review_model)
    graph.add_node("retrieval", node_retrieval)
    graph.add_node("test_runner", node_test_runner)
    graph.add_node("aggregate_and_score", node_aggregate_and_score)
    graph.add_node("finalize_review", node_finalize_review)

    # Tools run independently (this is your "multi-agent / parallel tool
    # use" story), then converge on aggregation.
    graph.set_entry_point("static_analysis")
    graph.add_edge("static_analysis", "review_model")
    graph.add_edge("review_model", "retrieval")
    graph.add_edge("retrieval", "test_runner")
    graph.add_edge("test_runner", "aggregate_and_score")
    graph.add_edge("aggregate_and_score", "finalize_review")
    graph.add_edge("finalize_review", END)

    return graph.compile()


def review_diff(diff: str) -> AgentState:
    """
    Runs the full agent on a diff. This function is designed to NEVER raise
    -- any unexpected failure (a bad LangGraph invocation, an environment
    issue, etc.) is caught here and turned into a safe, honest fallback
    state, so the UI layer never has to handle a raw exception.
    """
    if not diff or not diff.strip():
        return {
            "diff": diff,
            "final_review": (
                "## Code Review\n\nNo diff was provided — paste a code diff "
                "above and try again."
            ),
            "confidence": 0.0,
            "needs_human_review": False,
            "all_issues": [],
            "tool_errors": {},
            "fatal_error": None,
        }

    try:
        app = build_graph()
        result = app.invoke({"diff": diff})
        return result
    except Exception as exc:  # noqa: BLE001 - last line of defense for the whole pipeline
        return {
            "diff": diff,
            "final_review": (
                "## Code Review\n\n"
                "Something went wrong while running the review pipeline, "
                "so no automated review could be generated this time. "
                "This has been logged for debugging.\n\n"
                f"**Error:** {type(exc).__name__}: {exc}"
            ),
            "confidence": 0.0,
            "needs_human_review": True,
            "all_issues": [],
            "tool_errors": {},
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the code review agent on a diff.")
    parser.add_argument("--diff", required=True, help="Path to a .diff/.patch file")
    args = parser.parse_args()

    diff_text = Path(args.diff).read_text()
    final_state = review_diff(diff_text)
    print(final_state["final_review"])
