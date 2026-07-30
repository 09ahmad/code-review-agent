"""
Shared state object passed through every node in the LangGraph orchestrator.

Keeping everything in one typed structure (rather than passing loose args
between functions) is what makes this a real agent state machine instead of
a linear script — and it's what lets you log/inspect the full run for
"observability" in an interview.
"""

from __future__ import annotations

from typing import TypedDict, Optional


class Issue(TypedDict):
    source: str          # which tool found it: "static_analysis" | "review_model" | "test_runner"
    severity: str         # "low" | "medium" | "high"
    description: str
    line: Optional[int]
    suggested_fix: Optional[str]


class AgentState(TypedDict, total=False):
    # input
    diff: str
    file_path: Optional[str]

    # intermediate tool outputs
    static_analysis_issues: list[Issue]
    review_model_issues: list[Issue]
    review_model_available: bool
    test_results: dict
    retrieved_context: list[str]

    # aggregation / decision
    all_issues: list[Issue]
    confidence: float
    needs_human_review: bool

    # error tracking — every tool node reports failures here instead of
    # raising, so one broken tool never crashes the whole review. The UI
    # surfaces these as calm, honest notices rather than stack traces.
    tool_errors: dict[str, str]
    fatal_error: Optional[str]

    # output
    final_review: str
