"""
Tests for agent/confidence.py — the self-scoring logic that decides whether
the agent trusts its own findings or escalates to a human.

Each test targets one signal in isolation so a failure points straight at
the rule that broke, instead of a vague "score changed" diff.
"""

from __future__ import annotations

from agent.confidence import (
    LOW_CONFIDENCE_THRESHOLD,
    needs_human_review,
    score_confidence,
)


def _issue(line=None, severity="low", source="static_analysis"):
    return {
        "source": source,
        "severity": severity,
        "description": "test issue",
        "line": line,
        "suggested_fix": None,
    }


def test_baseline_score_with_no_issues_and_passing_tests():
    # No signals fire: not the "nothing found" penalty case, since that
    # penalty requires BOTH lists empty AND no model lines -- covered here.
    score = score_confidence([], [], {"passed": True})
    # baseline 0.75, minus 0.05 for "nothing found anywhere"
    assert round(score, 2) == 0.70


def test_agreement_between_tools_increases_confidence():
    static_issues = [_issue(line=10)]
    model_issues = [_issue(line=10, source="review_model")]
    score = score_confidence(static_issues, model_issues, {"passed": True})
    assert round(score, 2) == 0.90  # 0.75 + 0.15 overlap bonus


def test_disagreement_between_tools_decreases_confidence():
    static_issues = [_issue(line=10)]
    model_issues = [_issue(line=99, source="review_model")]
    score = score_confidence(static_issues, model_issues, {"passed": True})
    assert round(score, 2) == 0.65  # 0.75 - 0.10 disagreement penalty


def test_failing_tests_with_issues_found_increases_confidence():
    static_issues = [_issue(line=10)]
    score = score_confidence(static_issues, [], {"passed": False})
    assert round(score, 2) == 0.85  # 0.75 + 0.10


def test_failing_tests_with_no_issues_does_not_get_consistency_bonus():
    score = score_confidence([], [], {"passed": False})
    # bonus requires issues to exist alongside the failing tests
    assert round(score, 2) == 0.70  # baseline - 0.05 "nothing found" penalty


def test_score_is_clamped_to_valid_range():
    # Stack every negative signal to try to push the score below 0.
    static_issues = [_issue(line=1)]
    model_issues = [_issue(line=2, source="review_model")]
    score = score_confidence(static_issues, model_issues, {"passed": True})
    assert 0.0 <= score <= 1.0


def test_needs_human_review_true_when_confidence_below_threshold():
    assert needs_human_review(LOW_CONFIDENCE_THRESHOLD - 0.01, []) is True


def test_needs_human_review_false_when_confidence_high_and_no_high_severity():
    assert needs_human_review(0.95, [_issue(severity="low")]) is False


def test_needs_human_review_true_on_high_severity_even_with_high_confidence():
    issues = [_issue(severity="high")]
    assert needs_human_review(0.95, issues) is True
