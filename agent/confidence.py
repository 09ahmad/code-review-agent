"""
Confidence scoring: decides whether the agent should trust its own output
and post the review, or escalate to a human.

This is the piece that makes the system "agentic" rather than a fixed
pipeline — it's making a decision about its own reliability, which is
exactly the kind of self-correction interviewers ask about.

The heuristic here is intentionally simple and explainable (good for an
interview: you can walk through exactly why a given score was produced).
A fancier version could use the model's own token log-probs, or a second
"critic" model call — mention that as a future extension.
"""

from __future__ import annotations

from agent.state import Issue

# Tunable thresholds — pull these out as config if you want to experiment.
LOW_CONFIDENCE_THRESHOLD = 0.6
HIGH_SEVERITY_AUTO_ESCALATE = True


def score_confidence(
    static_issues: list[Issue],
    review_model_issues: list[Issue],
    test_results: dict,
) -> float:
    """
    Returns a confidence score in [0, 1] representing how much the agent
    trusts its combined findings enough to auto-post them without a human.

    Signals used:
      - Agreement between static analysis and the fine-tuned model
        (if both flag issues in a similar area, confidence goes up)
      - Whether tests passed (failing tests + review issues = more certain
        something is actually wrong)
      - Presence of high-severity issues (these should generally get a
        human's eyes regardless of "confidence", handled separately)
    """
    score = 0.75  # baseline

    static_lines = {i["line"] for i in static_issues if i.get("line")}
    model_lines = {i["line"] for i in review_model_issues if i.get("line")}
    if static_lines and model_lines:
        overlap = static_lines & model_lines
        if overlap:
            score += 0.15  # tools agree — more confident
        else:
            score -= 0.1  # tools disagree — less confident

    if not test_results.get("passed", True) and (
        static_issues or review_model_issues
    ):
        score += 0.1  # failing tests + found issues = consistent story

    if not static_issues and not review_model_issues and not model_lines:
        score -= 0.05  # nothing found anywhere — mildly less confident
        # this is a deliberate choice: "no issues" should not be free confidence

    return max(0.0, min(1.0, score))


def needs_human_review(
    confidence: float, all_issues: list[Issue]
) -> bool:
    """Decide whether to escalate, based on confidence and severity."""
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True
    if HIGH_SEVERITY_AUTO_ESCALATE:
        if any(i["severity"] == "high" for i in all_issues):
            return True
    return False
