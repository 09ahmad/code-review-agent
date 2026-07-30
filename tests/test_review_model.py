"""
Tests for agent/tools/review_model.py.

Uses lightweight fakes instead of the real ~3GB model throughout, since
the thing under test is "does the wrapper logic behave correctly" (prompt
format, repetition truncation, parsing, size cap, timeout), not the
model's own output quality.
"""

from __future__ import annotations

import time

import agent.tools.review_model as review_model_module


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self, fake_review="- severity: high\n  description: test issue\n  line: 1\n"):
        self.captured_prompt = None
        self._fake_review = fake_review

    def __call__(self, prompt, return_tensors="pt"):
        self.captured_prompt = prompt
        return _FakeBatch({"input_ids": [[1, 2, 3]]})

    def decode(self, ids, skip_special_tokens=True):
        if ids == [1, 2, 3]:
            return self.captured_prompt
        return self.captured_prompt + self._fake_review


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return ["__FAKE_GENERATED_SEQUENCE__"]


class _SlowFakeModel:
    """Simulates a generation call that takes longer than the timeout."""
    device = "cpu"

    def generate(self, **kwargs):
        time.sleep(2)
        return ["__FAKE_GENERATED_SEQUENCE__"]


# ---------------------------------------------------------------------------
# Prompt format (training/inference match)
# ---------------------------------------------------------------------------

def test_prompt_uses_raw_completion_template_not_chat_template(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    monkeypatch.setattr(review_model_module, "_load_model", lambda: (_FakeModel(), fake_tokenizer))
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    review_model_module.review_with_finetuned_model("+ x = 1\n")

    prompt = fake_tokenizer.captured_prompt
    assert prompt.startswith("### Instruction:\n")
    assert "### Diff:\n+ x = 1\n" in prompt
    assert prompt.endswith("### Review:\n")
    assert "<|im_start|>" not in prompt


def test_prompt_matches_training_template_byte_for_byte():
    training_template = "### Instruction:\n{system_prompt}\n\n### Diff:\n{diff}\n\n### Review:\n"
    assert review_model_module.PROMPT_TEMPLATE == training_template


# ---------------------------------------------------------------------------
# Repetition-loop truncation + parser
# ---------------------------------------------------------------------------

def test_truncates_hallucinated_repetition_at_first_hash_marker():
    raw = (
        "- severity: medium\n  description: SQL injection vulnerability.\n  lines: [2]\n"
        "- severity: low\n  description: Missing docstring for function.\n  lines: [1] ### Review:\n"
        "- severity: high\n  description: SQL injection vulnerability.\n  lines: [2]\n"
    )
    truncated = review_model_module._truncate_at_hallucinated_continuation(raw)
    assert "###" not in truncated
    assert truncated.count("SQL injection vulnerability.") == 1


def test_parser_handles_plural_lines_with_bracketed_list():
    issues = review_model_module._parse_issues(
        "- severity: medium\n  description: SQL injection vulnerability.\n  lines: [2]\n"
    )
    assert len(issues) == 1
    assert issues[0]["line"] == 2


def test_parser_still_handles_original_singular_line_format():
    issues = review_model_module._parse_issues(
        "- severity: high\n  description: SQL injection.\n  line: 2\n"
    )
    assert len(issues) == 1
    assert issues[0]["line"] == 2


# ---------------------------------------------------------------------------
# Large-diff size cap
# ---------------------------------------------------------------------------

def test_oversized_diff_is_rejected_without_loading_the_model(monkeypatch):
    def fail_if_called():
        raise AssertionError("_load_model should never be called for an oversized diff")

    monkeypatch.setattr(review_model_module, "_load_model", fail_if_called)
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    huge_diff = "+ x = 1\n" * (review_model_module.MAX_DIFF_CHARS)  # way over the cap
    raw_text, issues = review_model_module.review_with_finetuned_model(huge_diff)

    assert "too large" in raw_text
    assert str(review_model_module.MAX_DIFF_CHARS) in raw_text
    assert issues == []


def test_diff_exactly_at_cap_is_not_rejected(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    monkeypatch.setattr(review_model_module, "_load_model", lambda: (_FakeModel(), fake_tokenizer))
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    diff_at_cap = "+" + ("x" * (review_model_module.MAX_DIFF_CHARS - 1))
    assert len(diff_at_cap) == review_model_module.MAX_DIFF_CHARS

    raw_text, _ = review_model_module.review_with_finetuned_model(diff_at_cap)
    assert "too large" not in raw_text


def test_diff_one_char_over_cap_is_rejected(monkeypatch):
    def fail_if_called():
        raise AssertionError("_load_model should never be called")

    monkeypatch.setattr(review_model_module, "_load_model", fail_if_called)
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    diff_over_cap = "+" + ("x" * review_model_module.MAX_DIFF_CHARS)
    assert len(diff_over_cap) == review_model_module.MAX_DIFF_CHARS + 1

    raw_text, issues = review_model_module.review_with_finetuned_model(diff_over_cap)
    assert "too large" in raw_text
    assert issues == []


# ---------------------------------------------------------------------------
# Generation timeout
# ---------------------------------------------------------------------------

def test_slow_generation_times_out_and_returns_control(monkeypatch):
    monkeypatch.setattr(review_model_module, "GENERATION_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(review_model_module, "_load_model", lambda: (_SlowFakeModel(), _FakeTokenizer()))
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    started = time.monotonic()
    raw_text, issues = review_model_module.review_with_finetuned_model("+ x = 1\n")
    elapsed = time.monotonic() - started

    # The whole point of the timeout: the caller gets control back close to
    # the timeout window, not after the full 2-second fake generation.
    assert elapsed < 1.5
    assert "timed out" in raw_text
    assert "0.3s" in raw_text
    assert issues == []


def test_fast_generation_completes_normally_within_timeout(monkeypatch):
    monkeypatch.setattr(review_model_module, "GENERATION_TIMEOUT_SECONDS", 5)
    fake_tokenizer = _FakeTokenizer(fake_review="- severity: high\n  description: real issue.\n  line: 1\n")
    monkeypatch.setattr(review_model_module, "_load_model", lambda: (_FakeModel(), fake_tokenizer))
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    raw_text, issues = review_model_module.review_with_finetuned_model("+ x = 1\n")

    assert "timed out" not in raw_text
    assert len(issues) == 1
    assert issues[0]["description"] == "real issue."


# ---------------------------------------------------------------------------
# Model-unavailable path
# ---------------------------------------------------------------------------

def test_model_unavailable_returns_stub_without_calling_tokenizer(monkeypatch):
    monkeypatch.setattr(review_model_module, "_load_model", lambda: (None, None))
    monkeypatch.setattr(review_model_module, "_last_failure_reason", "No trained adapter found.")
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda *a, **k: None)

    raw_text, issues = review_model_module.review_with_finetuned_model("+ x = 1\n")

    assert raw_text.startswith("[Fine-tuned model unavailable")
    assert issues == []


# ---------------------------------------------------------------------------
# Debug log isolation (regression guard against test pollution)
# ---------------------------------------------------------------------------

def test_review_never_writes_real_debug_log_when_mocked(monkeypatch):
    calls = []
    monkeypatch.setattr(review_model_module, "_write_debug_log", lambda status, detail: calls.append(status))
    fake_tokenizer = _FakeTokenizer()
    monkeypatch.setattr(review_model_module, "_load_model", lambda: (_FakeModel(), fake_tokenizer))

    review_model_module.review_with_finetuned_model("+ x = 1\n")

    assert calls == ["SUCCESS"]
