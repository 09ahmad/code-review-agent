"""
Loads the QLoRA-fine-tuned review model (adapter produced by
finetuning/train_qlora_colab.ipynb) and runs inference on a diff.

IMPORTANT: this must build the prompt using the exact same raw
"### Instruction: / ### Diff: / ### Review:" completion-style template
that finetuning/prepare_dataset.py and train_qlora_colab.ipynb actually
train on -- NOT the model's ChatML chat template. The training notebook
tokenizes that raw text directly (no apply_chat_template call anywhere in
it), so the LoRA adapter only ever learned to treat the literal string
"### Review:\\n" as the cue to start generating structured output. Wrapping
the same diff in ChatML at inference time is a different surface pattern
the adapter never saw fine-tuned behavior on -- it doesn't error, it just
silently falls back toward generic base-model behavior. Keep this in sync
with the notebook if either one changes.

Falls back to a clear stub response if no adapter is found yet, or if
model loading/generation fails for any reason -- so the rest of the agent
pipeline (static analysis, retrieval, confidence scoring) is always
demoable, even before fine-tuning is done or if something's wrong with the
local model environment (e.g. no GPU, out of memory, missing packages).

LARGE-DIFF SAFETY: review_with_finetuned_model() enforces a character cap
(MAX_DIFF_CHARS) before even attempting generation, and wraps generation
in a hard wall-clock timeout (GENERATION_TIMEOUT_SECONDS). Important
honesty about the timeout: transformers/torch generation isn't safely
interruptible mid-call from another thread, so the underlying model.generate()
call is NOT forcibly killed when the timeout fires -- it keeps running in
a background thread until it finishes on its own. What the timeout
actually buys you is that THIS FUNCTION stops waiting and returns control
back to the caller (and the UI) immediately, instead of the whole app
appearing frozen indefinitely. A fully rigorous version of this would run
generation in a separate process (which CAN be forcibly killed) rather
than a thread -- flagged here as a known next step, not implemented, since
it's a much bigger architectural change than this project's scope.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

# Forcing these two lazy submodules to fully initialize now (at import
# time) avoids a real race condition where their first-touch lazy import
# can collide with a thread's shutdown/recycling and fail with
# "RuntimeError: can't register atexit after shutdown", which then
# surfaces confusingly as a transformers ImportError.
import concurrent.futures
import concurrent.futures.thread
import concurrent.futures.process
_ = concurrent.futures.ThreadPoolExecutor
_ = concurrent.futures.ProcessPoolExecutor

from agent.state import Issue

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ADAPTER_PATH = _PROJECT_ROOT / "finetuning" / "qlora-code-review-adapter"
ADAPTER_PATH = os.getenv("ADAPTER_PATH", str(_DEFAULT_ADAPTER_PATH))

SYSTEM_PROMPT = (
    "You are an expert code reviewer. Given a code diff, identify real "
    "issues (bugs, security problems, style violations, missing tests). "
    "For each issue, output: severity (low/medium/high), a one-line "
    "description, and the line(s) affected. If there are no issues, say so."
)

# IMPORTANT: this must match, character for character, the template used
# in finetuning/train_qlora_colab.ipynb and finetuning/prepare_dataset.py.
# The adapter was fine-tuned on plain Alpaca-style completions of this
# exact template ("### Review:\n" is the literal cue the model learned to
# treat as "now output structured issues") -- it was NOT trained using
# Qwen's ChatML chat template.
PROMPT_TEMPLATE = "### Instruction:\n{system_prompt}\n\n### Diff:\n{diff}\n\n### Review:\n"

# A diff longer than this is rejected before generation is even attempted,
# rather than risking a very slow (or effectively stuck) single-pass
# generation call on CPU. ~6000 characters is a deliberately conservative
# cap for a 1.5B model doing single-shot review on modest hardware --
# comfortably covers the kind of single-function/single-file diffs this
# tool is actually good at, while explicitly refusing to silently degrade
# on a multi-file, multi-hundred-line PR.
MAX_DIFF_CHARS = 6000

# Wall-clock budget for a single generation call. See the large-diff
# safety note in the module docstring for what this timeout does and does
# NOT guarantee.
GENERATION_TIMEOUT_SECONDS = 45

import threading

_model = None
_tokenizer = None
# Set by EITHER the loading step or the generation step, whichever fails,
# with a FULL traceback (not just the exception message) -- so
# get_unavailable_reason() always shows the real, complete picture of
# whatever actually went wrong, no matter which step it happened in.
_last_failure_reason = None
# Guards _load_model() so that if multiple calls happen close together
# (e.g. impatient double-clicking, or overlapping Streamlit reruns), only
# ONE of them actually loads the ~3GB model -- the rest wait for that one
# to finish and then reuse its result, instead of all independently
# starting their own full model load at the same time.
_load_lock = threading.Lock()

# A single-worker executor dedicated to running generation calls under a
# timeout. Long-lived (module-level) rather than created fresh per call,
# so an abandoned (timed-out) generation doesn't accumulate its own new
# executor/thread on every single call.
_generation_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="review-model-generation"
)


def _load_model():
    """
    Lazily loads the fine-tuned model. Cached across calls within a process
    ONLY on success -- a failed attempt is retried on the next call rather
    than being permanently blacklisted for the rest of the process.

    Protected by a lock so that if multiple calls arrive close together,
    only one of them actually performs the (slow, memory-heavy) load --
    the rest wait for it and reuse the result.

    Returns (model, tokenizer) or (None, None) if unavailable. Never raises.
    """
    global _model, _tokenizer, _last_failure_reason

    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return _model, _tokenizer

        if not Path(ADAPTER_PATH).exists():
            _last_failure_reason = f"No trained adapter found at '{ADAPTER_PATH}'."
            return None, None

        try:
            import torch
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

            _model, _tokenizer = model, tokenizer
            return _model, _tokenizer
        except Exception:  # noqa: BLE001 - deliberately broad, see module note
            _last_failure_reason = "LOADING FAILED:\n" + traceback.format_exc()
            return None, None


_base_model = None
_base_tokenizer = None
_base_load_lock = threading.Lock()


def load_base_model_only():
    """
    Loads the plain base model with NO LoRA adapter attached -- used only
    by finetuning/evaluate.py to get a fair "before fine-tuning" baseline
    to compare the adapter against. The live app never calls this.

    Returns (model, tokenizer) or (None, None). Never raises.
    """
    global _base_model, _base_tokenizer
    if _base_model is not None:
        return _base_model, _base_tokenizer

    with _base_load_lock:
        if _base_model is not None:
            return _base_model, _base_tokenizer
        try:
            import torch
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
            )
            _base_model, _base_tokenizer = model, tokenizer
            return _base_model, _base_tokenizer
        except Exception:  # noqa: BLE001
            return None, None


def get_unavailable_reason():
    """Returns the full failure detail (loading OR generation), if any."""
    return _last_failure_reason


def review_with_finetuned_model(diff):
    """
    Runs the fine-tuned model on a diff, with two safety nets a raw
    "just call model.generate()" version doesn't have:

      1. A hard character cap (MAX_DIFF_CHARS) rejects oversized diffs
         BEFORE attempting generation at all, rather than discovering the
         hard way that a huge multi-file diff makes single-pass CPU
         generation impractically slow.
      2. A wall-clock timeout (GENERATION_TIMEOUT_SECONDS) around the
         generation call itself, so a pathologically slow generation
         doesn't leave the whole app looking frozen with no way out. See
         the module docstring for exactly what this timeout does and does
         NOT guarantee (it can't force-kill an in-progress generate() call).

    Returns:
        (raw_text_review, parsed_issues)

    Never raises -- returns a clear stub/error message instead if the diff
    is too large, the model can't be loaded, generation times out, or
    generation fails for any other reason.
    """
    global _last_failure_reason

    if len(diff) > MAX_DIFF_CHARS:
        msg = (
            f"[Diff too large for fine-tuned model review: {len(diff)} characters "
            f"exceeds the {MAX_DIFF_CHARS}-character cap] "
            "Skipped the model review step to avoid an impractically slow "
            "single-pass generation call on a large diff. Static analysis "
            "and tests still ran on the full diff. For large PRs, review "
            "one file (or one logical chunk) at a time."
        )
        _write_debug_log("SKIPPED (diff too large)", msg)
        return msg, []

    model, tokenizer = _load_model()

    if model is None:
        reason = _last_failure_reason or "unknown reason"
        stub_text = (
            f"[Fine-tuned model unavailable: {reason}] "
            "Falling back to static analysis and retrieval only."
        )
        _write_debug_log("LOAD FAILED", reason)
        return stub_text, []

    future = _generation_executor.submit(_generate_with_model, model, tokenizer, diff)
    try:
        return future.result(timeout=GENERATION_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        msg = (
            f"[Fine-tuned model review timed out after {GENERATION_TIMEOUT_SECONDS}s] "
            "Falling back to static analysis and retrieval only. Note: the "
            "underlying generation call may still be running in the "
            "background -- it isn't force-killed (see module docstring) -- "
            "but the app itself stops waiting."
        )
        _last_failure_reason = msg
        _write_debug_log("GENERATION TIMEOUT", msg)
        return msg, []


def _generate_with_model(model, tokenizer, diff):
    """
    Runs generation + parsing for an already-loaded (model, tokenizer)
    pair, with NO size cap or timeout of its own -- those are
    review_with_finetuned_model's job (the live-app-facing wrapper).

    Split out on purpose so finetuning/evaluate.py can call this exact same
    code path directly against both the fine-tuned model and the plain
    base model for a fair, apples-to-apples comparison, without eval runs
    (which use small, fixed examples) being affected by production-only
    guardrails meant for arbitrary user input.
    """
    global _last_failure_reason
    try:
        prompt = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, diff=diff)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            # A no-repeat n-gram constraint helps the model avoid sliding
            # into an exact-repeat loop in the first place, but it's not a
            # full fix on its own -- see the truncation below.
            no_repeat_ngram_size=6,
        )
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )
        review_text = full_text[len(prompt_decoded):].strip()
        review_text = _truncate_at_hallucinated_continuation(review_text)
        _write_debug_log("SUCCESS", review_text)
        return review_text, _parse_issues(review_text)
    except Exception:  # noqa: BLE001
        _last_failure_reason = "GENERATION FAILED:\n" + traceback.format_exc()
        _write_debug_log("GENERATION FAILED", _last_failure_reason)
        stub_text = (
            "[Review model failed during generation] "
            "Falling back to static analysis and retrieval only. "
            "See Tool status for the full error."
        )
        return stub_text, []


def _truncate_at_hallucinated_continuation(review_text):
    """
    The raw "### Instruction / ### Diff / ### Review" completion template
    has no natural stop token the way a chat template's <|im_end|> does --
    nothing tells the model "you're done" after it finishes the real
    review. In practice this small model sometimes runs straight into a
    hallucinated repeat of its own answer, or a fake new "### Review:" /
    "### Instruction:" block, and keeps going until it hits the token cap.

    A literal "###" should never legitimately appear inside real review
    content (it's reserved template syntax), so truncating at the first
    occurrence reliably keeps only the model's actual first answer.
    """
    if "###" in review_text:
        review_text = review_text.split("###", 1)[0]
    return review_text.strip()


def _write_debug_log(status, detail):
    """
    Writes ground-truth diagnostic info directly to a plain text file on
    disk, bypassing Streamlit's UI/session/caching layer entirely.
    """
    import datetime
    log_path = _PROJECT_ROOT / "debug_log.txt"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"TIMESTAMP: {datetime.datetime.now()}\n")
            f.write(f"STATUS: {status}\n")
            f.write(f"DETAIL:\n{detail}\n")
    except Exception:  # noqa: BLE001
        pass  # logging must never itself break the app


def _parse_issues(review_text):
    """
    Very lightweight parser for the model's YAML-ish output format.
    For production you'd want the model to emit strict JSON (or use
    function-calling / structured output) — flag this as a "next step" in
    interviews if asked how you'd harden it.

    Accepts both "line: 2" and "lines: [2]" / "lines: [2, 5]" -- the model
    doesn't always reproduce the training format's exact singular "line:"
    key or bare-int value.
    """
    issues = []
    current = {}
    for line in review_text.splitlines():
        line = line.strip("- ").strip()
        if not line:
            continue
        if line.startswith("severity:"):
            if current:
                issues.append(_finalize_issue(current))
            current = {"severity": line.split(":", 1)[1].strip()}
        elif line.startswith("description:"):
            current["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("line:") or line.startswith("lines:"):
            val = line.split(":", 1)[1].strip()
            val = val.strip("[]")
            first_num = val.split(",")[0].strip()
            current["line"] = int(first_num) if first_num.isdigit() else None
        elif line.startswith("suggested_fix:"):
            current["suggested_fix"] = line.split(":", 1)[1].strip()
    if current:
        issues.append(_finalize_issue(current))
    return [i for i in issues if i["severity"] != "none"]


def _finalize_issue(d):
    return Issue(
        source="review_model",
        severity=d.get("severity", "low"),
        description=d.get("description", ""),
        line=d.get("line"),
        suggested_fix=d.get("suggested_fix"),
    )
