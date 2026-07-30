"""
Static analysis tool: runs `ruff` on the changed Python file(s) and converts
its output into the shared Issue format.

Ruff is used because it's fast (Rust-based) and covers a huge rule set
(pyflakes, pycodestyle, bugbear, etc.) with zero config needed for a demo.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from agent.state import Issue


def _extract_code_from_diff(diff_text: str) -> str:
    """
    Converts diff-style text (lines prefixed with '+', '-', or a unified-diff
    context space) into plausible plain Python source, so a linter can
    actually parse it.

    Without this step, a linter fed raw diff text (e.g. "+ def foo():")
    sees invalid syntax on every single line -- not because the underlying
    code has a problem, but because '+' isn't valid Python. That produces
    noisy, meaningless "invalid-syntax" spam instead of real findings.

    Handles two different diff conventions, since both show up in practice:
      (a) real unified diff (e.g. `git diff` output): the marker character
          is immediately followed by the original line, indentation and
          all -- there is no separator space.
      (b) hand-authored "readable" diffs (common in docs/demos, including
          this app's own built-in example diffs): exactly one separator
          space is inserted after the marker, in addition to the code's
          own indentation -- e.g. '+ def foo():' / '+     return 1'.
    Blindly assuming one or the other silently corrupts indentation for
    whichever convention wasn't assumed, so this auto-detects instead.
    """
    code_lines = []
    for line in diff_text.splitlines():
        # Skip unified-diff metadata (file headers, hunk markers) if present
        if line.startswith(("diff --git", "index ", "--- ", "+++ ", "@@")):
            continue
        # Skip removed lines entirely -- they're not part of the resulting file
        if line.startswith("-") and not line.startswith("---"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            rest = line[1:]
        elif line.startswith(" "):
            # unified-diff context line -- marker is one char, same as above
            rest = line[1:]
        else:
            # No diff marker at all -- assume it's already plain code
            rest = line
        code_lines.append(rest)

    # Detect convention (b): a top-level statement is essentially always at
    # column 0 in real source, and Python is never structurally indented by
    # exactly one space -- so a lone leading space on the first non-blank
    # line is a strong, simple signal that a separator space is present and
    # needs to be stripped from every line, not just that one.
    first_nonblank = next((l for l in code_lines if l.strip()), "")
    uses_separator_space = first_nonblank.startswith(" ") and not first_nonblank.startswith("  ")

    if uses_separator_space:
        code_lines = [l[1:] if l.startswith(" ") else l for l in code_lines]

    return "\n".join(code_lines)


def run_static_analysis(diff_text: str) -> list[Issue]:
    """
    Runs ruff on the code contained in a diff and returns structured issues.

    Accepts diff-style text (lines prefixed with '+'/'-') OR plain code --
    diff markers are stripped internally before linting, since a linter
    needs valid, parseable source rather than raw diff syntax.
    """
    code = _extract_code_from_diff(diff_text)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            ["ruff", "check", tmp_path, "--output-format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if not result.stdout.strip():
            return []

        raw_issues = json.loads(result.stdout)
        issues: list[Issue] = []
        for item in raw_issues:
            issues.append(
                Issue(
                    source="static_analysis",
                    severity=_map_severity(item.get("code", "")),
                    description=f"{item.get('code')}: {item.get('message')}",
                    line=item.get("location", {}).get("row"),
                    suggested_fix=item.get("fix", {}).get("message")
                    if item.get("fix")
                    else None,
                )
            )
        return issues
    except FileNotFoundError:
        # ruff not installed in this environment — fail soft so the agent
        # can still run the rest of the pipeline in a demo/dev environment.
        return []
    except subprocess.TimeoutExpired:
        return []
    except json.JSONDecodeError:
        # ruff produced unexpected output (e.g. a version mismatch) —
        # degrade gracefully rather than crashing the whole review.
        return []
    except Exception:  # noqa: BLE001 - last line of defense for this tool
        return []
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def _map_severity(rule_code: str) -> str:
    """Very rough heuristic mapping of ruff rule prefixes to severity."""
    high_prefixes = ("S",)       # flake8-bandit (security)
    medium_prefixes = ("B", "E9")  # bugbear, syntax-adjacent errors
    if rule_code.startswith(high_prefixes):
        return "high"
    if rule_code.startswith(medium_prefixes):
        return "medium"
    return "low"
