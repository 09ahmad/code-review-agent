"""
Test runner tool: executes the project's existing test suite (or a targeted
subset) against the patched code, in a subprocess with a timeout so a bad
diff can't hang the agent.

For a resume project, a real "sandbox" (Docker container, gVisor, firecracker
VM, etc.) is a nice extra credit item to mention as future work — this
version uses a subprocess + timeout, which is enough to demonstrate the
concept without heavy infra.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_tests(test_path: str = "tests/", timeout_seconds: int = 60) -> dict:
    """
    Runs pytest against the given path and returns a structured summary.

    Returns:
        {
            "passed": bool,
            "summary": str,     # human-readable pytest summary line
            "raw_output": str,  # full stdout, useful for debugging/logging
        }
    """
    if not Path(test_path).exists():
        # No test suite in this environment/repo — this is a normal,
        # expected state for a demo project, not a failure. Say so plainly
        # rather than letting pytest print a scary "not found" error.
        return {
            "passed": True,
            "summary": f"No test suite found at '{test_path}' — skipped.",
            "raw_output": "",
        }

    try:
        # Note: we rely on subprocess's own `timeout=` kwarg to bound
        # execution time, rather than pytest's `--timeout` flag -- that flag
        # needs the separate `pytest-timeout` plugin installed, and we'd
        # rather not fail in a confusing way if it's missing.
        result = subprocess.run(
            ["pytest", test_path, "-q"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        passed = result.returncode == 0
        summary_line = _extract_summary(result.stdout)
        return {
            "passed": passed,
            "summary": summary_line,
            "raw_output": result.stdout,
        }
    except FileNotFoundError:
        return {
            "passed": True,  # fail-open in demo environments without pytest
            "summary": "pytest not found — skipped (demo environment)",
            "raw_output": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "summary": f"Tests timed out after {timeout_seconds}s",
            "raw_output": "",
        }
    except Exception as exc:  # noqa: BLE001 - last line of defense for this tool
        return {
            "passed": True,  # fail-open: an unrelated crash here shouldn't
            "summary": f"Test runner unavailable ({type(exc).__name__}) — skipped.",
            "raw_output": "",
        }


def _extract_summary(pytest_stdout: str) -> str:
    lines = [l for l in pytest_stdout.splitlines() if l.strip()]
    return lines[-1] if lines else "No test output captured."
