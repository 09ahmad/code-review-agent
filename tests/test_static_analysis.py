"""
Tests for agent/tools/static_analysis.py.

The first group is a regression test for a real bug: `_extract_code_from_diff`
was stripping one extra leading space from every '+' and context line, on
top of the diff marker itself. Unified diff has no such separator -- the
marker character is immediately followed by the original line, indentation
and all. That extra strip silently ate one space of indentation from any
indented line, which corrupts the reconstructed source for anything nested
(inside a function, an if-block, etc.) -- exactly the code we most want the
linter and confidence scorer to see correctly.
"""

from __future__ import annotations

import shutil

import pytest

from agent.tools.static_analysis import _extract_code_from_diff, run_static_analysis


def test_extracts_added_lines_preserving_indentation():
    diff = (
        "+def get_user(user_id):\n"
        "+    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
        "+    return db.execute(query)\n"
    )
    code = _extract_code_from_diff(diff)
    lines = code.splitlines()
    assert lines[0] == "def get_user(user_id):"
    # Before the fix, this line lost its leading space and became
    # "   query = ..." (3 spaces instead of 4) -- broken indentation.
    assert lines[1] == '    query = f"SELECT * FROM users WHERE id = {user_id}"'
    assert lines[2] == "    return db.execute(query)"


def test_extracts_hand_typed_separator_space_diffs_preserving_indentation():
    # This is the exact text used by demo/app.py's built-in "SQL injection"
    # example: hand-typed with a separator space after each '+', on top of
    # the code's own indentation ('+     query = ...' = separator + 4
    # spaces of real indent, not 5 spaces of real indent).
    diff = (
        '+ def get_user(user_id):\n'
        '+     query = f"SELECT * FROM users WHERE id = {user_id}"\n'
        '+     return db.execute(query)\n'
    )
    code = _extract_code_from_diff(diff)
    lines = code.splitlines()
    assert lines[0] == "def get_user(user_id):"
    assert lines[1] == '    query = f"SELECT * FROM users WHERE id = {user_id}"'
    assert lines[2] == "    return db.execute(query)"
    # Must be valid, parseable Python -- this is what was producing the
    # "invalid-syntax: Unexpected indentation" error in the UI.
    import ast
    ast.parse(code)


def test_extracts_context_lines_preserving_indentation():
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    code = _extract_code_from_diff(diff)
    lines = code.splitlines()
    assert lines[0] == "def foo():"
    assert lines[1] == "    return 2"


def test_removed_lines_are_dropped():
    diff = "+kept = 1\n-removed = 2\n"
    code = _extract_code_from_diff(diff)
    assert "removed" not in code
    assert "kept = 1" in code


def test_diff_metadata_headers_are_skipped():
    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        "+x = 1\n"
    )
    code = _extract_code_from_diff(diff)
    assert code.strip() == "x = 1"


def test_plain_code_without_diff_markers_passes_through_unchanged():
    # Note: a plain-code line starting with a space is indistinguishable
    # from a diff "context line" marker, so this only round-trips exactly
    # for lines with no leading whitespace -- a separate, pre-existing
    # ambiguity in this function, not something this test set covers.
    code_in = "def foo():\nreturn 1\n"
    code = _extract_code_from_diff(code_in)
    assert code == "def foo():\nreturn 1"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_nested_indented_code_lints_without_indentation_errors():
    # Regression guard at the integration level, using the exact text from
    # demo/app.py's built-in "SQL injection" example. With either bug
    # present (missing OR wrong-convention space stripping), the
    # reconstructed source has broken indentation, which ruff reports as
    # E999 (syntax error) instead of the real issue we care about.
    diff = (
        '+ def get_user(user_id):\n'
        '+     query = f"SELECT * FROM users WHERE id = {user_id}"\n'
        '+     return db.execute(query)\n'
    )
    issues = run_static_analysis(diff)
    codes = [issue["description"].split(":")[0] for issue in issues]
    assert "E999" not in codes  # E999 = ruff's syntax-error code
