"""
tests/test_risk_gate.py
~~~~~~~~~~~~~~~~~~~~~~~~
Manual verification harness for claude-plugin/scripts/risk_gate.py.

Not a pytest file — this plugin lives outside triage/, and per this repo's
core.md Rule 1 (no tool/framework-specific code in core), it isn't part of
the triage package or its pytest suite. Mirrors the existing non-pytest
pattern used by scripts/smoke_test.py in the main repo.

Run with:
    python3 claude-plugin/tests/test_risk_gate.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PLUGIN_ROOT / "scripts" / "risk_gate.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HOOKS_JSON = PLUGIN_ROOT / "hooks" / "hooks.json"

# Run risk_gate.py with the repo root on PYTHONPATH so `import triage`
# resolves to the local checkout without requiring `pip install -e .`.
REPO_ROOT = PLUGIN_ROOT.parent

CASES = [
    ("high_risk_bash.json", "deny"),
    ("high_risk_mcp.json", "deny"),
    ("medium_risk_bash.json", "ask"),
    ("clean_bash.json", "allow"),
]


def run_case(fixture: str, expected_decision: str) -> None:
    payload = (FIXTURES / fixture).read_text()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert proc.returncode == 0, f"{fixture}: nonzero exit {proc.returncode}, stderr={proc.stderr}"
    out = json.loads(proc.stdout)
    got = out["hookSpecificOutput"]["permissionDecision"]
    assert got == expected_decision, (
        f"{fixture}: expected {expected_decision}, got {got} ({proc.stdout!r})"
    )
    print(f"PASS  {fixture:30s} -> {got}")


def test_matcher_excludes_write() -> None:
    hooks = json.loads(HOOKS_JSON.read_text())
    matcher = hooks["hooks"]["PreToolUse"][0]["matcher"]
    assert not re.fullmatch(matcher, "Write"), "default matcher must not match built-in Write tool"
    assert re.fullmatch(matcher, "Bash")
    assert re.fullmatch(matcher, "mcp__email__send_email")
    print(f"PASS  matcher {matcher!r} excludes Write, includes Bash and mcp__*")


def test_fail_open_missing_triage() -> None:
    payload = (FIXTURES / "clean_bash.json").read_text()
    # -S skips `site` initialization, so even an editable pip install of
    # triage-agent (a .pth entry processed by `site`) is invisible here —
    # `import triage` fails regardless of how this machine has triage
    # installed, genuinely exercising the fail-open path.
    proc = subprocess.run(
        [sys.executable, "-S", str(SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert proc.returncode == 0, f"nonzero exit {proc.returncode}, stderr={proc.stderr}"
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "WARNING" in proc.stderr, f"expected fail-open WARNING on stderr, got: {proc.stderr!r}"
    print("PASS  fail-open path (triage unimportable) -> allow + stderr warning")


if __name__ == "__main__":
    for fixture, expected in CASES:
        run_case(fixture, expected)
    test_matcher_excludes_write()
    test_fail_open_missing_triage()
    print("\nAll risk-gate harness checks passed.")
