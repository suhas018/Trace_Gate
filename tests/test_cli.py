import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scenarios" / "example.yaml"
GOOD = ROOT / "scenarios" / "behavior_good.yaml"
BUGGY = ROOT / "scenarios" / "behavior_buggy.yaml"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tracegate.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_baseline_creates_file(tmp_path):
    out = tmp_path / "baseline.json"
    res = run_cli("baseline", str(SUITE), "-o", str(out), "--mutation")
    assert res.returncode == 0, res.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["summary"]["scenarios"] == 2
    assert data["summary"]["mean_kill_rate"] == 1.0


def test_gate_passes_on_good_behavior(tmp_path):
    ref = tmp_path / "baseline.json"
    run_cli("baseline", str(SUITE), "-o", str(ref))
    res = run_cli("gate", str(SUITE), "-b", str(GOOD), "-r", str(ref))
    assert res.returncode == 0, res.stderr
    assert "gate: PASS" in res.stdout


def test_gate_fails_on_buggy_behavior(tmp_path):
    ref = tmp_path / "baseline.json"
    run_cli("baseline", str(SUITE), "-o", str(ref))
    res = run_cli("gate", str(SUITE), "-b", str(BUGGY), "-r", str(ref))
    assert res.returncode == 1
    assert "REGRESSION" in res.stdout
    assert "forbidden_tool_called" in res.stdout or "termination_contract_violated" in res.stdout


def test_mutate_command_reports_kill_rate():
    res = run_cli("mutate", str(SUITE))
    assert res.returncode == 0, res.stderr
    assert "kill-rate=1.00" in res.stdout


def test_run_command_no_behavior_derives_plan():
    res = run_cli("run", str(SUITE))
    assert res.returncode == 0, res.stderr
    assert "mean=" in res.stdout
    assert "sc_summarize_orders" in res.stdout
