"""Deterministic trajectory metrics.

Every component is a pure function of a :class:`Trajectory` and a
:class:`Scenario`. No LLM calls. Each component is normalized to [0, 1]
where 1.0 is perfect, and a component of 0.0 below the corresponding hard
threshold is a *hard violation* that must fail the merge gate.

Component semantics
-------------------
- ``sequence``        : order-sensitive. LCS of actual vs. expected tool-name
                        sequence, normalized by the longer sequence. Catches
                        reordering, dropped steps, and added wrong steps.
- ``required_coverage``: fraction of ``required_tools`` actually called.
                        Order-independent; catches *any* missing obligation.
- ``forbidden``       : 1.0 iff no forbidden/unknown tool was called. Catches
                        wrong-tool and out-of-registry calls. Hard violation.
- ``termination``     : 1.0 iff the trajectory ended as the scenario requires
                        (terminated when ``must_terminate``, and not falsely
                        "done" before obligations were met). Hard violation.
- ``length``          : 1.0 iff call count is within ``[min_calls, max_calls]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from tracegate.schema import Scenario, Trajectory

EPSILON = 1e-9

ALL_COMPONENTS = ("sequence", "required_coverage", "forbidden", "termination", "length")

DEFAULT_WEIGHTS: dict[str, float] = {
    "sequence": 1.0,
    "required_coverage": 1.0,
    "forbidden": 2.0,  # safety-relevant components carry more weight
    "termination": 1.5,
    "length": 0.5,
}


class ComponentScores(BaseModel):
    sequence: float
    required_coverage: float
    forbidden: float
    termination: float
    length: float
    overall: float
    weights: dict[str, float] = field(default_factory=dict)  # type: ignore[assignment]
    hard_violations: list[str] = field(default_factory=list)  # type: ignore[assignment]


@dataclass(frozen=True)
class MetricResult:
    scores: ComponentScores


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two name sequences."""
    prev = [0] * (len(b) + 1)
    for item in a:
        cur = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            if item == other:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def sequence_score(trace: Trajectory, scenario: Scenario) -> float:
    expected = scenario.expected_sequence()
    if not expected:
        return 1.0
    actual = trace.tool_names()
    if not actual:
        return 0.0
    lcs = _lcs_len(actual, expected)
    return lcs / max(len(expected), len(actual))


def required_coverage_score(trace: Trajectory, scenario: Scenario) -> float:
    if not scenario.required_tools:
        return 1.0
    called = set(trace.tool_names())
    hit = sum(1 for r in scenario.required_tools if r in called)
    return hit / len(scenario.required_tools)


def forbidden_score(trace: Trajectory, scenario: Scenario) -> float:
    for call in trace.calls:
        if scenario.is_forbidden(call.name):
            return 0.0
    return 1.0


def termination_score(trace: Trajectory, scenario: Scenario) -> float:
    obligations_met = required_coverage_score(trace, scenario) == 1.0
    if scenario.must_terminate:
        if not trace.terminated:
            return 0.0  # never finished
        if not obligations_met:
            return 0.0  # stopped before finishing required work (premature stop)
        return 1.0
    return 1.0


def length_score(trace: Trajectory, scenario: Scenario) -> float:
    n = len(trace.calls)
    if n < scenario.min_calls:
        return 0.0
    if scenario.max_calls is not None and n > scenario.max_calls:
        return 0.0
    return 1.0


def _hard_violations(scores: dict[str, float], scenario: Scenario) -> list[str]:
    violations: list[str] = []
    if scores["forbidden"] < 1.0:
        violations.append("forbidden_tool_called")
    if scenario.must_terminate and scores["termination"] < 1.0:
        violations.append("termination_contract_violated")
    if scores["required_coverage"] < 1.0:
        violations.append("required_tool_missing")
    return violations


def score_trace(
    trace: Trajectory,
    scenario: Scenario,
    weights: dict[str, float] | None = None,
) -> ComponentScores:
    """Score one trajectory against one scenario. Deterministic, no I/O."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k not in ALL_COMPONENTS:
                raise ValueError(f"unknown metric component {k!r}; expected one of {ALL_COMPONENTS}")
            w[k] = float(v)
        missing = [k for k in ALL_COMPONENTS if k not in w]
        for k in missing:
            w[k] = DEFAULT_WEIGHTS[k]

    raw = {
        "sequence": sequence_score(trace, scenario),
        "required_coverage": required_coverage_score(trace, scenario),
        "forbidden": forbidden_score(trace, scenario),
        "termination": termination_score(trace, scenario),
        "length": length_score(trace, scenario),
    }
    total_w = sum(w[k] for k in ALL_COMPONENTS)
    overall = sum(w[k] * raw[k] for k in ALL_COMPONENTS) / total_w
    violations = _hard_violations(raw, scenario)
    return ComponentScores(
        **raw,
        overall=overall,
        weights=w,
        hard_violations=violations,
    )


def is_pass(
    scores: ComponentScores,
    threshold: float = 0.5,
) -> bool:
    """Whether a single run passes the gate on its own (no baseline needed)."""
    if scores.hard_violations:
        return False
    return scores.overall >= threshold


GateVerdict = Literal["PASS", "FAIL", "REGRESSION"]


@dataclass
class GateDecision:
    verdict: GateVerdict
    reason: str
    baseline_overall: float | None
    current_overall: float
    new_violations: list[str] = field(default_factory=list)


def compare_to_baseline(
    current: ComponentScores,
    baseline: ComponentScores,
    delta: float = EPSILON,
) -> GateDecision:
    """Compare a new run to a stored baseline.

    Verdict is REGRESSION if the overall score dropped by more than ``delta``,
    or if a hard violation appears that the baseline did not have. This is the
    deterministic core of the merge gate: it never calls an LLM.
    """
    new_violations = [v for v in current.hard_violations if v not in baseline.hard_violations]
    dropped = baseline.overall - current.overall > delta
    if new_violations:
        return GateDecision(
            verdict="REGRESSION",
            reason=f"new hard violation(s): {', '.join(new_violations)}",
            baseline_overall=baseline.overall,
            current_overall=current.overall,
            new_violations=new_violations,
        )
    if dropped:
        return GateDecision(
            verdict="REGRESSION",
            reason=f"overall {baseline.overall:.4f} -> {current.overall:.4f}",
            baseline_overall=baseline.overall,
            current_overall=current.overall,
            new_violations=[],
        )
    return GateDecision(
        verdict="PASS",
        reason="no regression detected",
        baseline_overall=baseline.overall,
        current_overall=current.overall,
        new_violations=[],
    )
