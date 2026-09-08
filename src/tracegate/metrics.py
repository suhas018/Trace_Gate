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

Custom Metrics
--------------
Register custom metrics via :class:`MetricRegistry`::

    from tracegate.metrics import registry, MetricPlugin

    class MyMetric:
        name = "custom_ratio"
        default_weight = 1.0
        is_hard_violation = False

        def score(self, trace: Trajectory, scenario: Scenario) -> float:
            return 1.0 if len(trace.calls) > 0 else 0.0

    registry.register(MyMetric())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tracegate.schema import Scenario, Trajectory

EPSILON = 1e-9


@runtime_checkable
class MetricPlugin(Protocol):
    """Protocol for custom metric components.

    Any object implementing this protocol can be registered as a metric.
    The ``score`` method receives a trajectory and scenario, returning a
    float in [0.0, 1.0] where 1.0 is perfect.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this metric (e.g., 'tool_arg_accuracy')."""
        ...

    @property
    def default_weight(self) -> float:
        """Default weight in the overall score calculation."""
        ...

    @property
    def is_hard_violation(self) -> bool:
        """If True, a score < 1.0 is a hard violation that fails the gate."""
        ...

    def score(self, trace: Trajectory, scenario: Scenario) -> float:
        """Compute the metric score. Must return [0.0, 1.0]."""
        ...


@dataclass
class _BuiltInMetric:
    """Wrapper to adapt a function to MetricPlugin protocol."""

    _name: str
    _weight: float
    _is_hard: bool
    _fn: callable  # type: ignore

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_weight(self) -> float:
        return self._weight

    @property
    def is_hard_violation(self) -> bool:
        return self._is_hard

    def score(self, trace: Trajectory, scenario: Scenario) -> float:
        return float(self._fn(trace, scenario))


class MetricRegistry:
    """Registry for metric plugins.

    Use the global ``registry`` instance to register custom metrics
    or retrieve all registered metrics.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, MetricPlugin] = {}

    def register(self, metric: MetricPlugin, *, override: bool = False) -> None:
        """Register a metric plugin.

        Args:
            metric: The metric to register.
            override: If True, allow overwriting an existing metric with the same name.

        Raises:
            ValueError: If a metric with the same name is already registered
                        and override is False.
        """
        if metric.name in self._metrics and not override:
            raise ValueError(
                f"metric {metric.name!r} already registered; "
                f"use override=True to replace it"
            )
        self._metrics[metric.name] = metric

    def unregister(self, name: str) -> MetricPlugin:
        """Remove and return a registered metric by name.

        Raises:
            KeyError: If the metric is not registered.
        """
        return self._metrics.pop(name)

    def get(self, name: str) -> MetricPlugin | None:
        """Retrieve a metric by name, or None if not found."""
        return self._metrics.get(name)

    def all(self) -> list[MetricPlugin]:
        """Return all registered metrics."""
        return list(self._metrics.values())

    def names(self) -> tuple[str, ...]:
        """Return names of all registered metrics."""
        return tuple(self._metrics.keys())

    def clear(self) -> None:
        """Remove all registered metrics. Primarily for testing."""
        self._metrics.clear()


# Global registry instance
registry = MetricRegistry()


def _register_builtins() -> None:
    """Register the 5 built-in metric components."""
    builtins = [
        _BuiltInMetric("sequence", 1.0, False, sequence_score),
        _BuiltInMetric("required_coverage", 1.0, False, required_coverage_score),
        _BuiltInMetric("forbidden", 2.0, True, forbidden_score),
        _BuiltInMetric("termination", 1.5, True, termination_score),
        _BuiltInMetric("length", 0.5, False, length_score),
    ]
    for m in builtins:
        registry.register(m, override=True)


ALL_COMPONENTS = ("sequence", "required_coverage", "forbidden", "termination", "length")

DEFAULT_WEIGHTS: dict[str, float] = {
    "sequence": 1.0,
    "required_coverage": 1.0,
    "forbidden": 2.0,  # safety-relevant components carry more weight
    "termination": 1.5,
    "length": 0.5,
}


class ComponentScores(BaseModel):
    """Scores for each metric component plus the overall weighted score."""

    model_config = {"extra": "allow"}  # Allow dynamic fields for custom metrics

    sequence: float
    required_coverage: float
    forbidden: float
    termination: float
    length: float
    overall: float
    weights: dict[str, float] = Field(default_factory=dict)
    hard_violations: list[str] = Field(default_factory=list)


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
    """Detect hard violations from scored components.

    Uses the registry to check which metrics are marked as hard violations.
    """
    violations: list[str] = []

    # Check built-in hard violations (backward compatible)
    if scores.get("forbidden", 1.0) < 1.0:
        violations.append("forbidden_tool_called")
    if scenario.must_terminate and scores.get("termination", 1.0) < 1.0:
        violations.append("termination_contract_violated")
    if scores.get("required_coverage", 1.0) < 1.0:
        violations.append("required_tool_missing")

    # Check custom hard violation metrics from registry
    for metric in registry.all():
        if metric.is_hard_violation and metric.name not in (
            "forbidden",
            "termination",
            "required_coverage",
        ):
            score = scores.get(metric.name, 1.0)
            if score < 1.0:
                violations.append(f"{metric.name}_violation")

    return violations


def get_default_weights() -> dict[str, float]:
    """Get default weights from the registry for all registered metrics."""
    weights: dict[str, float] = {}
    for metric in registry.all():
        weights[metric.name] = metric.default_weight
    return weights


def score_trace(
    trace: Trajectory,
    scenario: Scenario,
    weights: dict[str, float] | None = None,
    *,
    include_custom: bool = True,
) -> ComponentScores:
    """Score one trajectory against one scenario. Deterministic, no I/O.

    Args:
        trace: The agent's trajectory to score.
        scenario: The scenario specification.
        weights: Optional custom weights. If provided, overrides default weights
                 for the specified metrics. Unknown metric names raise ValueError.
        include_custom: If True (default), include all registered metrics (built-in
                        and custom) in the score. If False, only use the 5 built-in
                        metrics.
    """
    # Build weight map from registry defaults
    w = get_default_weights()

    # Apply user-provided weight overrides
    if weights:
        valid_metrics = set(w.keys())
        for k, v in weights.items():
            if k not in valid_metrics:
                raise ValueError(
                    f"unknown metric component {k!r}; "
                    f"expected one of {sorted(valid_metrics)}"
                )
            w[k] = float(v)

    # Determine which metrics to compute
    if include_custom:
        metric_names = tuple(w.keys())
    else:
        metric_names = ALL_COMPONENTS
        # Ensure only built-in weights are used
        w = {k: w[k] for k in ALL_COMPONENTS if k in w}

    # Score each metric
    raw: dict[str, float] = {}
    for name in metric_names:
        metric = registry.get(name)
        if metric is not None:
            raw[name] = metric.score(trace, scenario)
        else:
            # Fallback for unknown metrics (should not happen with registry)
            raise ValueError(f"metric {name!r} not found in registry")

    # Compute overall weighted score
    total_w = sum(w.get(k, 0.0) for k in metric_names)
    if total_w > 0:
        overall = sum(w.get(k, 0.0) * raw[k] for k in metric_names) / total_w
    else:
        overall = 0.0

    # Detect hard violations
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


# Register built-in metrics at module load time
_register_builtins()
