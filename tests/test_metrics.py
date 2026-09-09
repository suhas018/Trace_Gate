import pytest

from tracegate.metrics import (
    MetricPlugin,
    MetricRegistry,
    registry,
    score_trace,
)
from tracegate.schema import Scenario, ToolCall, Trajectory


def make_scenario(**kw):
    defaults = dict(
        id="s",
        prompt="p",
        expected_tool_calls=[{"name": "a"}, {"name": "b"}],
        required_tools=["a", "b"],
        forbidden_tools=["z"],
        allowed_tools=["a", "b"],
    )
    defaults.update(kw)
    return Scenario(**defaults)


def make_trace(calls, terminated=True):
    return Trajectory(
        scenario_id="s",
        calls=[ToolCall(index=i, name=n) for i, n in enumerate(calls)],
        terminated=terminated,
    )


def test_perfect_trace_scores_1():
    sc = make_scenario()
    tr = make_trace(["a", "b"])
    s = score_trace(tr, sc)
    assert s.overall == pytest.approx(1.0)
    assert s.hard_violations == []


def test_wrong_tool_is_hard_violation():
    sc = make_scenario()
    tr = make_trace(["a", "z"])  # forbidden tool
    s = score_trace(tr, sc)
    assert s.forbidden == 0.0
    assert "forbidden_tool_called" in s.hard_violations


def test_unknown_tool_outside_allowed_is_violation():
    sc = make_scenario()
    tr = make_trace(["a", "b", "HACKER_CMD"])
    s = score_trace(tr, sc)
    assert s.forbidden == 0.0


def test_premature_termination_fails_termination():
    sc = make_scenario()
    tr = make_trace(["a"], terminated=True)  # stopped before calling b
    s = score_trace(tr, sc)
    assert s.termination == 0.0
    assert "termination_contract_violated" in s.hard_violations


def test_never_terminates_fails_termination():
    sc = make_scenario()
    tr = make_trace(["a", "b"], terminated=False)
    s = score_trace(tr, sc)
    assert s.termination == 0.0


def test_reorder_lowers_sequence_not_coverage():
    sc = make_scenario()
    tr = make_trace(["b", "a"])  # same set, wrong order
    s = score_trace(tr, sc)
    assert s.required_coverage == 1.0
    assert s.sequence < 1.0
    assert s.overall < 1.0


def test_missing_required_lowers_coverage():
    sc = make_scenario()
    tr = make_trace(["a"])
    s = score_trace(tr, sc)
    assert s.required_coverage == pytest.approx(0.5)
    assert "required_tool_missing" in s.hard_violations


def test_length_bounds():
    sc = make_scenario(min_calls=2, max_calls=2)
    assert score_trace(make_trace(["a", "b"]), sc).length == 1.0
    assert score_trace(make_trace(["a", "b", "c"]), sc).length == 0.0
    assert score_trace(make_trace(["a"]), sc).length == 0.0


def test_empty_expected_sequence_is_neutral():
    sc = make_scenario(expected_tool_calls=[])
    tr = make_trace(["a", "b"])
    s = score_trace(tr, sc)
    assert s.sequence == 1.0


def test_unknown_weight_component_raises():
    sc = make_scenario()
    tr = make_trace(["a", "b"])
    with pytest.raises(ValueError):
        score_trace(tr, sc, weights={"not_a_metric": 1.0})


# ---- Custom Metric Plugin Tests ----


class SimpleMetric:
    """A simple custom metric for testing."""

    @property
    def name(self) -> str:
        return "simple_ratio"

    @property
    def default_weight(self) -> float:
        return 1.0

    @property
    def is_hard_violation(self) -> bool:
        return False

    def score(self, trace: Trajectory, scenario: Scenario) -> float:
        # Simple metric: 1.0 if any calls, 0.0 otherwise
        return 1.0 if len(trace.calls) > 0 else 0.0


class HardViolationMetric:
    """A custom metric that can trigger hard violations."""

    @property
    def name(self) -> str:
        return "must_have_result"

    @property
    def default_weight(self) -> float:
        return 1.0

    @property
    def is_hard_violation(self) -> bool:
        return True

    def score(self, trace: Trajectory, scenario: Scenario) -> float:
        # Fail if any call has no result
        for call in trace.calls:
            if call.result is None:
                return 0.0
        return 1.0


def test_registry_has_builtins():
    """Built-in metrics should be registered at import time."""
    names = registry.names()
    assert "sequence" in names
    assert "required_coverage" in names
    assert "forbidden" in names
    assert "termination" in names
    assert "length" in names


def test_register_custom_metric():
    """Custom metrics can be registered."""
    test_registry = MetricRegistry()
    metric = SimpleMetric()
    test_registry.register(metric)
    assert test_registry.get("simple_ratio") is metric


def test_register_duplicate_raises():
    """Registering a duplicate metric without override raises."""
    test_registry = MetricRegistry()
    metric = SimpleMetric()
    test_registry.register(metric)
    with pytest.raises(ValueError, match="already registered"):
        test_registry.register(metric)


def test_register_duplicate_with_override():
    """Registering with override=True replaces the metric."""
    test_registry = MetricRegistry()
    metric1 = SimpleMetric()
    metric2 = SimpleMetric()
    test_registry.register(metric1)
    test_registry.register(metric2, override=True)
    assert test_registry.get("simple_ratio") is metric2


def test_unregister_metric():
    """Unregistering removes the metric."""
    test_registry = MetricRegistry()
    metric = SimpleMetric()
    test_registry.register(metric)
    removed = test_registry.unregister("simple_ratio")
    assert removed is metric
    assert test_registry.get("simple_ratio") is None


def test_unregister_unknown_raises():
    """Unregistering an unknown metric raises KeyError."""
    test_registry = MetricRegistry()
    with pytest.raises(KeyError):
        test_registry.unregister("nonexistent")


def test_metric_plugin_protocol():
    """SimpleMetric should satisfy MetricPlugin protocol."""
    metric = SimpleMetric()
    assert isinstance(metric, MetricPlugin)


def test_custom_metric_in_score_trace():
    """Custom metrics are included in score_trace output."""
    sc = make_scenario()
    tr = make_trace(["a", "b"])

    # Register custom metric temporarily
    metric = SimpleMetric()
    registry.register(metric, override=True)
    try:
        scores = score_trace(tr, sc)
        assert hasattr(scores, "simple_ratio")
        assert scores.simple_ratio == 1.0  # type: ignore
        assert "simple_ratio" in scores.weights
    finally:
        registry.unregister("simple_ratio")


def test_custom_metric_zero_score():
    """Custom metric can return 0.0 for empty trajectories."""
    sc = make_scenario()
    tr = make_trace([])

    metric = SimpleMetric()
    registry.register(metric, override=True)
    try:
        scores = score_trace(tr, sc)
        assert scores.simple_ratio == 0.0  # type: ignore
    finally:
        registry.unregister("simple_ratio")


def test_include_custom_false_ignores_custom():
    """include_custom=False excludes custom metrics."""
    sc = make_scenario()
    tr = make_trace(["a", "b"])

    metric = SimpleMetric()
    registry.register(metric, override=True)
    try:
        scores = score_trace(tr, sc, include_custom=False)
        assert not hasattr(scores, "simple_ratio")
    finally:
        registry.unregister("simple_ratio")


def test_custom_hard_violation_metric():
    """Custom hard violation metrics trigger violations."""
    sc = make_scenario()
    # Create trace with a call that has no result
    tr = Trajectory(
        scenario_id="s",
        calls=[ToolCall(index=0, name="a", result=None)],
        terminated=True,
    )

    metric = HardViolationMetric()
    registry.register(metric, override=True)
    try:
        scores = score_trace(tr, sc)
        assert scores.must_have_result == 0.0  # type: ignore
        assert "must_have_result_violation" in scores.hard_violations
    finally:
        registry.unregister("must_have_result")


def test_custom_metric_weight_override():
    """Custom metrics can have their weights overridden."""
    sc = make_scenario()
    tr = make_trace(["a", "b"])

    metric = SimpleMetric()
    registry.register(metric, override=True)
    try:
        scores = score_trace(tr, sc, weights={"simple_ratio": 5.0})
        assert scores.weights["simple_ratio"] == 5.0
    finally:
        registry.unregister("simple_ratio")


def test_registry_clear():
    """Clear removes all metrics."""
    test_registry = MetricRegistry()
    test_registry.register(SimpleMetric())
    assert len(test_registry.all()) == 1
    test_registry.clear()
    assert len(test_registry.all()) == 0


def test_registry_all_returns_list():
    """all() returns a list of registered metrics."""
    test_registry = MetricRegistry()
    test_registry.register(SimpleMetric())
    all_metrics = test_registry.all()
    assert isinstance(all_metrics, list)
    assert len(all_metrics) == 1


# ---- P1 New Metric Tests ----

def test_arg_accuracy_no_pins_is_one():
    sc = make_scenario(expected_tool_calls=[{"name": "a"}, {"name": "b"}])  # no pinned args
    tr = make_trace(["a", "b"])
    s = score_trace(tr, sc)
    assert s.arg_accuracy == pytest.approx(1.0)


def test_arg_accuracy_pinned_match():
    from tracegate.schema import ExpectedToolCall

    sc = Scenario(
        id="s",
        prompt="p",
        expected_tool_calls=[ExpectedToolCall(name="a", arguments={"x": 1}), ExpectedToolCall(name="b", arguments={"y": 2})],
        required_tools=["a", "b"],
    )
    from tracegate.schema import ToolCall, Trajectory

    tr_good = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="a", arguments={"x": 1}), ToolCall(index=1, name="b", arguments={"y": 2})], terminated=True)
    tr_bad = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="a", arguments={"x": 99}), ToolCall(index=1, name="b", arguments={"y": 2})], terminated=True)
    assert score_trace(tr_good, sc).arg_accuracy == pytest.approx(1.0)
    assert score_trace(tr_bad, sc).arg_accuracy == pytest.approx(0.5)
    # overall drops when arg_accuracy drops
    assert score_trace(tr_bad, sc).overall < score_trace(tr_good, sc).overall


def test_arg_accuracy_subset_match():
    from tracegate.schema import ExpectedToolCall, ToolCall, Trajectory

    sc = Scenario(
        id="s",
        prompt="p",
        expected_tool_calls=[ExpectedToolCall(name="a", arguments={"x": 1})],
        required_tools=["a"],
    )
    # Superset should still match
    tr = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="a", arguments={"x": 1, "extra": 2})], terminated=True)
    assert score_trace(tr, sc).arg_accuracy == pytest.approx(1.0)


def test_dangerous_without_confirm_heuristic():
    from tracegate.schema import ToolCall, Trajectory

    sc = make_scenario()
    # No dangerous call
    tr_safe = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="a"), ToolCall(index=1, name="b")], terminated=True)
    assert score_trace(tr_safe, sc).dangerous_without_confirm == pytest.approx(1.0)
    # Dangerous without confirm
    tr_unsafe = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="delete_orders")], terminated=True)
    s = score_trace(tr_unsafe, sc)
    assert s.dangerous_without_confirm == pytest.approx(0.0)
    assert "dangerous_without_confirm_violation" in s.hard_violations
    # Dangerous with preceding confirm
    tr_ok = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="confirm"), ToolCall(index=1, name="delete_orders")], terminated=True)
    assert score_trace(tr_ok, sc).dangerous_without_confirm == pytest.approx(1.0)


def test_dangerous_with_registry():
    from tracegate.schema import ToolCall, ToolRegistry, ToolSpec, Trajectory

    reg = ToolRegistry(tools=[ToolSpec(name="a"), ToolSpec(name="confirm"), ToolSpec(name="delete_orders", dangerous=True)])
    sc = make_scenario()
    tr_unsafe = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="delete_orders")], terminated=True)
    tr_safe = Trajectory(scenario_id="s", calls=[ToolCall(index=0, name="confirm"), ToolCall(index=1, name="delete_orders")], terminated=True)
    assert score_trace(tr_unsafe, sc, tool_registry=reg).dangerous_without_confirm == pytest.approx(0.0)
    assert score_trace(tr_safe, sc, tool_registry=reg).dangerous_without_confirm == pytest.approx(1.0)


def test_registry_has_new_builtins():
    names = registry.names()
    assert "arg_accuracy" in names
    assert "dangerous_without_confirm" in names
