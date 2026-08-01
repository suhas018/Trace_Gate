import pytest

from tracegate.metrics import score_trace
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
