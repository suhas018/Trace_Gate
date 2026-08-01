import pytest

from tracegate.schema import Scenario, ToolCall, ToolRegistry, TraceGateError, Trajectory


def test_scenario_required_and_forbidden_overlap_raises():
    with pytest.raises(TraceGateError):
        Scenario(
            id="x",
            prompt="p",
            required_tools=["a"],
            forbidden_tools=["a"],
        )


def test_scenario_duplicate_required_raises():
    with pytest.raises(TraceGateError):
        Scenario(id="x", prompt="p", required_tools=["a", "a"])


def test_scenario_min_calls_without_tools_raises():
    with pytest.raises(TraceGateError):
        Scenario(id="x", prompt="p", min_calls=1)


def test_registry_get_unknown_raises():
    reg = ToolRegistry(tools=[{"name": "a"}])
    assert reg.has("a")
    with pytest.raises(TraceGateError):
        reg.get("nope")


def test_is_forbidden_allowed_tools():
    s = Scenario(
        id="x",
        prompt="p",
        required_tools=["a"],
        allowed_tools=["a", "b"],
        forbidden_tools=["z"],
    )
    assert s.is_forbidden("z")
    assert s.is_forbidden("c")  # not in allowed_tools
    assert not s.is_forbidden("a")
    assert not s.is_forbidden("b")


def test_is_forbidden_without_allowed_tools():
    s = Scenario(id="x", prompt="p", required_tools=["a"], forbidden_tools=["z"])
    assert s.is_forbidden("z")
    assert not s.is_forbidden("anything_else")  # no whitelist -> only explicit forbids


def test_trajectory_helpers():
    t = Trajectory(
        scenario_id="x",
        calls=[
            ToolCall(index=0, name="a"),
            ToolCall(index=1, name="b"),
        ],
    )
    assert t.tool_names() == ["a", "b"]
    assert t.tool_called("a")
    assert not t.tool_called("z")
    assert t.first_call_of("b").index == 1
