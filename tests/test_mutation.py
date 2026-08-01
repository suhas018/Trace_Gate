import pytest

from tracegate.agent import AgentBehavior, ScriptedAgent
from tracegate.metrics import score_trace
from tracegate.mutation import apply_mutation, build_mutations, run_mutation_suite
from tracegate.schema import Scenario, ToolRegistry


def scenario():
    return Scenario(
        id="sc",
        prompt="p",
        expected_tool_calls=[{"name": "get_orders"}, {"name": "confirm"}, {"name": "cancel_order"}],
        required_tools=["get_orders", "confirm", "cancel_order"],
        forbidden_tools=["delete_orders"],
        allowed_tools=["get_orders", "compute_total", "confirm", "cancel_order"],
        min_calls=3,
        max_calls=5,
        must_terminate=True,
    )


def registry():
    return ToolRegistry(
        tools=[
            {"name": "get_orders"},
            {"name": "compute_total"},
            {"name": "confirm", "dangerous": True},
            {"name": "cancel_order", "dangerous": True},
            {"name": "delete_orders", "dangerous": True},
        ]
    )


def golden_trace():
    agent = ScriptedAgent(registry())
    return agent.run(scenario())


def test_all_mutations_are_killed():
    sc = scenario()
    trace = golden_trace()
    scores = score_trace(trace, sc)
    assert scores.overall == pytest.approx(1.0)
    result = run_mutation_suite(sc, trace, scores)
    assert result.total == 5
    assert result.killed == 5
    assert result.kill_rate == pytest.approx(1.0)
    for r in result.results:
        assert r.killed, f"{r.name} survived: {r.detail}"


def test_wrong_tool_mutation_flagged():
    sc = scenario()
    tr = apply_mutation("wrong_tool", golden_trace(), sc)
    scores = score_trace(tr, sc)
    assert scores.forbidden == 0.0


def test_premature_termination_mutation_flagged():
    sc = scenario()
    tr = apply_mutation("premature_termination", golden_trace(), sc)
    scores = score_trace(tr, sc)
    assert scores.termination == 0.0


def test_reorder_mutation_lowers_sequence():
    sc = scenario()
    tr = apply_mutation("reorder", golden_trace(), sc)
    scores = score_trace(tr, sc)
    assert scores.sequence < 1.0


def test_mutation_applicability_gating():
    sc = scenario()
    sc2 = sc.model_copy(update={"id": "tiny", "expected_tool_calls": [{"name": "get_orders"}]})
    tr = golden_trace()
    tr2 = tr.model_copy(update={"calls": tr.calls[:1], "scenario_id": "tiny"})
    names = [m.name for m in build_mutations(tr2, sc2)]
    assert "reorder" not in names  # needs >=2 calls


def test_golden_trace_derived_plan_is_ordered():
    agent = ScriptedAgent(registry())
    tr = agent.run(scenario())
    assert tr.tool_names() == ["get_orders", "confirm", "cancel_order"]
    assert tr.terminated is True


def test_behavior_override():
    agent = ScriptedAgent(registry(), behavior=AgentBehavior(plan=["get_orders", "cancel_order"]))
    tr = agent.run(scenario())
    assert tr.tool_names() == ["get_orders", "cancel_order"]
