import pytest

from tracegate.agent import AgentBehavior, BehaviorMappedAgent, ScriptedAgent
from tracegate.metrics import compare_to_baseline, score_trace
from tracegate.schema import Scenario, ToolRegistry, Trajectory
from tracegate.suite import run_gate, run_suite, run_suite_with_mutations


def scenario():
    return Scenario(
        id="sc",
        prompt="p",
        expected_tool_calls=[{"name": "get_orders"}, {"name": "compute_total"}],
        required_tools=["get_orders", "compute_total"],
        forbidden_tools=["delete_orders"],
        allowed_tools=["get_orders", "compute_total"],
    )


def registry():
    return ToolRegistry(
        tools=[
            {"name": "get_orders"},
            {"name": "compute_total"},
            {"name": "delete_orders", "dangerous": True},
        ]
    )


def test_compare_to_baseline_pass_and_regression():
    sc = scenario()
    good = score_trace(ScriptedAgent(registry()).run(sc), sc)
    bad = score_trace(
        ScriptedAgent(registry(), AgentBehavior(plan=["get_orders", "delete_orders"])).run(sc), sc
    )
    assert compare_to_baseline(good, good).verdict == "PASS"
    assert compare_to_baseline(bad, good).verdict == "REGRESSION"


def test_gate_passes_for_good_agent_and_fails_for_buggy():
    sc = scenario()
    reg = registry()
    good_agent = ScriptedAgent(reg)
    baseline = run_suite([sc], good_agent)

    assert run_gate([sc], good_agent, baseline).passed

    buggy = ScriptedAgent(reg, AgentBehavior(plan=["get_orders", "delete_orders"]))
    report = run_gate([sc], buggy, baseline)
    assert not report.passed
    assert len(report.failing()) == 1


def test_scenario_missing_from_baseline_fails():
    sc = scenario()
    reg = registry()
    baseline = run_suite([sc], ScriptedAgent(reg))
    sc2 = sc.model_copy(update={"id": "other"})
    report = run_gate([sc2], ScriptedAgent(reg), baseline)
    assert not report.passed


def test_behavior_mapped_agent_per_scenario():
    reg = registry()
    behaviors = {
        "sc": AgentBehavior(plan=["get_orders", "compute_total"]),
        "sc_bad": AgentBehavior(plan=["get_orders", "delete_orders"]),
    }
    agent = BehaviorMappedAgent(reg, behaviors)
    sc = scenario()
    tr_good = agent.run(sc)
    assert tr_good.tool_names() == ["get_orders", "compute_total"]
    sc_bad = sc.model_copy(update={"id": "sc_bad"})
    tr_bad = agent.run(sc_bad)
    assert tr_bad.tool_names() == ["get_orders", "delete_orders"]


def test_suite_report_roundtrip_and_mutation():
    sc = scenario()
    reg = registry()
    report = run_suite_with_mutations([sc], ScriptedAgent(reg))
    data = report.model_dump(mode="json")
    from tracegate.suite import SuiteReport

    loaded = SuiteReport.model_validate(data)
    assert loaded.summary["scenarios"] == 1
    assert loaded.mutation[0].kill_rate == pytest.approx(1.0)


def test_scores_are_deterministic():
    sc = scenario()
    reg = registry()
    a = run_suite([sc], ScriptedAgent(reg))
    b = run_suite([sc], ScriptedAgent(reg))
    assert a.model_dump() == b.model_dump()
