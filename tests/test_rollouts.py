"""Tests for multi-rollout gating (nondeterminism fix) and judge gating."""

import pytest

from tracegate.agent import AgentBehavior, ScriptedAgent
from tracegate.judge import SampledJudgeResult, sample_judge
from tracegate.metrics import score_trace
from tracegate.schema import Scenario, ToolRegistry
from tracegate.suite import run_gate, run_suite


def scenario():
    return Scenario(
        id="sc",
        prompt="p",
        expected_tool_calls=[{"name": "get_orders"}, {"name": "compute_total"}],
        required_tools=["get_orders", "compute_total"],
        forbidden_tools=["delete_orders"],
        allowed_tools=["get_orders", "compute_total", "delete_orders"],
        min_calls=2,
        max_calls=4,
        must_terminate=True,
    )


def registry():
    return ToolRegistry(
        tools=[
            {"name": "get_orders"},
            {"name": "compute_total"},
            {"name": "delete_orders", "dangerous": True},
        ]
    )


class FlakyAgent:
    """Alternates between a clean plan and a buggy plan — a stochastic agent."""

    def __init__(self, registry, period=2):
        self.registry = registry
        self.period = period
        self.count = 0

    def run(self, scenario):
        clean = self.count == 0
        self.count = (self.count + 1) % self.period
        plan = (
            ["get_orders", "compute_total"]
            if clean
            else ["get_orders", "delete_orders"]  # forbidden tool
        )
        return ScriptedAgent(self.registry, AgentBehavior(plan=plan)).run(scenario)


def test_single_rollout_can_miss_a_bad_roll():
    sc = scenario()
    agent = FlakyAgent(registry())
    # First roll is clean; a single-sample gate would pass it.
    scores = score_trace(agent.run(sc), sc)
    assert scores.hard_violations == []


def test_multi_rollout_gates_on_worst():
    sc = scenario()
    agent = FlakyAgent(registry())
    report = run_suite([sc], agent, num_rollouts=3)
    assert len(report.rollouts["sc"]) == 3
    assert report.results[0].scores.hard_violations  # worst roll was buggy
    # The gating representative is the worst roll, which is below the mean.
    assert report.results[0].scores.overall < report.summary["mean_over_rollouts"]


def test_gate_catches_flaky_agent_with_rollouts():
    sc = scenario()
    reg = registry()
    good = ScriptedAgent(reg)
    baseline = run_suite([sc], good, num_rollouts=3)

    flaky = FlakyAgent(reg)
    # num_rollouts=1 with a lucky first clean roll -> false PASS.
    assert run_gate([sc], flaky, baseline, num_rollouts=1).passed
    # num_rollouts=3 -> the bad roll is seen -> REGRESSION.
    report = run_gate([sc], flaky, baseline, num_rollouts=3)
    assert not report.passed


class CyclingJudge:
    """Returns scores that vary per call (0.9, 0.5, 0.7)."""

    def __init__(self):
        self.calls = 0

    def judge(self, scenario, trace):
        scores = [0.9, 0.5, 0.7]
        value = scores[self.calls % len(scores)]
        self.calls += 1
        from tracegate.judge import JudgeResult, label_for

        return JudgeResult(
            scenario_id=scenario.id, score=value, label=label_for(value), judge_model="fake"
        )


def test_sample_judge_aggregates():
    sc = scenario()
    trace = ScriptedAgent(registry()).run(sc)
    result = sample_judge(sc, trace, CyclingJudge(), n=3)
    assert result.samples == 3
    assert result.score == pytest.approx(0.7)  # mean(0.9, 0.5, 0.7)
    assert result.score_min == pytest.approx(0.5)
    assert result.score_max == pytest.approx(0.9)
    assert result.label == "PARTIAL"  # label_for(0.7)
    assert result.agreement == pytest.approx(2 / 3)  # majority label PARTIAL


def test_opt_in_judge_gate_blocks_low_satisfaction():
    sc = scenario()
    reg = registry()
    good = ScriptedAgent(reg)
    baseline = run_suite([sc], good)

    judge_scores = {
        "sc": SampledJudgeResult(
            scenario_id="sc", score=0.4, label="FAILED", samples=3, agreement=1.0, judge_model="fake"
        )
    }
    # Deterministic trajectory is clean, so without judge-gating it passes.
    assert run_gate([sc], good, baseline).passed
    # With judge-gating enabled, the goal-refusal is caught.
    report = run_gate([sc], good, baseline, judge_scores=judge_scores, judge_min_score=0.6)
    assert not report.passed
    assert "goal_not_satisfied" in report.failing()[0].decision.new_violations


def test_judge_gate_fails_closed_when_judge_unavailable():
    sc = scenario()
    reg = registry()
    good = ScriptedAgent(reg)
    baseline = run_suite([sc], good)
    judge_scores = {
        "sc": SampledJudgeResult(scenario_id="sc", score=None, label="UNAVAILABLE", samples=0)
    }
    report = run_gate([sc], good, baseline, judge_scores=judge_scores, judge_min_score=0.6)
    assert not report.passed
    assert "judge_unavailable" in report.failing()[0].decision.new_violations
