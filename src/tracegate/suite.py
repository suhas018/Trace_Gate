"""Suite execution, baseline comparison, and the merge gate.

A *baseline* is a serialized :class:`SuiteReport` captured when the agent was
known-good. The gate re-runs the suite against the current agent and compares
each scenario to the baseline deterministically (no LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from tracegate.agent import AgentAdapter
from tracegate.judge import LLMJudge, SampledJudgeResult, sample_judge
from tracegate.metrics import ComponentScores, GateDecision, compare_to_baseline
from tracegate.mutation import MutationSuiteResult, run_mutation_suite
from tracegate.runner import RunResult, run_scenario
from tracegate.schema import Scenario, Trajectory


class SuiteReport(BaseModel):
    """Snapshot of one full suite run. This is the on-disk baseline format.

    With ``num_rollouts > 1``, ``results`` holds the *worst* rollout per
    scenario (the representative used for gating) and ``rollouts`` keeps every
    rollout's scores so the spread is visible.
    """

    run_id: str = ""
    results: list[RunResult] = field(default_factory=list)  # type: ignore[assignment]
    mutation: list[MutationSuiteResult] = field(default_factory=list)
    judges: list[SampledJudgeResult] = field(default_factory=list)
    rollouts: dict[str, list[ComponentScores]] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)

    def scores_for(self, scenario_id: str) -> ComponentScores | None:
        for r in self.results:
            if r.scenario_id == scenario_id:
                return r.scores
        return None

    def trace_for(self, scenario_id: str) -> Trajectory | None:
        for r in self.results:
            if r.scenario_id == scenario_id:
                return r.trace
        return None


def _worst(runs: list[RunResult]) -> RunResult:
    """Representative rollout: lowest overall score, violations break ties."""
    return min(runs, key=lambda r: (r.scores.overall, -len(r.scores.hard_violations)))


def run_suite(
    scenarios: list[Scenario],
    agent: AgentAdapter,
    weights: dict[str, float] | None = None,
    run_id: str = "",
    num_rollouts: int = 1,
) -> SuiteReport:
    """Run every scenario through the agent and score each trajectory.

    With ``num_rollouts > 1`` the agent is sampled multiple times per scenario
    and the *worst* trajectory is kept as the gating representative. This makes
    the gate robust to stochastic agents: a single bad roll (llama3.1 at
    temperature 0 still varies) can no longer hide behind one lucky sample.
    """
    results: list[RunResult] = []
    rollouts: dict[str, list[ComponentScores]] = {}
    for scenario in scenarios:
        runs = [run_scenario(scenario, agent, weights=weights) for _ in range(max(1, num_rollouts))]
        rollouts[scenario.id] = [r.scores for r in runs]
        results.append(_worst(runs))
    overalls = [r.scores.overall for r in results]
    mean_overall = sum(overalls) / len(overalls) if overalls else 0.0
    rollout_means = [
        sum(s.overall for s in scores_list) / len(scores_list)
        for scores_list in rollouts.values()
    ]
    violations = sum(1 for r in results if r.scores.hard_violations)
    summary = {
        "scenarios": len(results),
        "mean_overall": mean_overall,
        "mean_over_rollouts": (sum(rollout_means) / len(rollout_means)) if rollout_means else 0.0,
        "min_overall": min(overalls) if overalls else 0.0,
        "max_overall": max(overalls) if overalls else 0.0,
        "scenarios_with_violations": violations,
    }
    return SuiteReport(run_id=run_id, results=results, rollouts=rollouts, summary=summary)


def run_suite_with_mutations(
    scenarios: list[Scenario],
    agent: AgentAdapter,
    weights: dict[str, float] | None = None,
    run_id: str = "",
    seed: int = 42,
    num_rollouts: int = 1,
) -> SuiteReport:
    """Run the suite and, for each scenario, mutation-test the golden trace."""
    report = run_suite(scenarios, agent, weights=weights, run_id=run_id, num_rollouts=num_rollouts)
    mutations: list[MutationSuiteResult] = []
    for result in report.results:
        mutations.append(
            run_mutation_suite(
                _scenario_by_id(scenarios, result.scenario_id),
                result.trace,
                result.scores,
                weights=weights,
                seed=seed,
            )
        )
    report.mutation = mutations
    if mutations:
        report.summary["mean_kill_rate"] = (
            sum(m.kill_rate for m in mutations) / len(mutations)
        )
    return report


def _scenario_by_id(scenarios: list[Scenario], scenario_id: str) -> Scenario:
    for s in scenarios:
        if s.id == scenario_id:
            return s
    raise KeyError(f"scenario not in suite: {scenario_id!r}")


def apply_judge(
    report: SuiteReport,
    scenarios: list[Scenario],
    judge: LLMJudge,
    samples: int = 3,
) -> SuiteReport:
    """Attach N-shot LLM-judge goal-satisfaction scores to an existing report.

    Informational unless the caller opts into judge-gating (the deterministic
    gate never calls a model). Each scenario is judged ``samples`` times and the
    mean/min/max/agreement are stored; judge failures degrade to UNAVAILABLE.
    """
    traces = {r.scenario_id: r.trace for r in report.results}
    judged: list[SampledJudgeResult] = []
    for scenario in scenarios:
        trace = traces.get(scenario.id)
        if trace is None:
            judged.append(
                SampledJudgeResult(
                    scenario_id=scenario.id, label="UNAVAILABLE", rationales=["no trace"]
                )
            )
            continue
        judged.append(sample_judge(scenario, trace, judge, n=samples))
    report.judges = judged
    scored = [j.score for j in judged if j.score is not None]
    report.summary["judge_satisfaction_mean"] = (
        sum(scored) / len(scored) if scored else 0.0
    )
    report.summary["judge_samples"] = float(samples)
    return report


@dataclass
class GateResult:
    """Per-scenario gate comparison plus a suite-wide verdict."""

    scenario_id: str
    decision: GateDecision
    trace: Trajectory
    current_scores: ComponentScores

    @property
    def passed(self) -> bool:
        return self.decision.verdict == "PASS"


@dataclass
class GateReport:
    per_scenario: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.per_scenario)

    def failing(self) -> list[GateResult]:
        return [g for g in self.per_scenario if not g.passed]


def run_gate(
    scenarios: list[Scenario],
    agent: AgentAdapter,
    baseline: SuiteReport,
    weights: dict[str, float] | None = None,
    delta: float = 1e-9,
    num_rollouts: int = 1,
    judge_scores: dict[str, SampledJudgeResult] | None = None,
    judge_min_score: float | None = None,
) -> GateReport:
    """Compare a fresh suite run against a stored baseline and produce verdicts.

    ``num_rollouts`` samples each scenario N times and gates on the worst
    trajectory (see :func:`run_suite`), so one bad stochastic roll cannot slip
    through.

    ``judge_min_score`` is the *opt-in* judge gate. It is off by default: the
    deterministic gate never calls a model. When enabled, a scenario whose
    trajectory passes but whose N-shot judge mean is below the threshold is
    downgraded to REGRESSION — this is how the goal-refusal blind spot gets
    closed. Fail-closed: an UNAVAILABLE judge with gating enabled fails.
    """
    current = run_suite(scenarios, agent, weights=weights, num_rollouts=num_rollouts)
    per_scenario: list[GateResult] = []
    for scenario in scenarios:
        base = baseline.scores_for(scenario.id)
        cur = current.scores_for(scenario.id)
        if base is None:
            per_scenario.append(
                GateResult(
                    scenario_id=scenario.id,
                    decision=GateDecision(
                        verdict="REGRESSION",
                        reason=f"scenario {scenario.id!r} not present in baseline",
                        baseline_overall=None,
                        current_overall=cur.overall if cur else 0.0,
                    ),
                    trace=current.trace_for(scenario.id),  # type: ignore[arg-type]
                    current_scores=cur,  # type: ignore[arg-type]
                )
            )
            continue
        decision = compare_to_baseline(cur, base, delta=delta)
        if judge_scores and judge_min_score is not None:
            j = judge_scores.get(scenario.id)
            if j is not None:
                if j.score is None:
                    decision = GateDecision(
                        verdict="REGRESSION",
                        reason="judge-gating enabled but judge unavailable (fail closed)",
                        baseline_overall=base.overall,
                        current_overall=cur.overall,
                        new_violations=[*decision.new_violations, "judge_unavailable"],
                    )
                elif j.score < judge_min_score:
                    decision = GateDecision(
                        verdict="REGRESSION",
                        reason=f"goal satisfaction {j.score:.2f} < {judge_min_score:.2f}",
                        baseline_overall=base.overall,
                        current_overall=cur.overall,
                        new_violations=[*decision.new_violations, "goal_not_satisfied"],
                    )
        per_scenario.append(
            GateResult(
                scenario_id=scenario.id,
                decision=decision,
                trace=current.trace_for(scenario.id),  # type: ignore[arg-type]
                current_scores=cur,  # type: ignore[arg-type]
            )
        )
    return GateReport(per_scenario=per_scenario)


__all__ = [
    "GateReport",
    "GateResult",
    "SuiteReport",
    "apply_judge",
    "run_gate",
    "run_suite",
    "run_suite_with_mutations",
]
