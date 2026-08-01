"""Single-scenario execution: agent -> trace -> deterministic scores."""

from __future__ import annotations

from pydantic import BaseModel

from tracegate.agent import AgentAdapter
from tracegate.metrics import ComponentScores, score_trace
from tracegate.schema import Scenario, Trajectory


class RunResult(BaseModel):
    scenario_id: str
    trace: Trajectory
    scores: ComponentScores


def run_scenario(
    scenario: Scenario,
    agent: AgentAdapter,
    weights: dict[str, float] | None = None,
) -> RunResult:
    """Run one scenario through an agent adapter and score the resulting trace."""
    trace = agent.run(scenario)
    scores = score_trace(trace, scenario, weights=weights)
    return RunResult(scenario_id=scenario.id, trace=trace, scores=scores)


__all__ = ["RunResult", "run_scenario"]
