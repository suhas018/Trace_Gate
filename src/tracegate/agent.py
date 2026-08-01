"""Agent adapters.

The harness is framework-agnostic: any adapter that implements
:class:`AgentAdapter` can be scored. This module ships a deterministic,
fully-controllable :class:`ScriptedAgent` — the reference adapter used for
tests, demos, and mutation validation. A ``langgraph`` adapter lives in
``tracegate.agents.langgraph`` and requires the optional dependency.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tracegate.schema import Scenario, ToolCall, ToolRegistry, Trajectory


@runtime_checkable
class AgentAdapter(Protocol):
    """Anything with a ``run(scenario) -> Trajectory`` is an adapter."""

    def run(self, scenario: Scenario) -> Trajectory:
        ...


class AgentBehavior(BaseModel):
    """Deterministic behavior override for the ScriptedAgent.

    ``plan``          : ordered tool names to call. None falls back to a plan
                        derived from the scenario (expected sequence + required).
    ``arguments``     : per-tool argument overrides (name -> dict).
    ``terminate``     : whether the agent emits a final answer at the end.
    ``max_steps``     : hard cap on tool calls; truncating the plan implies the
                        agent stopped early (useful for simulating regressions).
    """

    plan: list[str] | None = None
    arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    terminate: bool = True
    max_steps: int | None = None


def derive_plan(scenario: Scenario) -> list[str]:
    """The ideal plan implied by a scenario: expected sequence + required extras."""
    plan: list[str] = []
    for e in scenario.expected_tool_calls:
        if e.name not in plan:
            plan.append(e.name)
    for r in scenario.required_tools:
        if r not in plan:
            plan.append(r)
    return plan


class ScriptedAgent:
    """A deterministic fake "model" that executes a plan of tool calls.

    It stands in for an LLM's planning step so the harness can be exercised
    without a model or network. Pointing it at a *buggy* plan (wrong tool,
    early stop) is how the demo shows the gate catching a regression.
    """

    def __init__(self, registry: ToolRegistry, behavior: AgentBehavior | None = None):
        self.registry = registry
        self.behavior = behavior or AgentBehavior()

    def _plan_for(self, scenario: Scenario) -> list[str]:
        if self.behavior.plan is not None:
            return self.behavior.plan
        return derive_plan(scenario)

    def run(self, scenario: Scenario, behavior: AgentBehavior | None = None) -> Trajectory:
        active = behavior or self.behavior
        original = self._plan_for(scenario)
        plan = original
        if active.max_steps is not None:
            plan = original[: active.max_steps]
        calls: list[ToolCall] = []
        for i, name in enumerate(plan):
            if self.registry.has(name):
                result = f"result:{name}"
            else:
                result = None
            calls.append(
                ToolCall(
                    index=i,
                    name=name,
                    arguments=active.arguments.get(name, {}),
                    result=result,
                )
            )
        truncated = active.max_steps is not None and len(original) > active.max_steps
        terminated = active.terminate and not truncated
        return Trajectory(
            scenario_id=scenario.id,
            calls=calls,
            terminated=terminated,
            final_answer=f"answer:{scenario.id}" if terminated else None,
        )


class BehaviorMappedAgent:
    """Wraps a ScriptedAgent, selecting a behavior per scenario id.

    This is the adapter the CLI uses when a behavior config maps individual
    scenarios to (possibly buggy) plans — the mechanism for demonstrating that
    the gate catches a regression in agent behavior.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        behaviors: dict[str, AgentBehavior] | None = None,
    ):
        self.registry = registry
        self.behaviors = behaviors or {}

    def run(self, scenario: Scenario) -> Trajectory:
        behavior = self.behaviors.get(scenario.id)
        return ScriptedAgent(self.registry, behavior or AgentBehavior()).run(scenario)


__all__ = [
    "AgentAdapter",
    "AgentBehavior",
    "BehaviorMappedAgent",
    "ScriptedAgent",
    "derive_plan",
]
