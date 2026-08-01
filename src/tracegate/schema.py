"""Core data model: scenarios, trajectories, and tool registries.

The types in this module form the *framework-independent trace schema*. A
``Trajectory`` is a normalized record of what an agent actually did (ordered
tool calls plus a termination flag) and a ``Scenario`` is a specification of
what it *should* have done. Nothing here depends on LangGraph, CrewAI, or any
other orchestration library.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class TraceGateError(Exception):
    """Raised when a scenario or trace violates structural invariants."""


class ToolSpec(BaseModel):
    """A tool an agent may call."""

    name: str
    description: str = ""
    dangerous: bool = Field(default=False, description="Requires confirmation / high blast radius")


class ToolRegistry(BaseModel):
    """The universe of tools a scenario's agent may invoke."""

    tools: list[ToolSpec] = Field(default_factory=list)

    @property
    def names(self) -> set[str]:
        return {t.name for t in self.tools}

    def has(self, name: str) -> bool:
        return name in self.names

    def spec(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def is_dangerous(self, name: str) -> bool:
        spec = self.spec(name)
        return bool(spec and spec.dangerous)

    def get(self, name: str) -> ToolSpec:
        spec = self.spec(name)
        if spec is None:
            raise TraceGateError(f"unknown tool in registry: {name!r}")
        return spec


class ToolCall(BaseModel):
    """One tool invocation in a trajectory."""

    index: int = Field(ge=0, description="Position of the call in the trajectory (0-based)")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class Trajectory(BaseModel):
    """Normalized record of what an agent actually did in one run."""

    scenario_id: str
    calls: list[ToolCall] = Field(default_factory=list)
    terminated: bool = Field(
        default=True, description="Whether the agent emitted a final answer / reached a terminal state"
    )
    final_answer: str | None = None

    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def tool_called(self, name: str) -> bool:
        return any(c.name == name for c in self.calls)

    def first_call_of(self, name: str) -> ToolCall | None:
        for c in self.calls:
            if c.name == name:
                return c
        return None


class ExpectedToolCall(BaseModel):
    """An expected step in the ideal trajectory.

    Order in the scenario's ``expected_tool_calls`` list is significant. Each
    entry can pin specific argument values; ``arguments=None`` means "any args".
    """

    name: str
    arguments: dict[str, Any] | None = None


class Scenario(BaseModel):
    """Specification of an ideal trajectory for one agent task."""

    id: str
    prompt: str
    description: str = ""

    expected_tool_calls: list[ExpectedToolCall] = Field(
        default_factory=list,
        description="Ideal ordered tool-call sequence (used by the order-sensitive sequence score)",
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools that must be called at least once, regardless of order",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Tools that must never be called. Any call is a hard gate violation.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Full set of valid tool names. Calls outside this set count as forbidden "
        "(unknown-tool violations). None disables the check.",
    )

    min_calls: int = Field(default=1, ge=0)
    max_calls: int | None = Field(default=None, ge=1, description="None means unlimited")
    must_terminate: bool = Field(default=True)

    @model_validator(mode="after")
    def _check_invariants(self) -> "Scenario":
        dupes = {t for t in self.required_tools if self.required_tools.count(t) > 1}
        if dupes:
            raise TraceGateError(f"scenario {self.id!r}: duplicate required_tools: {sorted(dupes)}")
        overlap = set(self.required_tools) & set(self.forbidden_tools)
        if overlap:
            raise TraceGateError(
                f"scenario {self.id!r}: tool in both required and forbidden: {sorted(overlap)}"
            )
        if self.min_calls > 0 and not self.required_tools and not self.expected_tool_calls:
            raise TraceGateError(
                f"scenario {self.id!r}: min_calls>0 but no expected/required tools to satisfy it"
            )
        expected_names = {e.name for e in self.expected_tool_calls}
        for r in self.required_tools:
            if r not in expected_names:
                # Allowed but unusual: requirement not part of the ideal sequence.
                pass
        return self

    def expected_sequence(self) -> list[str]:
        return [e.name for e in self.expected_tool_calls]

    def is_forbidden(self, name: str) -> bool:
        if name in self.forbidden_tools:
            return True
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return True
        return False
