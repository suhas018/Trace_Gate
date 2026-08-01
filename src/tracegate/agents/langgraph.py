"""Optional LangGraph adapter.

Requires the optional dependency: ``pip install tracegate[langgraph]``.

This adapter instruments a LangGraph ``StateGraph``-built agent and normalizes
its tool-call history into a :class:`~tracegate.schema.Trajectory`, so the
deterministic metrics, mutation testing, and the CI gate all run against a real
LangGraph agent without any framework-specific scoring code.

The module imports LangGraph lazily so the core harness never depends on it.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from tracegate.schema import Scenario, ToolCall, ToolRegistry, Trajectory

_LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


class LangGraphAdapter:
    """Adapts a LangGraph agent (a compiled ``CompiledStateGraph``) to the harness.

    The agent graph must route tool calls through a node whose state key
    ``messages`` contains ``ToolMessage`` entries (the standard LangGraph tool
    pattern). Each ``ToolMessage`` becomes a :class:`ToolCall`.
    """

    def __init__(self, graph: Any, registry: ToolRegistry | None = None):
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraphAdapter requires langgraph; install with `pip install tracegate[langgraph]`"
            )
        self.graph = graph
        self.registry = registry

    def run(self, scenario: Scenario, initial_state: dict[str, Any] | None = None) -> Trajectory:
        state = dict(initial_state or {})
        state.setdefault("messages", [{"role": "user", "content": scenario.prompt}])
        result = self.graph.invoke(state)

        messages = result.get("messages", [])
        calls: list[ToolCall] = []
        for msg in messages:
            if getattr(msg, "type", None) == "tool":
                calls.append(
                    ToolCall(
                        index=len(calls),
                        name=getattr(msg, "name", "<unknown>"),
                        arguments={},
                        result=getattr(msg, "content", None),
                    )
                )
        # The *last* model/assistant message determines whether the loop
        # terminated normally. A final AIMessage (no tool call) == done.
        final = None
        for msg in reversed(messages):
            if getattr(msg, "type", None) in ("ai", "human"):
                final = msg
                break
        is_final_ai = final is not None and getattr(final, "type", None) == "ai"
        return Trajectory(
            scenario_id=scenario.id,
            calls=calls,
            terminated=is_final_ai,
            final_answer=str(getattr(final, "content", None)) if final is not None else None,
        )


__all__ = ["LangGraphAdapter"]
