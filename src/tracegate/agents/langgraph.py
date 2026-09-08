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
        # Respect recursion_limit stashed by build_ollama_tool_agent (or caller-provided config)
        rec_limit = getattr(self.graph, "_tracegate_recursion_limit", None)
        if rec_limit is not None:
            result = self.graph.invoke(state, config={"recursion_limit": rec_limit})
        else:
            result = self.graph.invoke(state)

        messages = result.get("messages", [])

        def _get(msg: Any, key: str, default: Any = None) -> Any:
            if isinstance(msg, dict):
                return msg.get(key, default)
            return getattr(msg, key, default)

        # Build map tool_call_id -> args from AIMessages (P0: populate arguments)
        tool_args: dict[str, dict[str, Any]] = {}
        for msg in messages:
            tcalls = _get(msg, "tool_calls")
            if tcalls:
                for tc in tcalls:
                    if isinstance(tc, dict):
                        tid = tc.get("id") or tc.get("tool_call_id") or ""
                        args = tc.get("args") or tc.get("arguments") or {}
                        name = tc.get("name") or ""
                    else:
                        tid = getattr(tc, "id", None) or getattr(tc, "tool_call_id", None) or ""
                        args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
                        name = getattr(tc, "name", "")  # noqa: F841 — kept for debugging
                    if tid:
                        tool_args[str(tid)] = dict(args) if isinstance(args, dict) else {}

        calls: list[ToolCall] = []
        for msg in messages:
            mtype = _get(msg, "type")
            if mtype == "tool":
                name = _get(msg, "name", "<unknown>") or "<unknown>"
                # content may be str / list / dict — normalize to str for result
                raw_result = _get(msg, "content", None)
                if isinstance(raw_result, (list, dict)):
                    import json as _json

                    try:
                        raw_result = _json.dumps(raw_result)
                    except Exception:
                        raw_result = str(raw_result)
                tid = _get(msg, "tool_call_id", None) or _get(msg, "id", None) or ""
                args = tool_args.get(str(tid), {}) if tid else {}
                calls.append(
                    ToolCall(
                        index=len(calls),
                        name=str(name),
                        arguments=dict(args),
                        result=raw_result,
                    )
                )
        # The *last* model/assistant message determines whether the loop
        # terminated normally. A final AIMessage (no tool call) == done.
        final = None
        for msg in reversed(messages):
            if _get(msg, "type") in ("ai", "human"):
                final = msg
                break
        is_final_ai = final is not None and _get(final, "type") == "ai"
        final_content = _get(final, "content", None) if final is not None else None
        if isinstance(final_content, (list, dict)):
            import json as _json2

            try:
                final_content = _json2.dumps(final_content)
            except Exception:
                final_content = str(final_content)
        return Trajectory(
            scenario_id=scenario.id,
            calls=calls,
            terminated=is_final_ai,
            final_answer=str(final_content) if final_content is not None else None,
        )


__all__ = ["LangGraphAdapter"]
