"""Real-agent factory: a LangGraph tool-calling agent running on Ollama.

Builds a standard ReAct-style loop (model -> tool -> ... -> final answer)
with the same ``messages``-keyed state that :class:`LangGraphAdapter` expects,
so real trajectories flow through the harness unchanged.

Requires: ``pip install tracegate[langgraph]`` plus ``langchain-ollama`` and a
pulled Ollama model.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful customer-support agent. "
    "You may only use the provided tools; never do calculations or look up data "
    "yourself when a tool exists for it. Always gather the data you need with a "
    "tool call before acting on it."
)


def build_ollama_tool_agent(
    tools: list[BaseTool],
    model: str = "llama3.1:8b",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    recursion_limit: int = 20,
    num_predict: int = 512,
    seed: int = 42,
):
    """Compile a LangGraph tool-calling agent backed by an Ollama model.

    ``num_predict`` caps per-step generation. llama3.1 (and several other
    Ollama models) sometimes ignores the tool-call contract and starts a
    runaway generation that never terminates; the cap truncates it so the
    harness still captures and scores whatever trajectory was produced.

    ``seed`` pins the sampler: without it, even ``temperature=0`` produces
    different trajectories across runs, which makes single-shot baselines
    untrustworthy.
    """
    try:
        from langgraph.graph import END, MessagesState, START, StateGraph
        from langgraph.prebuilt import ToolNode
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install langgraph: pip install tracegate[langgraph]") from exc

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        seed=seed,
    ).bind_tools(tools)

    def call_model(state):
        return {"messages": [llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])]}

    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


__all__ = ["DEFAULT_SYSTEM_PROMPT", "build_ollama_tool_agent"]
