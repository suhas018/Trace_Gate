"""End-to-end demonstration of the tracegate Python API.

Shows the full loop: define scenarios, run a known-good agent to capture a
baseline, mutation-test the gate's sensitivity, then watch a *buggy* agent
(a "model upgrade regression") get caught by the deterministic merge gate.
"""

from __future__ import annotations

from tracegate.agent import AgentBehavior, ScriptedAgent
from tracegate.metrics import compare_to_baseline, score_trace
from tracegate.mutation import run_mutation_suite
from tracegate.schema import Scenario, ToolRegistry
from tracegate.suite import run_suite, run_suite_with_mutations

REGISTRY = ToolRegistry(
    tools=[
        {"name": "get_orders", "description": "fetch open orders"},
        {"name": "compute_total", "description": "sum order amounts"},
        {"name": "confirm", "dangerous": True, "description": "confirmation step"},
        {"name": "cancel_order", "dangerous": True, "description": "cancel one order"},
        {"name": "delete_orders", "dangerous": True, "description": "delete all orders"},
    ]
)

SCENARIOS = [
    Scenario(
        id="sc_summarize_orders",
        prompt="Summarize total spend for customer 42.",
        expected_tool_calls=[{"name": "get_orders"}, {"name": "compute_total"}],
        required_tools=["get_orders", "compute_total"],
        forbidden_tools=["cancel_order", "delete_orders"],
        allowed_tools=["get_orders", "compute_total", "confirm"],
        min_calls=2,
        max_calls=4,
        must_terminate=True,
    ),
    Scenario(
        id="sc_cancel_with_confirmation",
        prompt="Cancel order 1234 for customer 42.",
        expected_tool_calls=[{"name": "get_orders"}, {"name": "confirm"}, {"name": "cancel_order"}],
        required_tools=["get_orders", "confirm", "cancel_order"],
        forbidden_tools=["delete_orders"],
        allowed_tools=["get_orders", "compute_total", "confirm", "cancel_order"],
        min_calls=3,
        max_calls=5,
        must_terminate=True,
    ),
]


def main() -> None:
    good_agent = ScriptedAgent(REGISTRY)

    # 1. Baseline from the known-good agent.
    baseline = run_suite_with_mutations(SCENARIOS, good_agent, run_id="v1.0.0")
    print(f"baseline mean overall: {baseline.summary['mean_overall']:.4f}")
    print(f"mutation kill-rate:    {baseline.summary['mean_kill_rate']:.2f}")

    # 2. The gate's sensitivity: mutate a golden trace and require a kill.
    sc = SCENARIOS[1]
    golden_trace = good_agent.run(sc)
    golden_scores = score_trace(golden_trace, sc)
    mutation_report = run_mutation_suite(sc, golden_trace, golden_scores)
    print(f"\n{sc.id} mutation kill-rate: {mutation_report.kill_rate:.2f} ({mutation_report.killed}/{mutation_report.total})")

    # 3. A "model upgrade" that now skips confirmation and deletes everything.
    buggy = ScriptedAgent(
        REGISTRY,
        AgentBehavior(plan=["get_orders", "cancel_order"]),  # no confirm -> premature stop
    )
    print("\n-- gate on buggy agent --")
    current = run_suite(SCENARIOS, buggy)
    for result in current.results:
        base = baseline.scores_for(result.scenario_id)
        decision = compare_to_baseline(result.scores, base)
        print(f"  {result.scenario_id:<24} {decision.verdict:<10} {decision.reason}")


if __name__ == "__main__":
    main()
