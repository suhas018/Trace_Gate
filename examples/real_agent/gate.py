"""CI-style gate against the *saved* real-agent baseline.

Mimics the merge gate: load ``baseline.json`` (captured from the GOOD agent),
run the current agent (here, the BUGGY prompt stands in for "the new model
version"), compare deterministically with multi-rollout sampling, and exit
non-zero on any regression so a CI pipeline can block the merge.

Optional: ``--judge-min-score`` enables the LLM-judge layer as a *gate*. Off by
default (the deterministic gate never calls a model). When enabled, a scenario
whose trajectory passes but whose N-shot judge mean is below the threshold is
treated as a regression — this closes the goal-refusal blind spot.

Usage:
  python demo.py --capture baseline.json                 # first: known-good baseline
  python gate.py baseline.json                            # deterministic gate only
  python gate.py baseline.json --judge-min-score 0.5      # + judge gate
  echo $LASTEXITCODE                                      # 0 = pass, 1 = regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tracegate.agents.langgraph import LangGraphAdapter
from tracegate.agents.ollama import build_ollama_tool_agent
from tracegate.io_utils import load_json, load_scenario_suite
from tracegate.judge import OllamaJudge
from tracegate.suite import SuiteReport, apply_judge, run_gate, run_suite

from prompts import BUGGY_PROMPT
from tools import cancel_order, compute_total, get_orders, refund_order

HERE = Path(__file__).parent
TOOLS = [get_orders, compute_total, cancel_order, refund_order]


def main() -> int:
    parser = argparse.ArgumentParser(description="gate the current agent against a saved baseline")
    parser.add_argument("reference", help="path to baseline.json")
    parser.add_argument("--rollouts", type=int, default=3,
                        help="samples per scenario; gate on the worst (default 3)")
    parser.add_argument("--judge-samples", type=int, default=3, help="N-shot judge samples")
    parser.add_argument("--judge-min-score", type=float, default=None,
                        help="if set, fail scenarios whose judge mean is below this (opt-in)")
    args = parser.parse_args()

    baseline = SuiteReport.model_validate(load_json(HERE / args.reference))
    scenarios, _ = load_scenario_suite(HERE / "scenarios.yaml")
    graph = build_ollama_tool_agent(TOOLS, system_prompt=BUGGY_PROMPT)
    current = LangGraphAdapter(graph)

    judge_scores = None
    if args.judge_min_score is not None:
        current_report = run_suite(scenarios, current, num_rollouts=args.rollouts)
        apply_judge(current_report, scenarios, OllamaJudge(), samples=args.judge_samples)
        judge_scores = {j.scenario_id: j for j in current_report.judges}

    report = run_gate(
        scenarios,
        current,
        baseline,
        num_rollouts=args.rollouts,
        judge_scores=judge_scores,
        judge_min_score=args.judge_min_score,
    )
    failures = 0
    for g in report.per_scenario:
        line = f"{g.scenario_id:<22} {g.decision.verdict:<10} {g.decision.reason}"
        if judge_scores and judge_scores.get(g.scenario_id):
            j = judge_scores[g.scenario_id]
            jscore = f"{j.score:.2f}" if j.score is not None else "n/a"
            line += f"  [judge {j.label} {jscore} x{j.samples}]"
        print(line)
        if not g.passed:
            failures += 1
            print(f"      overall={g.current_scores.overall:.4f} "
                  f"violations={g.current_scores.hard_violations or '-'}")
    print(f"\ngate: {'PASS' if failures == 0 else f'{failures} regression(s)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
