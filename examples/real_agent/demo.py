"""Real-agent demo: run the harness against a live llama3.1:8b LangGraph agent.

Flow:
  1. capture a baseline with the GOOD prompt (gate should pass),
  2. mutation-test the captured golden traces (kill-rate),
  3. optionally run the LLM judge for goal-satisfaction (non-gating),
  4. run the same scenarios with the BUGGY prompt (prompt drift),
  5. show the deterministic gate blocking the drift as REGRESSION.

Usage:
  python demo.py                      full showcase (in-memory baseline)
  python demo.py --capture baseline.json   save the GOOD-agent baseline
  python demo.py --judge              also run the LLM-as-judge layer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tracegate.agents.langgraph import LangGraphAdapter
from tracegate.agents.ollama import build_ollama_tool_agent
from tracegate.io_utils import load_scenario_suite, save_json
from tracegate.judge import build_judge
from tracegate.metrics import compare_to_baseline
from tracegate.suite import apply_judge, run_gate, run_suite, run_suite_with_mutations

from prompts import BUGGY_PROMPT, GOOD_PROMPT
from tools import cancel_order, compute_total, get_orders, refund_order

HERE = Path(__file__).parent
TOOLS = [get_orders, compute_total, cancel_order, refund_order]


def build_adapter(system_prompt: str) -> LangGraphAdapter:
    graph = build_ollama_tool_agent(TOOLS, system_prompt=system_prompt)
    return LangGraphAdapter(graph)


def show(trace, label: str) -> None:
    calls = " -> ".join(c.name for c in trace.calls) if trace.calls else "<no tool calls>"
    print(f"  {label:<22} calls=[{calls}] terminated={trace.terminated}")


def print_judges(report) -> None:
    if not report.judges:
        return
    print("\n  judge goal-satisfaction (non-gating):")
    for j in report.judges:
        score = f"{j.score:.2f}" if j.score is not None else "n/a"
        print(f"    {j.scenario_id:<22} {j.label:<10} score={score}  {j.rationale}")
    mean = report.summary.get("judge_satisfaction_mean")
    if mean is not None:
        print(f"    mean goal-satisfaction: {mean:.2f}")


def capture_verified(scenarios, agent, judge, retries: int, rollouts: int, judge_samples: int):
    """Roll the baseline until the deterministic gate verifies it.

    Tool-calling LLMs are stochastic even at temperature=0: a single run can
    roll a bad trajectory. We sample ``rollouts`` per scenario, gate on the
    worst, and re-roll until (a) no hard violations and (b) the mutation
    kill-rate is high enough to trust the gate. Refusing to write a bad
    baseline is a feature, not a bug.
    """
    last_kill = 0.0
    for attempt in range(1, retries + 1):
        baseline = run_suite_with_mutations(
            scenarios, agent, run_id="good@llama3.1:8b", num_rollouts=rollouts
        )
        if judge:
            apply_judge(baseline, scenarios, judge, samples=judge_samples)
        bad = [r for r in baseline.results if r.scores.hard_violations]
        kill = baseline.summary.get("mean_kill_rate", 0.0)
        last_kill = kill
        if not bad and kill >= 0.8:
            return baseline, attempt
        print(f"  attempt {attempt}: {len(bad)} violation(s), kill-rate={kill:.2f} -> re-rolling",
              file=sys.stderr)
    raise RuntimeError(
        f"could not capture a verified baseline in {retries} attempts "
        f"(last kill-rate {last_kill:.2f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", metavar="PATH", help="capture GOOD-agent baseline to PATH")
    parser.add_argument("--retries", type=int, default=3, help="max capture attempts")
    parser.add_argument("--rollouts", type=int, default=3,
                        help="samples per scenario; gate on the worst (default 3)")
    parser.add_argument("--judge-samples", type=int, default=3,
                        help="N-shot judge samples per scenario (default 3)")
    parser.add_argument("--judge", action="store_true", help="run the LLM-as-judge layer")
    parser.add_argument("--judge-provider", choices=["ollama", "api"], default="ollama",
                        help="judge backend: 'ollama' (local) or 'api' (OpenAI-compatible endpoint)")
    parser.add_argument("--judge-model", default=None, help="judge model (provider default if omitted)")
    parser.add_argument("--judge-base-url", default=None,
                        help="api judge base URL, e.g. https://api.openai.com/v1 or http://localhost:11434/v1")
    parser.add_argument("--judge-api-key", default=None,
                        help="api judge key (defaults to OPENAI_API_KEY env)")
    args = parser.parse_args()

    judge = build_judge(
        provider=args.judge_provider,
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )

    scenarios, _ = load_scenario_suite(HERE / "scenarios.yaml")
    print(f"loaded {len(scenarios)} scenarios\n")

    good = build_adapter(GOOD_PROMPT)

    if args.capture:
        print("== capture baseline (GOOD prompt) ==")
        judge_for_capture = judge if args.judge else None
        baseline, attempt = capture_verified(
            scenarios, good, judge_for_capture, args.retries, args.rollouts, args.judge_samples
        )
        save_json(baseline.model_dump(mode="json"), HERE / args.capture)
        print(f"verified on attempt {attempt}; wrote baseline -> {args.capture}")
        print(f"mean overall={baseline.summary['mean_overall']:.4f} "
              f"kill-rate={baseline.summary.get('mean_kill_rate', 0):.2f} "
              f"(rollouts={args.rollouts})")
        if args.judge:
            print_judges(baseline)
        print("\n  NOTE: deterministic gate checks the PATH (tool calls). The judge checks the")
        print("  GOAL (final answer). They can disagree - a clean path with a refusal answer is")
        print("  exactly the case the judge layer exists to catch.")
        return 0

    print("== 1. baseline (GOOD prompt) ==")
    baseline = run_suite_with_mutations(
        scenarios, good, run_id="good@llama3.1:8b", num_rollouts=args.rollouts
    )
    for r in baseline.results:
        show(r.trace, r.scenario_id)
        print(f"     overall={r.scores.overall:.4f} violations={r.scores.hard_violations or '-'}")
    print(f"  mean overall={baseline.summary['mean_overall']:.4f} "
          f"(rollouts={args.rollouts}, mean-over-rollouts="
          f"{baseline.summary['mean_over_rollouts']:.4f})")
    if baseline.mutation:
        for m in baseline.mutation:
            print(f"    kill-rate {m.scenario_id}: {m.kill_rate:.2f} ({m.killed}/{m.total})")
    if args.judge:
        apply_judge(baseline, scenarios, judge, samples=args.judge_samples)
        print_judges(baseline)

    print("\n== 2. sanity: same agent vs its own baseline (must PASS) ==")
    sanity = run_gate(scenarios, good, baseline, num_rollouts=args.rollouts)
    for g in sanity.per_scenario:
        print(f"  {g.scenario_id:<22} {g.decision.verdict:<10} {g.decision.reason}")

    print("\n== 3. prompt drift (BUGGY prompt) -> gate must REGRESS ==")
    buggy = build_adapter(BUGGY_PROMPT)
    current = run_suite(scenarios, buggy, num_rollouts=args.rollouts)
    if args.judge:
        apply_judge(current, scenarios, judge, samples=args.judge_samples)
    failures = 0
    for r in current.results:
        show(r.trace, r.scenario_id)
        decision = compare_to_baseline(r.scores, baseline.scores_for(r.scenario_id))
        print(f"     {decision.verdict:<10} {decision.reason}")
        if decision.verdict != "PASS":
            failures += 1
    if args.judge:
        print_judges(current)
    print(f"\ngate: {'PASS' if failures == 0 else f'{failures} regression(s) detected'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
