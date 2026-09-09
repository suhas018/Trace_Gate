"""Command-line interface: baseline / run / gate / mutate.

All commands are deterministic and offline (no model, no network) when used
with the ScriptedAgent. ``gate`` is the CI entrypoint: it exits non-zero on any
regression so it can block a merge in GitHub Actions.
"""

from __future__ import annotations

import argparse
import sys

from tracegate.agent import AgentBehavior, BehaviorMappedAgent, ScriptedAgent
from tracegate.io_utils import load_behavior_config, load_json, load_scenario_suite, save_json
from tracegate.metrics import ComponentScores
from tracegate.suite import (
    SuiteReport,
    apply_judge,
    run_gate,
    run_suite,
    run_suite_with_mutations,
)

MAX_COLS = 14


def _build_agent(tools_registry, behaviors: dict[str, AgentBehavior], per_scenario: bool):
    from tracegate.schema import ToolRegistry

    registry = tools_registry or ToolRegistry()
    if per_scenario:
        return BehaviorMappedAgent(registry, behaviors)
    behavior = next(iter(behaviors.values())) if behaviors else None
    return ScriptedAgent(registry, behavior)


def _fmt(x: float, width: int = 8) -> str:
    return f"{x:.4f}".rjust(width)


def _print_scores(scores: ComponentScores) -> None:
    print(f"{'overall':<10}{'seq':>7}{'cov':>7}{'forb':>7}{'term':>7}{'len':>7}")
    print(
        f"{_fmt(scores.overall):<10}"
        f"{_fmt(scores.sequence):>7}"
        f"{_fmt(scores.required_coverage):>7}"
        f"{_fmt(scores.forbidden):>7}"
        f"{_fmt(scores.termination):>7}"
        f"{_fmt(scores.length):>7}"
    )
    if scores.hard_violations:
        print(f"  hard violations: {', '.join(scores.hard_violations)}")


def _print_report(report: SuiteReport) -> None:
    print(f"run: {report.run_id or '(untitled)'}")
    print(f"{'scenario':<{MAX_COLS}}{'overall':>8}  violations")
    for r in report.results:
        vio = ",".join(r.scores.hard_violations) if r.scores.hard_violations else "-"
        print(f"{r.scenario_id:<{MAX_COLS}}{_fmt(r.scores.overall):>8}  {vio}")
    s = report.summary
    print(
        f"\nsummary: mean={s.get('mean_overall', 0):.4f} "
        f"min={s.get('min_overall', 0):.4f} "
        f"scenarios_with_violations={s.get('scenarios_with_violations', 0)}"
    )
    if report.mutation:
        _print_mutations(report)


def _print_mutations(report: SuiteReport) -> None:
    print("\nmutation kill-rate:")
    for m in report.mutation:
        print(f"  {m.scenario_id:<{MAX_COLS}} kill-rate={m.kill_rate:.2f} ({m.killed}/{m.total})")
        for r in m.results:
            mark = "KILLED" if r.killed else "SURVIVED"
            print(f"      {r.name:<24} {mark:<9} {r.detail}")
    mean = report.summary.get("mean_kill_rate")
    if mean is not None:
        print(f"  mean kill-rate: {mean:.2f}")


def _run_report(scenarios, registry, behaviors, behavior_path, mutation, seed, run_id, rollouts):
    per_scenario = behavior_path is not None
    agent = _build_agent(registry, behaviors, per_scenario)
    if mutation:
        return run_suite_with_mutations(
            scenarios, agent, run_id=run_id, seed=seed, num_rollouts=rollouts, tool_registry=registry
        )
    return run_suite(scenarios, agent, run_id=run_id, num_rollouts=rollouts, tool_registry=registry)


def cmd_run(args) -> int:
    scenarios, registry = load_scenario_suite(args.suite)
    behaviors = load_behavior_config(args.behavior)
    report = _run_report(
        scenarios, registry, behaviors, args.behavior, args.mutation, args.seed, args.run_id,
        args.rollouts,
    )
    _print_report(report)
    if args.out:
        save_json(report.model_dump(mode="json"), args.out)
        print(f"\nwrote {args.out}")
    return 0


def _capture_verified(scenarios, agent, args, out: str) -> int:
    """Verify-before-write: roll the baseline until it is trustworthy.

    A baseline is only written once (a) no scenario produced hard violations
    and (b) the mutation kill-rate is at least ``--verify-kill-rate``. Weak
    baselines are refused, not persisted — the CI equivalent of the demo's
    capture loop.
    """
    # Need registry for metrics that require it (dangerous_without_confirm)
    # We reload it from args.suite; fallback to agent's registry if needed.
    try:
        _, _reg = load_scenario_suite(args.suite)
    except Exception:
        _reg = getattr(agent, "registry", None)
    last_kill = 0.0
    for attempt in range(1, args.retries + 1):
        # Vary seed per attempt so re-rolls produce different mutants (P1 fix)
        attempt_seed = args.seed + attempt - 1
        report = run_suite_with_mutations(
            scenarios, agent, run_id=args.run_id, seed=attempt_seed, num_rollouts=args.rollouts, tool_registry=_reg
        )
        bad = [r for r in report.results if r.scores.hard_violations]
        kill = report.summary.get("mean_kill_rate", 0.0)
        last_kill = kill
        if not bad and kill >= args.verify_kill_rate:
            save_json(report.model_dump(mode="json"), out)
            print(f"verified on attempt {attempt}; wrote baseline -> {out}")
            print(
                f"mean overall={report.summary['mean_overall']:.4f} "
                f"kill-rate={kill:.2f} (rollouts={args.rollouts})"
            )
            _print_report(report)
            return 0
        print(
            f"  attempt {attempt}: {len(bad)} violation(s), kill-rate={kill:.2f} -> re-rolling",
            file=sys.stderr,
        )
    raise RuntimeError(
        f"could not capture a verified baseline in {args.retries} attempts "
        f"(last kill-rate {last_kill:.2f}, required {args.verify_kill_rate:.2f})"
    )


def cmd_baseline(args) -> int:
    scenarios, registry = load_scenario_suite(args.suite)
    behaviors = load_behavior_config(args.behavior)
    out = args.out or "baseline.json"
    if args.verify_kill_rate is not None:
        agent = _build_agent(registry, behaviors, args.behavior is not None)
        return _capture_verified(scenarios, agent, args, out)
    report = _run_report(
        scenarios, registry, behaviors, args.behavior, args.mutation, args.seed, args.run_id,
        args.rollouts,
    )
    save_json(report.model_dump(mode="json"), out)
    print(f"baseline written to {out}")
    _print_report(report)
    return 0


def cmd_gate(args) -> int:
    scenarios, registry = load_scenario_suite(args.suite)
    behaviors = load_behavior_config(args.behavior)
    per_scenario = args.behavior is not None
    agent = _build_agent(registry, behaviors, per_scenario)
    baseline = SuiteReport.model_validate(load_json(args.reference))
    # Optional judge gating (P1: wire judge to CLI)
    judge_scores = None
    if getattr(args, "judge_min_score", None) is not None:
        from tracegate.judge import build_judge

        judge = build_judge(
            provider=getattr(args, "judge_provider", "ollama") or "ollama",
            model=getattr(args, "judge_model", None),
            base_url=getattr(args, "judge_base_url", None),
            api_key=getattr(args, "judge_api_key", None),
            temperature=getattr(args, "judge_temperature", 0.0) or 0.0,
        )
        # Need a pre-run to score traces before gating
        tmp_report = run_suite(scenarios, agent, num_rollouts=args.rollouts, tool_registry=registry)
        tmp_report = apply_judge(tmp_report, scenarios, judge, samples=getattr(args, "judge_samples", 3) or 3)
        judge_scores = {j.scenario_id: j for j in tmp_report.judges}
        print(f"judge: provider={getattr(args, 'judge_provider', 'ollama')} min_score={args.judge_min_score} samples={getattr(args, 'judge_samples', 3)}")
        for j in tmp_report.judges:
            print(f"  {j.scenario_id:<{MAX_COLS}} {j.label:<10} {j.score}")

    report = run_gate(
        scenarios,
        agent,
        baseline,
        delta=args.delta,
        num_rollouts=args.rollouts,
        tool_registry=registry,
        judge_scores=judge_scores,
        judge_min_score=getattr(args, "judge_min_score", None),
    )

    print(f"{'scenario':<{MAX_COLS}}{'verdict':<10}  reason")
    failures = 0
    for g in report.per_scenario:
        print(f"{g.scenario_id:<{MAX_COLS}}{g.decision.verdict:<10}  {g.decision.reason}")
        if not g.passed:
            failures += 1
            _print_scores(g.current_scores)
    print(f"\ngate: {'PASS' if failures == 0 else f'{failures} regression(s)'}")
    return 0 if failures == 0 else 1


def cmd_mutate(args) -> int:
    scenarios, registry = load_scenario_suite(args.suite)
    behaviors = load_behavior_config(args.behavior)
    report = _run_report(
        scenarios, registry, behaviors, args.behavior, True, args.seed, args.run_id,
        args.rollouts,
    )
    _print_mutations(report)
    if args.out:
        save_json(report.model_dump(mode="json"), args.out)
        print(f"\nwrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracegate",
        description="Deterministic, mutation-validated regression infrastructure for agentic pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_text in (
        ("run", cmd_run, "run the suite and print a report"),
        ("baseline", cmd_baseline, "capture a known-good baseline report"),
        ("gate", cmd_gate, "compare a fresh run against a baseline; CI entrypoint"),
        ("mutate", cmd_mutate, "run the suite with mutation testing (kill-rate)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("suite", help="path to scenario suite YAML")
        p.add_argument("-b", "--behavior", default=None, help="path to behavior config YAML")
        p.add_argument("-o", "--out", default=None, help="output JSON path")
        p.add_argument("--run-id", default="", help="label for this run")
        p.add_argument("--seed", type=int, default=42, help="RNG seed for mutations")
        p.add_argument("--mutation", action="store_true", help="include mutation kill-rate")
        p.add_argument("--rollouts", type=int, default=1,
                       help="samples per scenario; gate on the worst (default 1)")
        p.add_argument("--verify-kill-rate", type=float, default=None,
                       help="baseline: verify-before-write; re-roll until kill-rate >= this "
                            "and no hard violations, else fail")
        p.add_argument("--retries", type=int, default=3,
                       help="baseline: max verify attempts (default 3)")
        p.add_argument("-r", "--reference", default=None, help="baseline JSON (gate)")
        p.add_argument("--delta", type=float, default=1e-9, help="regression tolerance (gate)")
        # Judge gating (P1: wire judge to CLI)
        p.add_argument("--judge-provider", default=None, help="judge provider: ollama or api/openai (gate)")
        p.add_argument("--judge-model", default=None, help="judge model name (gate)")
        p.add_argument("--judge-base-url", default=None, help="judge base_url for OpenAI-compatible endpoint (gate)")
        p.add_argument("--judge-api-key", default=None, help="judge API key (gate)")
        p.add_argument("--judge-min-score", type=float, default=None, help="enable judge gating: fail if satisfaction < this (gate, default off)")
        p.add_argument("--judge-samples", type=int, default=3, help="N-shot samples for judge (gate)")
        p.add_argument("--judge-temperature", type=float, default=0.0, help="judge temperature (gate)")
        p.set_defaults(fn=fn)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
