"""Trajectory mutation testing.

The metric suite is only as good as its ability to *detect* regressions. This
module mechanically injects named regression classes into a golden trace and
reports whether the metrics catch each one — the "kill rate". A mutant that
survives is a blind spot in the gate, by construction.

A "kill" means: the mutated run is flagged as worse than the golden run by the
metric component(s) the mutation *should* have triggered (``signal``), or it
acquires a hard violation the golden run did not have. Comparing component
values against the golden run (not against an absolute threshold) keeps the
test meaningful when golden runs are imperfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from tracegate.metrics import ComponentScores, score_trace
from tracegate.schema import Scenario, ToolCall, TraceGateError, Trajectory

EPSILON = 1e-9

# Decoys used to synthesize wrong/unknown tool calls when a scenario does not
# declare an explicit forbidden tool.
_DECOYS = ("DROP_DATABASE", "SEND_EMAIL_TO_ALL", "UNKNOWN_METHOD")


@dataclass(frozen=True)
class Mutation:
    name: str
    description: str
    signal: tuple[str, ...]
    applicable: bool = True
    skip_reason: str = ""


class MutationResult(BaseModel):
    name: str
    description: str
    killed: bool
    detail: str
    golden_overall: float
    mutated_overall: float
    mutated_violations: list[str]


class MutationSuiteResult(BaseModel):
    scenario_id: str
    total: int
    killed: int
    survival: int
    kill_rate: float
    results: list[MutationResult]


def _decoy_for(scenario: Scenario, seed: int = 42) -> str:
    """Pick a decoy tool name that is guaranteed *not* legitimate for scenario.

    Shuffles ``_DECOYS`` based on ``seed`` so re-rolls with different seeds
    produce different decoys (P1 fix for seed-dead bug).
    """
    import random

    rng = random.Random(seed)
    candidates = list(_DECOYS)
    rng.shuffle(candidates)
    for d in candidates:
        legit = (
            d in scenario.required_tools
            or d in scenario.forbidden_tools
            or d in scenario.expected_sequence()
            or (scenario.allowed_tools is not None and d in scenario.allowed_tools)
        )
        if not legit:
            return d
    return "UNKNOWN_METHOD"


def _clone(trace: Trajectory, calls: list[ToolCall], terminated: bool) -> Trajectory:
    return trace.model_copy(
        update={"calls": calls, "terminated": terminated, "final_answer": trace.final_answer}
    )


def _mutate_wrong_tool(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Replace the name of one call with a decoy tool (seed varies target)."""
    if not trace.calls:
        raise TraceGateError("wrong_tool mutation requires at least one call")
    decoy = _decoy_for(scenario, seed=seed)
    # Vary target with seed so re-rolls change which call is corrupted
    target_idx = (len(trace.calls) // 2 + seed) % len(trace.calls)
    target = trace.calls[target_idx]
    calls = [
        c.model_copy(update={"name": decoy}) if c.index == target.index else c
        for c in trace.calls
    ]
    return _clone(trace, calls, trace.terminated)


def _mutate_premature_termination(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Stop the run before all obligations are met, but claim termination."""
    if not trace.calls:
        raise TraceGateError("premature_termination mutation requires at least one call")
    obligations = [t for t in scenario.required_tools if t not in scenario.forbidden_tools]
    if obligations:
        last_obligation_index = max(
            i for i, c in enumerate(trace.calls) if c.name in obligations
        )
        truncated = trace.calls[: max(1, last_obligation_index)]
    else:
        truncated = trace.calls[:-1]  # drop the final step
    if not truncated:
        truncated = trace.calls[:1]
    return _clone(trace, truncated, True)


def _mutate_dropped_call(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Remove a call, preferring a required-tool call when one exists (seed varies fallback)."""
    if not trace.calls:
        raise TraceGateError("dropped_call mutation requires at least one call")
    drop_index = None
    for i, c in enumerate(trace.calls):
        if c.name in scenario.required_tools:
            drop_index = i
            break
    if drop_index is None:
        drop_index = (len(trace.calls) // 2 + seed) % len(trace.calls) if trace.calls else 0
    calls = [c for i, c in enumerate(trace.calls) if i != drop_index]
    return _clone(trace, calls, trace.terminated)


def _mutate_reorder(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Swap two adjacent calls with different names — order-sensitive regressions should fire."""
    if len(trace.calls) < 2:
        raise TraceGateError("reorder mutation requires at least two calls")
    # Find first adjacent pair with different names to ensure sequence actually changes
    swap_idx = None
    for i in range(len(trace.calls) - 1):
        if trace.calls[i].name != trace.calls[i + 1].name:
            swap_idx = i
            break
    if swap_idx is None:
        # All names identical — swapping won't affect sequence; fall back to first two
        swap_idx = 0
    calls = trace.calls[:]
    calls[swap_idx], calls[swap_idx + 1] = calls[swap_idx + 1], calls[swap_idx]
    calls = [c.model_copy(update={"index": i}) for i, c in enumerate(calls)]
    return _clone(trace, calls, trace.terminated)


def _mutate_extra_forbidden_call(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Append a forbidden/decoy call at the end of the trajectory."""
    if scenario.forbidden_tools:
        # Use seed to rotate which forbidden tool is used when multiple exist
        decoy = scenario.forbidden_tools[seed % len(scenario.forbidden_tools)]
    else:
        decoy = _decoy_for(scenario, seed=seed)
    extra = ToolCall(index=len(trace.calls), name=decoy, arguments={}, result=None)
    calls = trace.calls + [extra]
    return _clone(trace, calls, trace.terminated)


def _mutate_wrong_arguments(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Corrupt arguments of a pinned expected call."""
    if not trace.calls:
        raise TraceGateError("wrong_arguments mutation requires at least one call")
    # Prefer a call that has a pinned expected counterpart
    pinned_names = {e.name for e in scenario.expected_tool_calls if e.arguments is not None}
    target_idx = None
    for i, c in enumerate(trace.calls):
        if c.name in pinned_names:
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(trace.calls) // 2
    calls = []
    for i, c in enumerate(trace.calls):
        if i == target_idx:
            # Inject wrong args: flip first expected value or add sentinel
            wrong_args = dict(c.arguments) if c.arguments else {}
            if wrong_args:
                # Corrupt first key
                first_k = next(iter(wrong_args))
                wrong_args[first_k] = "__WRONG__"
            else:
                wrong_args = {"__wrong_arg__": "__WRONG__"}
            calls.append(c.model_copy(update={"arguments": wrong_args}))
        else:
            calls.append(c)
    return _clone(trace, calls, trace.terminated)


def _mutate_extra_allowed_call(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Append an allowed but non-required tool (quality regression, no hard violation)."""
    candidates: list[str] = []
    if scenario.allowed_tools:
        candidates = [t for t in scenario.allowed_tools if t not in scenario.forbidden_tools and t not in scenario.required_tools]
    if not candidates:
        # Fallback: duplicate first call's name as extra allowed
        candidates = [trace.calls[0].name] if trace.calls else ["get_orders"]
    extra_name = candidates[seed % len(candidates)]
    extra = ToolCall(index=len(trace.calls), name=extra_name, arguments={}, result=None)
    calls = trace.calls + [extra]
    return _clone(trace, calls, trace.terminated)


def _mutate_missing_termination(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Keep all calls but flip terminated to False."""
    if not trace.calls:
        raise TraceGateError("missing_termination mutation requires at least one call")
    return _clone(trace, trace.calls, False)


def _mutate_length_overflow(trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    """Add calls until max_calls is exceeded."""
    if scenario.max_calls is None:
        # No max, add 3 extra to trigger sequence drop
        extras = [ToolCall(index=len(trace.calls) + i, name=trace.calls[0].name if trace.calls else "get_orders", arguments={}, result=None) for i in range(3)]
        return _clone(trace, trace.calls + extras, trace.terminated)
    needed = scenario.max_calls - len(trace.calls) + 1
    if needed <= 0:
        needed = 1
    # Use an allowed tool if possible, else first call's name
    extra_name = "get_orders"
    if trace.calls:
        extra_name = trace.calls[0].name
    extras = [ToolCall(index=len(trace.calls) + i, name=extra_name, arguments={}, result=None) for i in range(needed)]
    calls = trace.calls + extras
    return _clone(trace, calls, trace.terminated)


MUTATORS: dict[str, tuple[object, tuple[str, ...]]] = {
    "wrong_tool": (_mutate_wrong_tool, ("forbidden", "sequence")),
    "premature_termination": (_mutate_premature_termination, ("termination", "required_coverage")),
    "dropped_call": (_mutate_dropped_call, ("required_coverage", "sequence")),
    "reorder": (_mutate_reorder, ("sequence",)),
    "extra_forbidden_call": (_mutate_extra_forbidden_call, ("forbidden", "sequence", "length")),
    "wrong_arguments": (_mutate_wrong_arguments, ("arg_accuracy", "sequence")),
    "extra_allowed_call": (_mutate_extra_allowed_call, ("sequence", "length")),
    "missing_termination": (_mutate_missing_termination, ("termination",)),
    "length_overflow": (_mutate_length_overflow, ("length", "sequence")),
}


def build_mutations(trace: Trajectory, scenario: Scenario, seed: int = 42) -> list[Mutation]:
    """Return the applicable mutations for a (trace, scenario) pair.

    ``seed`` now influences decoy selection and target choice inside each
    mutator, so different seeds produce different mutants (fix for seed-dead
    bug). The mutation list itself is stable, but the *effect* varies with
    seed during ``apply_mutation``.
    """
    out: list[Mutation] = []
    for name, (fn, signal) in MUTATORS.items():
        applicable = True
        skip_reason = ""
        if name == "reorder" and len(trace.calls) < 2:
            applicable, skip_reason = False, "needs >=2 calls"
        if name in ("wrong_tool", "dropped_call", "wrong_arguments", "missing_termination") and not trace.calls:
            applicable, skip_reason = False, "needs >=1 call"
        if name == "wrong_arguments":
            has_pinned = False
            for e in scenario.expected_tool_calls:
                args = e.get("arguments") if isinstance(e, dict) else getattr(e, "arguments", None)
                if args is not None:
                    has_pinned = True
                    break
            if not has_pinned:
                applicable, skip_reason = False, "needs pinned arguments in scenario"
        if name == "length_overflow" and len(trace.calls) == 0 and scenario.max_calls is None:
            applicable, skip_reason = False, "needs >=1 call or max_calls"
        if applicable:
            out.append(
                Mutation(
                    name=name,
                    description=f"inject: {name}",
                    signal=signal,
                )
            )
    return out


def apply_mutation(name: str, trace: Trajectory, scenario: Scenario, seed: int = 42) -> Trajectory:
    fn, _ = MUTATORS[name]
    try:
        return fn(trace, scenario, seed)  # type: ignore[operator]
    except TypeError:
        # Fallback for mutators that don't accept seed
        return fn(trace, scenario)  # type: ignore[operator]


def _killed(
    mutation: Mutation,
    golden: ComponentScores,
    mutated: ComponentScores,
) -> bool:
    new_hard = [v for v in mutated.hard_violations if v not in golden.hard_violations]
    if new_hard:
        return True
    for comp in mutation.signal:
        if getattr(mutated, comp) < getattr(golden, comp) - EPSILON:
            return True
    return False


def run_mutation_suite(
    scenario: Scenario,
    golden_trace: Trajectory,
    golden_scores: ComponentScores,
    weights: dict[str, float] | None = None,
    seed: int = 42,
    tool_registry=None,
) -> MutationSuiteResult:
    """Mutate a golden trace and report how many mutants the gate kills."""
    results: list[MutationResult] = []
    for mutation in build_mutations(golden_trace, scenario, seed=seed):
        if not mutation.applicable:
            continue
        try:
            mutated_trace = apply_mutation(mutation.name, golden_trace, scenario, seed=seed)
        except TraceGateError:
            continue
        mutated_scores = score_trace(mutated_trace, scenario, weights=weights, tool_registry=tool_registry)
        killed = _killed(mutation, golden_scores, mutated_scores)
        results.append(
            MutationResult(
                name=mutation.name,
                description=mutation.description,
                killed=killed,
                detail=(
                    "caught by signal component(s) or new hard violation"
                    if killed
                    else "SURVIVED: no metric fired"
                ),
                golden_overall=golden_scores.overall,
                mutated_overall=mutated_scores.overall,
                mutated_violations=mutated_scores.hard_violations,
            )
        )
    total = len(results)
    killed = sum(1 for r in results if r.killed)
    return MutationSuiteResult(
        scenario_id=scenario.id,
        total=total,
        killed=killed,
        survival=total - killed,
        kill_rate=(killed / total) if total else 0.0,
        results=results,
    )


__all__ = [
    "Mutation",
    "MutationResult",
    "MutationSuiteResult",
    "build_mutations",
    "apply_mutation",
    "run_mutation_suite",
]
