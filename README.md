# tracegate

Deterministic, mutation-validated regression infrastructure for agentic pipelines.

Single-turn evals grade the final answer. Agents fail on the *path*: a dropped
tool call, a forbidden action, a never-reached obligation — all invisible in the
last message. tracegate treats the **trajectory** (the ordered tool calls an
agent makes) as the artifact under test, and turns "did this update break the
agent?" into a deterministic merge gate.

```
                       ┌─────────────────────────────┐
                       │        baseline.json        │
                       │  (verified known-good path) │
                       └─────────────▲───────────────┘
                                     │ deterministic compare
   scenario.yaml ──► agent ──► trajectory ──► 5 metrics ──► verdict
   (prompt +            │                            │
    tool contract)      └── 3 rollouts ── worst gates ┘
                                      │
                               mutation suite
                          (kill-rate: how well does
                            the gate detect faults?)
```

Two independent layers, and they measure different things:

| layer | measures | gating role |
|---|---|---|
| **deterministic gate** | the *path* — tool calls vs. contract | blocks merge, no model in the loop |
| **LLM judge** (opt-in) | the *goal* — final answer satisfies intent | never blocks by default; `--judge-min-score` to enable |

They disagree on purpose: a clean tool path followed by a refusal answer is a
regression the deterministic gate cannot see. The judge layer exists for that
case. A broken path is caught by the deterministic gate before any judge runs.

## Why deterministic first

Every decision that gates merges is pure arithmetic over structured tool calls
— no model, no temperature, no flake. The only nondeterminism left is the agent
itself, and that is handled explicitly with multi-rollout gating (see below).

## The five metrics

| component | what it catches | range | hard |
|---|---|---|---|
| `sequence` | reordered / dropped / added steps (LCS, normalized) | 0–1 | — |
| `required_coverage` | a required tool was never called | 0–1 | — |
| `forbidden` | a forbidden, wrong, or unknown tool was called | 0/1 | yes |
| `termination` | never stopped, or stopped before obligations were met | 0/1 | yes |
| `length` | suspiciously few or many calls | 0/1 | — |

Hard violations fail a scenario outright. `overall` is the product of the
non-hard components after hard violations are applied. `compare_to_baseline`
turns old-score vs. new-score into a verdict with tolerance (`--delta`):

- **REGRESSION** — a hard violation appeared, or a component dropped below the
  baseline minus delta
- **PASS** — within tolerance
- **IMPROVED** — a hard violation cleared or a component rose above baseline

## Mutation-validated baselines

A gate is only trustworthy if it can detect faults. tracegate self-tests the
suite: it injects faults into the baseline trajectory (`wrong_tool`,
`premature_termination`, `dropped_call`, `reorder`,
`extra_forbidden_call`), runs the gate against each mutation, and reports a
**kill-rate** — the fraction of injected faults the gate detects.

```
kill-rate = detected mutations / total mutations
```

Baseline capture is **verify-before-write**: a capture whose own kill-rate falls
below `0.8` is refused and re-run rather than silently written as a weak
baseline.

## Multi-rollout gating (nondeterminism)

LLM agents are nondeterministic even at temperature 0 (verified on llama3.1:8b).
A single lucky run can false-pass a regressed agent, and a single unlucky run
can false-fail a healthy one. tracegate samples each scenario `N` times
(`--rollouts`, default 3) and:

- **gates on the worst rollout** (a healthy agent never produces a hard
  violation, so the worst is stable),
- stores every rollout in the report (`mean_over_rollouts`, `min_overall`),
- refuses to write a baseline until a full verified capture succeeds.

## The LLM judge layer (opt-in, N-shot)

A `JudgeAgent` scores whether the *final answer* satisfies the scenario's goal
in `[0,1]` with a label (`SATISFIED` / `UNSATISFIED` / `UNAVAILABLE`). Because a
single judge call is unreliable (a refusal was scored 0.00 in one session and
0.80 in another on the same trajectory), the judge is:

- **N-shot** — sampled N times with varied sampling; results report `mean`,
  `min`, `max`, `agreement`, and every rationale;
- **opt-in** — the deterministic gate never calls a model; the judge only runs
  if you pass `--judge-min-score`;
- **fail-closed** — a judge that errors yields `UNAVAILABLE`, which is a
  regression, never a silent pass.

## Install

```
pip install -e ".[dev]"          # core + tests
pip install -e ".[ollama]"       # + real-agent demo deps (langgraph, ollama)
```

Requires Python ≥ 3.10.

## Quickstart (offline — no model)

A `ScriptedAgent` replays scripted trajectories, so the whole harness runs in
milliseconds with zero LLM calls.

```
# 1. baseline from a known-good scripted agent
tracegate baseline scenarios/example.yaml -b scenarios/behavior_good.yaml -o baseline.json

# 2. gate a candidate agent against it
tracegate gate scenarios/example.yaml -b scenarios/behavior_buggy.yaml -r baseline.json

# 3. sanity-check the suite itself
tracegate mutate scenarios/example.yaml
```

`tracegate run` runs a single scenario without a baseline. Add `--rollouts 5`
anywhere to control sampling.

## CI / merge gate

The gate is an exit code, so it blocks merges in CI. `.github/workflows/ci.yml`
runs it on every PR and push:

1. **Unit tests** (`pytest`).
2. **Verify-before-write baseline** — rebuilds the baseline with
   `--verify-kill-rate 0.8`; CI fails if the harness can no longer produce a
   trustworthy (kill-rate ≥ 0.8, no hard violations) baseline.
3. **Known-good must PASS** — the reference agent must still clear its own
   baseline.
4. **Drift must still be detected** — the buggy agent must REGRESS; CI fails if
   it passes, i.e. if an edit silently blinded the gate.
5. **Score card** — posts/updates a PR comment with the gate output and the
   mutation kill-rate.

Point branch protection at the `test` and `gate` jobs and a regression in the
agent, the scenarios, or the harness itself blocks the merge.

## Real-agent demo (LangGraph + Ollama)

`examples/real_agent/` is a fully working agent (order-management tools) built
with LangGraph and run against a local llama3.1:8b via Ollama.

```
# capture a verified known-good baseline from the GOOD prompt
python examples/real_agent/demo.py --capture baseline.json --judge

# block the merge when the prompt regresses (buggy prompt stands in for a bad release)
python examples/real_agent/gate.py examples/real_agent/baseline.json --rollouts 3

# also fail scenarios whose goal is unmet even though the path is clean
python examples/real_agent/gate.py examples/real_agent/baseline.json --judge-min-score 0.5
```

The demo prints a score card:

```
== sanity gate (GOOD prompt vs baseline) ==
sc_total_spend    PASS     overall=1.0000 vs 1.0000
sc_cancel_order   PASS     overall=1.0000 vs 1.0000
== prompt drift (BUGGY prompt vs baseline) ==
sc_total_spend    REGRESSION  new hard violation(s): termination_contract_violated, required_tool_missing
sc_cancel_order   REGRESSION  new hard violation(s): termination_contract_violated, required_tool_missing
```

## Scenario format

```yaml
registry:
  tools:
    - name: cancel_order
      description: "Cancel an order"
      dangerous: true
      allowed_roles: [customer_service]

scenarios:
  - id: sc_cancel_order
    prompt: "Cancel my order"
    expected_tool_calls:
      - { name: get_orders }
      - { name: cancel_order }
    required_tools: [get_orders, cancel_order]
    forbidden_tools: [refund_order]      # refunding is a different permission
    allowed_tools: [get_orders, cancel_order, compute_total]
    min_calls: 1
    max_calls: 5
    must_terminate: true
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design rationale, the
findings from the real-agent build, and the extension points.

## Status

- deterministic gate, 5 metrics, hard violations, regression verdicts: done
- mutation suite + verify-before-write baseline capture: done
- multi-rollout gating: done
- opt-in N-shot LLM judge: done
- LangGraph adapter + Ollama real-agent demo: done
- roadmap: more metric components, multi-agent topologies, judge per-tool-call
  grounding, GitHub App for comment-on-PR gating

## License

MIT
