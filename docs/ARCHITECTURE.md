# tracegate — Architecture

This document is the design write-up behind the harness: the data model, the
reasoning for each component, what the real-agent build actually taught us, and
where the design extends.

## 1. The problem

Single-turn evals grade a final answer against a rubric. Agentic pipelines are
not single-turn systems: an agent acts by calling tools, and the *sequence and
legality* of those calls is where failures live. Three failure classes are
invisible in a final-answer evaluation:

1. **Path errors** — a required tool was never called, calls were reordered,
   an obligation was left unmet, or a forbidden tool was invoked.
2. **Permission errors** — the agent performed an action it is not entitled to
   (a refund instead of a cancel, a delete instead of a query).
3. **Goal errors** — the path is clean but the agent refuses, hallucinates
   completion, or satisfies a different intent.

A regression harness for agents must therefore record and compare **trajectories
of tool calls**, not just answers.

## 2. Data model

```
Scenario
  id, prompt, goal
  tool_registry          # names, descriptions, danger flags, roles
  expected_tool_calls    # ordered, for sequence scoring
  required_tools         # coverage obligations
  forbidden_tools        # hard illegal actions
  allowed_tools          # permission envelope
  min_calls, max_calls, must_terminate

Trajectory
  run_id, scenario_id
  calls: ToolCall[]      # name, args, timestamp
  final_response
```

Everything downstream is pure arithmetic over `ToolCall` names and order. There
is no free text in the deterministic path, which is what keeps the gate fast and
deterministic.

`AgentAdapter` is the only seam between the harness and a real agent. The
framework-agnostic `Trajectory` means a LangGraph graph, an Ollama loop, a
`ScriptedAgent` (for tests), or a future `BehaviorMappedAgent` all produce the
same shape.

## 3. Why the deterministic gate calls no model

The merge gate is the highest-stakes, highest-frequency path. If it calls an LLM
it inherits that model's nondeterminism, cost, and failure modes, and the gate
itself needs a meta-gate. Restricting the gate to deterministic math over
structured calls means:

- the verdict is reproducible bit-for-bit;
- the gate runs in microseconds (unit-test speed, so it can gate every PR);
- `kill-rate` measures *the gate's* sensitivity, not the judge's.

The LLM is deliberately exiled to an optional second layer with different
semantics (Section 6).

## 4. Metric components

`overall` is the weighted average of all components (weights: `sequence` 1.0,
`required_coverage` 1.0, `forbidden` 2.0, `termination` 1.5, `length` 0.5;
`DEFAULT_WEIGHTS` in `metrics.py`). Hard violations still gate directly;
`overall` provides the soft drift signal within tolerance:

- **sequence** — Longest-common-subsequence similarity of observed vs. expected
  calls, normalized to `[0,1]`. Catches reordering and dropped or invented
  steps.
- **required_coverage** — fraction of `required_tools` actually called.
  Catches omitted obligations.
- **forbidden** (hard) — any call not in `allowed_tools`, or in
  `forbidden_tools`, fails the scenario.
- **termination** (hard) — `must_terminate` scenarios fail if the agent never
  finished, or if it stopped before `required_tools` were satisfied.
- **length** — call count outside `[min_calls, max_calls]`.

Hard violations are the contract: they bypass numeric tolerance entirely. This
is the difference between "slightly worse" (PASS with a delta) and "illegal"
(REGRESSION always).

`compare_to_baseline` compares per-component scores with a `--delta` tolerance:

| new vs. baseline | verdict |
|---|---|
| new hard violation appeared | REGRESSION |
| component dropped below `baseline - delta` | REGRESSION |
| within tolerance | PASS |
| violation cleared / component rose above `baseline + delta` | IMPROVED |

## 5. Mutation-validated baselines

A gate is trustworthy only if it detects the faults it is meant to prevent. The
mutation suite is a self-test: it mutates a reference trajectory with known
faults and checks the gate catches them.

| mutation | injected fault | expected detection |
|---|---|---|
| `wrong_tool` | replace a call with a decoy | forbidden |
| `premature_termination` | stop before obligations met | termination |
| `dropped_call` | remove a required call | required_coverage, sequence |
| `reorder` | swap two calls | sequence |
| `extra_forbidden_call` | append a forbidden call | forbidden, sequence |

`kill-rate = detected / total` is the suite's sensitivity, reported per
scenario and overall. Capture is **verify-before-write**: a capture whose
kill-rate is below `0.8` is refused and retried rather than persisted as a weak
baseline. A baseline is not an assumption; it is a claim that must pass its own
tests before it can gate anything.

## 6. Two layers, deliberately different

The deterministic gate and the LLM judge measure disjoint things and are wired
with opposite defaults:

| | deterministic gate | LLM judge |
|---|---|---|
| measures | path (tool calls) | goal (final answer) |
| default | always runs | opt-in |
| failure | blocks merge (exit non-zero) | reports only, unless `--judge-min-score` |
| determinism | pure | N-shot sampled, mean/min/max/agreement |
| model error | n/a | fail-closed: UNAVAILABLE = regression |

This asymmetry is the point. Path regressions are caught cheaply and
deterministically; goal regressions are caught by an explicitly authorized,
costlier, noisier layer whose reliability is itself reported (`agreement` across
samples) so a human can decide whether to trust it in their CI.

## 7. Nondeterminism handling

Empirically (llama3.1:8b via Ollama), an agent at `temperature=0` is still
nondeterministic: identical inputs produce different tool paths across runs. The
harness does not try to eliminate this; it samples through it:

- each scenario runs `N` rollouts (`--rollouts`, default 3);
- the **worst** rollout is the gating representative — a healthy agent never
  produces a hard violation, so worst-of-N is stable, while a regressed agent
  cannot hide behind one lucky clean sample;
- all rollouts are retained in the report (`mean_over_rollouts`, `min_overall`,
  per-run scores) so the spread is visible, not hidden;
- verify-before-write refuses a baseline that any rollout violates.

## 8. The LLM judge, hardened

A single judge call is unreliable. The refusal scenario was scored `0.00` in one
session and `0.80` in another on the same trajectory. The judge therefore:

- samples N times with varied sampling (the judge model uses no fixed seed so
  samples genuinely differ);
- reports `mean`, `min`, `max`, and `agreement` alongside every rationale;
- is never the sole source of truth for a merge unless explicitly enabled with a
  threshold; and
- is fail-closed — errors map to `UNAVAILABLE`, which is a regression.

Rationales are retained (one per sample) for audit, which is the property that
makes a noisy judge reviewable rather than mystical.

The judge is provider-neutral. `OllamaJudge` (local, `langchain-ollama`) and
`OpenAICompatibleJudge` (any OpenAI-compatible `/chat/completions` endpoint,
pure `urllib`, no dependency) implement the same `LLMJudge` seam — swap one for
the other without touching prompt, parse, sampling, or gating. This keeps the
judge layer portable across local, cloud, and self-hosted models.

## 9. Findings from the real-agent build

These are the concrete failures the demo agent surfaced, each of which
changed the design:1. **Nondeterminism at temperature 0.** No seed pinning made trajectory output
   reproducible; the fix is multi-rollout gating (Section 7), not seed worship.
2. **Runaway generation.** A tool-loop produced an 8K-token non-terminating
   run. Fixed in the agent with `num_predict` caps; the harness enforces the
   same contract via `must_terminate` + `max_calls`.
3. **Schema friction.** `list[float]` in a tool's input schema produced
   validation loops in the model loop; the tool was redesigned to accept a
   `str` of amounts and parse them. Real harness value: trajectory comparison
   must be resilient to argument-representation drift, not coupled to it.
4. **Goal-refusal divergence.** A clean trajectory (all calls correct, proper
   termination) whose final answer refused the user scored `PASS` on the
   deterministic gate and `FAILED 0.00` on the judge. This is the canonical
   case for the two-layer design — neither layer alone sees it.

## 10. Extension points

- **New metric components** — add a component to `metrics.py`; it flows through
  scoring, regression comparison, and the mutation suite automatically.
- **New adapters** — implement `AgentAdapter` for ReAct loops, MCP, or
  multi-agent graphs; the trajectory contract is framework-agnostic.
- **Per-call grounding for the judge** — feed the judge the tool-call trace
  alongside the final answer so goal-vs-path disagreement is grounded.
- **CI integration** — `gate.py` and `tracegate gate` are exit-code gates;
  a GitHub Action can gate PRs and comment the score card (see
  `.github/workflows/ci.yml` for the pattern).
