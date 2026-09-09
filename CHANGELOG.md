# CHANGELOG

Quick reference for what was done and when.

---

## September 2026

### 2026-09-08 (P1)

| Commit | Description |
|--------|-------------|
| `pending` | **P1 blind spots** — arg_accuracy + dangerous metrics, 4 new mutators, judge prompt & CLI |

**P1 — Measurement Blind Spots (follow-up to P0):**
- `arg_accuracy` metric (`metrics.py:267`): subset match on pinned `ExpectedToolCall.arguments`, weight 1.2, `score_trace(..., tool_registry)` plumbing via `registry`, `_call_metric` helper
- `dangerous_without_confirm` metric (`metrics.py:307`): `DangerousWithoutConfirmMetric` with heuristic fallback + `ToolRegistry.dangerous` flag, weight 2.0 hard, `seen_confirm` guard (confirm itself never flagged)
- 4 new mutators (`mutation.py:166`): `wrong_arguments` (→ arg_accuracy), `extra_allowed_call` (→ sequence/length), `missing_termination` (→ termination), `length_overflow` (→ length/sequence); tot 9 (8 for example.yaml, 9 with pinned args); fixed `seed` (shuffle + target_idx) and `_mutate_reorder` same-name, `_capture_verified` `seed+attempt-1`
- CLI plumbing: `_run_report`, `run_gate`, `run_suite*` now accept `tool_registry`; `cli.py:167` gate `--judge-*` flags (`provider/model/base-url/api-key/min-score/samples/temperature`) with `apply_judge` pre-run
- Judge prompt (`judge.py:123`): `max_total_chars=8000` global budget + `scenario.description` injection; per-result 500 remains
- `dangerous` now hard via `registry` + `tool_registry` param in `suite`/`runner`/`mutation`; `build_mutations` handles dict vs object; tests: `test_all_mutations_are_killed` expects ≥5, 6 new tests `test_arg_accuracy_*`/`test_dangerous_*`; all 85 tests passing; gate probes still PASS (overall 0.9638 vs 0.9444 due to new weights, kill-rate 8/8)

### 2026-09-08 (P0)

| Commit | Description |
|--------|-------------|
| `e8a53a1` | **P0 hardening** — 9 correctness fixes (PR #7) + roadmap rebuild after full audit |

**P0 — Correctness & Hardening (audit 2026-09-08):**
- Docs ↔ code: `overall` is weighted avg (weights 1.0/1.0/2.0/1.5/0.5) — fixed `README.md:53`, `docs/ARCHITECTURE.md:67`, `docs/blog.md:67` (was *product*)
- `ComponentScores` + `SuiteReport` typing: `field()` → `Field(default_factory=…)` (`metrics.py:196`, `suite.py:30`)
- Atomic writes: `save_json` tmp+rename (`io_utils.py:85`), no half-written baselines
- Schema versioning: `SCHEMA_VERSION=1.0` (`schema.py:3`), `SuiteReport.schema_version` with default (old baselines still load)
- Per-scenario error isolation: `suite.py:56` `_error_result`, suite no longer crashes on one agent exception; `agent_error` hard violation
- Dedup + registry validation: duplicate `id` error + unknown tool refs error (`io_utils.py:51`, `65`)
- Single-source `__version__` via `importlib.metadata` (`__init__.py:13`)
- `LangGraphAdapter` now populates `arguments` via `tool_call_id` correlation (`agents/langgraph.py:39`), handles dict/object messages, result normalization
- `agents/ollama.py` lazy import + `recursion_limit` now stashed on graph and passed to `invoke` via config
- Roadmap rebuilt (ROADMAP.md gitignored) — P0-P5 reprioritized; all 80 tests still passing; gate probes (example + gate_only) verified

---

## August 2026

### 2026-08-01 (continued)

| Commit | Description |
|--------|-------------|
| `pending` | **feat: Custom metric plugin system** - `MetricPlugin` protocol, `MetricRegistry`, register/unregister custom metrics |

**What was added:**
- `MetricPlugin` protocol for custom metrics
- `MetricRegistry` class with `register()`, `unregister()`, `get()`, `all()`, `clear()`
- Global `registry` instance for easy access
- `score_trace()` now supports custom metrics via `include_custom` parameter
- Custom hard violation metrics automatically detected
- 14 new tests for the plugin system

**Usage example:**
```python
from tracegate.metrics import registry, MetricPlugin

class MyMetric:
    name = "custom_ratio"
    default_weight = 1.0
    is_hard_violation = False

    def score(self, trace, scenario):
        return 1.0 if len(trace.calls) > 0 else 0.0

registry.register(MyMetric())
```

---

### 2026-08-01

| Commit | Description |
|--------|-------------|
| `09100a4` | Merge PR #5 - `feat/judge-provider` |
| `9bfbcb1` | Added `--judge-provider` flag to swap judge backend in demo |
| `7dfe1c5` | Merge PR #4 - `feat/api-judge` |
| `f77d52d` | Added `OpenAICompatibleJudge` - provider-neutral LLM judge via any OpenAI-compatible endpoint |
| `8163dd0` | Merge PR #3 - `docs/branch-protection` |
| `39aa6f4` | Documented branch protection enabled on main (test + gate required) |
| `adfbc9e` | Added gate-only drift probe (overall-drop path) for red-PR demos |
| `bd1db26` | Made forbidden-tool calls advisory instead of hard violations |
| `ad33e48` | Merge PR #1 - `ci/update-actions` |
| `55658b3` | Bumped checkout@v5, setup-python@v6 (clear Node 20 deprecation) |
| `a23aaa5` | Added CI PR merge gate with verify-before-write, drift sensitivity check, score card |
| `f92d521` | **Initial release** - tracegate: deterministic, mutation-validated trajectory regression |

---

## Summary of Major Milestones

| Milestone | Date | Status |
|-----------|------|--------|
| Core deterministic gate (5 metrics) | 2026-08-01 | ✅ Done |
| Mutation suite + verify-before-write | 2026-08-01 | ✅ Done |
| Multi-rollout gating | 2026-08-01 | ✅ Done |
| Opt-in N-shot LLM judge | 2026-08-01 | ✅ Done |
| OpenAI-compatible judge | 2026-08-01 | ✅ Done |
| CI/CD merge gate pipeline | 2026-08-01 | ✅ Done |
| Branch protection on main | 2026-08-01 | ✅ Done |
| LangGraph + Ollama demo | 2026-08-01 | ✅ Done |

---

## Key Decisions

- **2026-08-01**: Forbidden-tool calls changed from hard violations to advisory (allows more flexible scenarios)
- **2026-08-01**: Gate-only drift probe added to catch subtle quality regressions without hard violations

---

*Last updated: 2026-08-01*
