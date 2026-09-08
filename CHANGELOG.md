# CHANGELOG

Quick reference for what was done and when.

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
