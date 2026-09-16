"""LLM-as-judge secondary score: goal satisfaction.

This is the *non-gating* layer of the harness. The deterministic metrics and
merge gate never call a model; the judge is a separate, clearly-labeled signal
that answers a question determinism cannot: "did the agent actually satisfy the
user's goal, even if its path was wrong or its path was right but the answer
was off?"

Design rules:
- The judge NEVER participates in ``compare_to_baseline`` / the merge gate.
- Judge output is versioned (``judge_model``) and informational.
- Judge failures degrade to ``UNAVAILABLE`` instead of failing a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from typing import Any, Protocol

from pydantic import BaseModel, Field

from tracegate.schema import Scenario, Trajectory

JUDGE_PROMPT_VERSION = "1.1"

# In-memory caches for cost reduction (P2: judge caching)
_MAX_CACHE_SIZE = 128
_SAMPLE_CACHE: dict[str, SampledJudgeResult] = {}
_SINGLE_CACHE: dict[str, JudgeResult] = {}


def _trace_hash(trace: Trajectory) -> str:
    """Stable hash for a trajectory's content (for cache keys)."""
    payload = json.dumps(
        {
            "scenario_id": trace.scenario_id,
            "calls": [(c.name, c.arguments, str(c.result)) for c in trace.calls],
            "terminated": trace.terminated,
            "final_answer": trace.final_answer,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _sample_cache_key(scenario: Scenario, trace: Trajectory, judge: LLMJudge, n: int) -> str:
    model = getattr(judge, "model_name", getattr(judge, "judge_model", str(type(judge).__name__)))
    prompt_hash = hashlib.sha256(scenario.prompt.encode()).hexdigest()[:8]
    return f"{scenario.id}:{_trace_hash(trace)}:{model}:{JUDGE_PROMPT_VERSION}:{n}:{prompt_hash}"


def clear_judge_cache() -> None:
    """Clear both single and sampled judge caches (useful for tests)."""
    _SAMPLE_CACHE.clear()
    _SINGLE_CACHE.clear()


def _evict_if_needed(cache: dict) -> None:
    if len(cache) > _MAX_CACHE_SIZE:
        # Remove oldest (first inserted) – dict preserves insertion order PY3.7+
        oldest = next(iter(cache))
        cache.pop(oldest)


class JudgeResult(BaseModel):
    scenario_id: str
    score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Normalized 0..1 goal satisfaction"
    )
    label: str = "UNAVAILABLE"  # SATISFIED | PARTIAL | FAILED | UNAVAILABLE
    rationale: str = ""
    judge_model: str = ""


class SampledJudgeResult(BaseModel):
    """N-shot judge output: a mean score plus the spread across samples.

    A single LLM-judge call is noisy. Sampling N times and reporting the mean,
    min/max, and label agreement makes the signal defensible enough to opt into
    gating on it.
    """

    scenario_id: str
    score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Mean goal satisfaction across samples"
    )
    label: str = "UNAVAILABLE"
    score_min: float | None = None
    score_max: float | None = None
    samples: int = 0
    agreement: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of samples with the majority label"
    )
    judge_model: str = ""
    rationales: list[str] = Field(default_factory=list)


class LLMJudge(Protocol):
    def judge(self, scenario: Scenario, trace: Trajectory) -> JudgeResult:
        ...


def sample_judge(
    scenario: Scenario,
    trace: Trajectory,
    judge: LLMJudge,
    n: int = 3,
    use_cache: bool = True,
) -> SampledJudgeResult:
    """Run the judge ``n`` times and aggregate into a :class:`SampledJudgeResult`.

    When ``use_cache`` is True (default), results are cached by
    ``(scenario.id, trace hash, judge model, prompt version, n)``. Repeated
    evaluations of the same trace (e.g., CI re-runs) return instantly without
    LLM calls. Cache is in-memory LRU (128 entries) and can be cleared via
    :func:`clear_judge_cache`.
    """
    if use_cache:
        key = _sample_cache_key(scenario, trace, judge, n)
        if key in _SAMPLE_CACHE:
            return _SAMPLE_CACHE[key]
    results = [judge.judge(scenario, trace) for _ in range(n)]
    scores = [r.score for r in results if r.score is not None]
    labels = [r.label for r in results]
    if scores:
        mean = sum(scores) / len(scores)
        score_min, score_max = min(scores), max(scores)
        label = label_for(mean)
    else:
        mean = score_min = score_max = None
        label = "UNAVAILABLE"
    if labels:
        majority = max(set(labels), key=labels.count)
        agreement = labels.count(majority) / len(labels)
    else:
        agreement = 0.0
    out = SampledJudgeResult(
        scenario_id=scenario.id,
        score=mean,
        label=label,
        score_min=score_min,
        score_max=score_max,
        samples=len(results),
        agreement=agreement,
        judge_model=results[0].judge_model if results else "",
        rationales=[r.rationale for r in results],
    )
    if use_cache:
        _SAMPLE_CACHE[key] = out
        _evict_if_needed(_SAMPLE_CACHE)
    return out


def _summarize_result(result: Any, max_chars: int = 500) -> str:
    """Compact, truncated, deterministic serialization of a tool-call result.

    The judge must be *grounded* on what the tools actually returned, not just
    on the fact that a tool was called. This renders an arbitrary result value
    (dict, list, str, scalar, None) into a bounded string so a huge payload
    (e.g. an emitted row) can't blow up the prompt.
    """
    if result is None:
        return "(no result)"
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=lambda o: str(o))
        except (TypeError, ValueError):
            text = str(result)
    if len(text) > max_chars:
        return text[:max_chars] + f"... (+{len(text) - max_chars} chars truncated)"
    return text


def build_judge_prompt(scenario: Scenario, trace: Trajectory, max_total_chars: int = 8000) -> str:
    """Build the judge prompt with global budget and scenario grounding.

    - Injects ``scenario.description`` when present (P1 fix: was missing).
    - Caps per-result at 500 chars via ``_summarize_result`` and total
      ``call -> result`` segment at ``max_total_chars`` (P1 global budget).
    """
    if trace.calls:
        calls = "; ".join(
            f'{c.name}{(json.dumps(c.arguments) if c.arguments else "")}'
            f" -> {_summarize_result(c.result)}"
            for c in trace.calls
        )
        if len(calls) > max_total_chars:
            calls = calls[:max_total_chars] + f"... (+{len(calls) - max_total_chars} chars truncated, {len(trace.calls)} calls)"
    else:
        calls = "(no tool calls)"
    answer = trace.final_answer if trace.final_answer else "(no final answer produced)"
    desc = f"\nSCENARIO DESCRIPTION: {scenario.description}" if scenario.description else ""
    return f"""You evaluate whether an AI agent's final answer satisfies a user's request.

USER REQUEST: {scenario.prompt}{desc}

AGENT TOOL CALLS (call -> result): {calls}

AGENT FINAL ANSWER: "{answer}"

Did the agent satisfy the user's request, given both the tools it called AND the
results those tools actually returned? If a tool returned an error or empty
result but the final answer asserts it succeeded, score accordingly.
Respond with ONLY a JSON object:
{{"score": 0.0 to 1.0, "rationale": "one sentence explaining your score"}}"""


def parse_judge_response(text: str) -> tuple[float | None, str]:
    """Parse ``{"score": ..., "rationale": ...}`` out of a model response.

    Tolerant to markdown fences, prose around the JSON, and 0-10 scales.
    Returns (score_normalized_0_1 | None, rationale).
    """
    if not text:
        return None, ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            score = obj.get("score")
            rationale = str(obj.get("rationale", ""))
            if isinstance(score, (int, float)):
                score = float(score)
                if score > 1.0:
                    score = score / 10.0
                score = max(0.0, min(1.0, score))
                return score, rationale
        except (ValueError, AttributeError):
            pass
    m = re.search(r"(?<!\d)(0(?:\.\d+)?|1\.0|10)(?!\d)", text)
    if m:
        score = float(m.group(1))
        if score > 1.0:
            score = score / 10.0
        return max(0.0, min(1.0, score)), ""
    return None, ""


def label_for(score: float | None) -> str:
    if score is None:
        return "UNAVAILABLE"
    if score >= 0.8:
        return "SATISFIED"
    if score >= 0.5:
        return "PARTIAL"
    return "FAILED"


class OllamaJudge:
    """An LLM judge backed by an Ollama model (lazy dependency import).

    No fixed ``seed`` by default: N-shot sampling needs *varied* samples to be
    reliable, so the judge is deliberately non-deterministic across calls.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        num_predict: int = 256,
        seed: int | None = None,
    ):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OllamaJudge requires langchain-ollama; install with `pip install tracegate[ollama]`"
            ) from exc
        self._model = ChatOllama(
            model=model, temperature=temperature, num_predict=num_predict, seed=seed
        )
        self.model_name = model

    def judge(self, scenario: Scenario, trace: Trajectory) -> JudgeResult:
        try:
            response = self._model.invoke(build_judge_prompt(scenario, trace))
        except Exception:  # noqa: BLE001 - degrade, never crash the run
            return JudgeResult(scenario_id=scenario.id, label="UNAVAILABLE", judge_model=self.model_name)
        score, rationale = parse_judge_response(str(response.content))
        return JudgeResult(
            scenario_id=scenario.id,
            score=score,
            label=label_for(score),
            rationale=rationale,
            judge_model=self.model_name,
        )


class OpenAICompatibleJudge:
    """Judge that calls any OpenAI-compatible ``/chat/completions`` endpoint.

    Pure-stdlib HTTP client (``urllib``) — no SDK dependency, so it works from
    the core install with OpenAI, Groq, Together, vLLM, LM Studio, or Ollama's
    OpenAI-compat endpoint. The judge layer is provider-neutral by construction:
    swap ``OllamaJudge`` for ``OpenAICompatibleJudge`` and nothing else changes.

    Fail-closed: network/auth/parse errors degrade to UNAVAILABLE, never crash.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
        timeout: float = 30.0,
    ):
        self.model_name = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout

    def _payload(self, scenario: Scenario, trace: Trajectory) -> dict:
        payload: dict = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a rigorous evaluator of AI agent goal satisfaction.",
                },
                {"role": "user", "content": build_judge_prompt(scenario, trace)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:  # omit for providers that reject the param
            payload["seed"] = self.seed
        return payload

    def _call(self, payload: dict) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        return obj["choices"][0]["message"]["content"]

    def judge(self, scenario: Scenario, trace: Trajectory) -> JudgeResult:
        try:
            content = self._call(self._payload(scenario, trace))
        except Exception:  # noqa: BLE001 - degrade, never crash the run
            return JudgeResult(
                scenario_id=scenario.id, label="UNAVAILABLE", judge_model=self.model_name
            )
        score, rationale = parse_judge_response(content)
        return JudgeResult(
            scenario_id=scenario.id,
            score=score,
            label=label_for(score),
            rationale=rationale,
            judge_model=self.model_name,
        )


class CachedJudge:
    """Wraps any ``LLMJudge`` with an in-memory cache (P2 cost reduction).

    Single-call cache keyed by ``(scenario.id, trace hash, model, prompt version)``.
    For N-shot, use :func:`sample_judge`'s own cache (which is higher-level).
    This wrapper is useful when the same trace is judged repeatedly outside
    ``sample_judge`` (e.g., gate re-runs).

    Example::

        judge = CachedJudge(OpenAICompatibleJudge(model="gpt-4o-mini"))
        judge.judge(scenario, trace)  # LLM call
        judge.judge(scenario, trace)  # cached, no LLM call
    """

    def __init__(self, inner: LLMJudge, maxsize: int = 128):
        self.inner = inner
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        # Expose model_name for cache-key compatibility with sample_judge
        self.model_name = getattr(inner, "model_name", getattr(inner, "judge_model", str(type(inner).__name__)))

    def judge(self, scenario: Scenario, trace: Trajectory) -> JudgeResult:
        key = f"{scenario.id}:{_trace_hash(trace)}:{self.model_name}:{JUDGE_PROMPT_VERSION}:{hashlib.sha256(scenario.prompt.encode()).hexdigest()[:8]}"
        if key in _SINGLE_CACHE:
            self.hits += 1
            return _SINGLE_CACHE[key]
        self.misses += 1
        res = self.inner.judge(scenario, trace)
        _SINGLE_CACHE[key] = res
        _evict_if_needed(_SINGLE_CACHE)
        return res

    def clear(self) -> None:
        _SINGLE_CACHE.clear()
        self.hits = self.misses = 0


def build_judge(
    provider: str = "ollama",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> LLMJudge:
    """Factory for the built-in judge backends.

    ``ollama`` : local model via langchain-ollama (the demo default).
    ``api``    : any OpenAI-compatible ``/chat/completions`` endpoint via the
                 stdlib-only :class:`OpenAICompatibleJudge`.

    The two backends are interchangeable behind the ``LLMJudge`` seam.
    """
    p = provider.lower()
    if p == "ollama":
        return OllamaJudge(model=model or "llama3.1:8b", temperature=temperature)
    if p in ("api", "openai"):
        return OpenAICompatibleJudge(
            model=model or "gpt-4o-mini",
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=temperature,
        )
    raise ValueError(
        f"unknown judge provider {provider!r}; expected 'ollama' or 'api'"
    )


__all__ = [
    "JudgeResult",
    "LLMJudge",
    "OllamaJudge",
    "OpenAICompatibleJudge",
    "SampledJudgeResult",
    "CachedJudge",
    "JUDGE_PROMPT_VERSION",
    "build_judge",
    "build_judge_prompt",
    "_summarize_result",
    "label_for",
    "parse_judge_response",
    "sample_judge",
    "clear_judge_cache",
]
