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

import json
import os
import re
import urllib.request
from typing import Protocol

from pydantic import BaseModel, Field

from tracegate.schema import Scenario, Trajectory


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
) -> SampledJudgeResult:
    """Run the judge ``n`` times and aggregate into a :class:`SampledJudgeResult`."""
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
    return SampledJudgeResult(
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


def build_judge_prompt(scenario: Scenario, trace: Trajectory) -> str:
    if trace.calls:
        calls = "; ".join(
            f"{c.name}({json.dumps(c.arguments) if c.arguments else ''})" for c in trace.calls
        )
    else:
        calls = "(no tool calls)"
    answer = trace.final_answer if trace.final_answer else "(no final answer produced)"
    return f"""You evaluate whether an AI agent's final answer satisfies a user's request.

USER REQUEST: {scenario.prompt}

AGENT TOOL CALLS: {calls}

AGENT FINAL ANSWER: "{answer}"

Did the agent satisfy the user's request? Respond with ONLY a JSON object:
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
    "build_judge",
    "build_judge_prompt",
    "label_for",
    "parse_judge_response",
    "sample_judge",
]
