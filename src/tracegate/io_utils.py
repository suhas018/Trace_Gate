"""YAML/JSON loading for scenario suites, tool registries, and behavior configs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from tracegate.agent import AgentBehavior
from tracegate.schema import Scenario, ToolRegistry, TraceGateError


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise TraceGateError(f"file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TraceGateError(f"expected a mapping at top level of {p}")
    return data


def load_scenario_suite(path: str | Path) -> tuple[list[Scenario], ToolRegistry | None]:
    """Load a suite file::

        registry:
          tools:
            - {name: get_orders, description: "...", dangerous: false}
        scenarios:
          - id: sc_001
            prompt: "..."
            ...
    """
    data = _load_yaml(path)

    registry: ToolRegistry | None = None
    reg = data.get("registry")
    if reg is not None:
        try:
            registry = ToolRegistry.model_validate(reg)
        except ValidationError as exc:
            raise TraceGateError(f"invalid registry in {path}: {exc}") from exc

    raw_scenarios = data.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise TraceGateError(f"{path}: no 'scenarios' list found")
    scenarios: list[Scenario] = []
    for idx, item in enumerate(raw_scenarios):
        if not isinstance(item, dict):
            raise TraceGateError(f"{path}: scenario #{idx} is not a mapping")
        if "id" not in item:
            raise TraceGateError(f"{path}: scenario #{idx} is missing an 'id'")
        try:
            scenarios.append(Scenario.model_validate(item))
        except ValidationError as exc:
            raise TraceGateError(f"{path}: scenario {item.get('id', idx)!r} invalid: {exc}") from exc
    return scenarios, registry


def load_behavior_config(path: str | Path | None) -> dict[str, AgentBehavior]:
    """Load a behavior map::

        sc_001: {plan: [get_orders, compute_total], terminate: true}
    """
    if path is None:
        return {}
    data = _load_yaml(path)
    behaviors: dict[str, AgentBehavior] = {}
    for scenario_id, raw in data.items():
        if raw is None:
            behaviors[scenario_id] = AgentBehavior()
            continue
        try:
            behaviors[scenario_id] = AgentBehavior.model_validate(raw)
        except ValidationError as exc:
            raise TraceGateError(
                f"behavior for {scenario_id!r} invalid: {exc}"
            ) from exc
    return behaviors


def save_json(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
