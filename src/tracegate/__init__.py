from tracegate.schema import (
    ExpectedToolCall,
    Scenario,
    ToolCall,
    ToolRegistry,
    TraceGateError,
    Trajectory,
)
from tracegate.metrics import (
    ComponentScores,
    MetricPlugin,
    MetricRegistry,
    registry,
    score_trace,
)
from tracegate.runner import run_scenario

__all__ = [
    "ExpectedToolCall",
    "Scenario",
    "ToolCall",
    "ToolRegistry",
    "TraceGateError",
    "Trajectory",
    "ComponentScores",
    "MetricPlugin",
    "MetricRegistry",
    "registry",
    "score_trace",
    "run_scenario",
]

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("tracegate")
except Exception:  # pragma: no cover — fallback for editable installs without metadata
    __version__ = "0.1.0"
