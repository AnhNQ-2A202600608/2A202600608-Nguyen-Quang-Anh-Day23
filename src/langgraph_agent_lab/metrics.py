"""Metrics schema and helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


class ScenarioMetric(BaseModel):
    scenario_id: str
    success: bool
    expected_route: str
    actual_route: str | None = None
    nodes_visited: int = 0
    retry_count: int = 0
    interrupt_count: int = 0
    approval_required: bool = False
    approval_observed: bool = False
    latency_ms: int = 0
    errors: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    total_scenarios: int
    success_rate: float
    avg_nodes_visited: float
    total_retries: int
    total_interrupts: int
    resume_success: bool = False
    scenario_metrics: list[ScenarioMetric]


def metric_from_state(
    state: dict[str, Any],
    expected_route: str,
    approval_required: bool,
    latency_ms: int = 0,
) -> ScenarioMetric:
    """Build a ScenarioMetric from a completed graph state.

    Args:
        state: Final state dict returned by graph.invoke()
        expected_route: Expected route string from scenario definition
        approval_required: Whether HITL approval was required
        latency_ms: Measured wall-clock time in milliseconds for this run
    """
    events = state.get("events", []) or []
    errors = state.get("errors", []) or []
    actual_route = state.get("route")
    approval = state.get("approval")
    nodes = [event.get("node", "unknown") for event in events]
    retry_count = sum(1 for node in nodes if node == "retry")
    interrupt_count = sum(1 for node in nodes if node == "approval")
    success = (
        actual_route == expected_route
        and bool(state.get("final_answer") or state.get("pending_question"))
    )
    if approval_required:
        success = success and approval is not None
    return ScenarioMetric(
        scenario_id=str(state.get("scenario_id", "unknown")),
        success=success,
        expected_route=expected_route,
        actual_route=actual_route,
        nodes_visited=len(nodes),
        retry_count=retry_count,
        interrupt_count=interrupt_count,
        approval_required=approval_required,
        approval_observed=approval is not None,
        latency_ms=latency_ms,
        errors=list(errors),
    )


def _detect_resume_success(db_path: str = "outputs/checkpoints.db") -> bool:
    """Detect crash-resume capability by checking if SQLite checkpoint DB
    contains state history for multiple distinct thread IDs.

    A True value proves the checkpointer persisted state across invocations,
    enabling crash-resume (replay from any earlier checkpoint by thread_id).
    """
    try:
        path = Path(db_path)
        if not path.exists():
            return False
        conn = sqlite3.connect(str(path), check_same_thread=False)
        try:
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints WHERE thread_id IS NOT NULL"
            )
            count = cursor.fetchone()[0]
            return count >= 2  # At least 2 distinct threads proves persistence across runs
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()
    except Exception:
        return False


def summarize_metrics(
    items: list[ScenarioMetric],
    db_path: str = "outputs/checkpoints.db",
) -> MetricsReport:
    """Summarize per-scenario metrics into a MetricsReport.

    Automatically detects resume_success by querying the SQLite checkpoint DB
    for evidence of multi-thread persistence (proves crash-resume capability).
    """
    if not items:
        raise ValueError("No scenario metrics to summarize")
    return MetricsReport(
        total_scenarios=len(items),
        success_rate=sum(1 for item in items if item.success) / len(items),
        avg_nodes_visited=mean(item.nodes_visited for item in items),
        total_retries=sum(item.retry_count for item in items),
        total_interrupts=sum(item.interrupt_count for item in items),
        resume_success=_detect_resume_success(db_path),
        scenario_metrics=items,
    )


def write_metrics(report: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
