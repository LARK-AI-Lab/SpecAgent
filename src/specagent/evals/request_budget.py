"""Crash-safe request-count budgets shared by live evaluation processes."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from specagent.protocol.accounting import RESOURCE_SCHEMA_VERSION

IDENTITY_METADATA_KEYS = (
    "logical_request_id",
    "transport_attempt_id",
    "transport_attempt",
    "parent_request_id",
    "parent_task_id",
    "parent_episode_id",
    "canonical_phase",
    "model",
    "request_sha256",
)


class ExternalRequestBudgetExceeded(RuntimeError):
    pass


class ExternalRequestBudget:
    """Reserve one counted external attempt before dispatching it.

    The ledger intentionally counts attempts, not successful responses or billable requests.
    Each reservation is fsynced under an exclusive lock, so retries and concurrent processes
    cannot cross the configured ceiling.
    """

    def __init__(self, path: Path, limit: int) -> None:
        if limit < 1:
            raise ValueError("external request limit must be positive")
        self.path = path
        self.limit = limit

    def reserve(
        self,
        *,
        phase: str,
        metadata: dict[str, Any] | None = None,
        phase_limit: int | None = None,
    ) -> int:
        if not phase.strip():
            raise ValueError("external request phase must be non-empty")
        if phase_limit is not None and phase_limit < 1:
            raise ValueError("external request phase limit must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            events = [json.loads(line) for line in stream if line.strip()]
            limits = [event.get("limit") for event in events if event.get("event") == "limits"]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in limits):
                raise ValueError("external request ledger contains an invalid limit")
            effective_limit = min([self.limit, *limits]) if limits else self.limit
            stream.seek(0, os.SEEK_END)
            limits_changed = False
            if not limits or self.limit < min(limits):
                stream.write(json.dumps({"event": "limits", "limit": self.limit}) + "\n")
                limits_changed = True
            recorded_phase_limits = [
                event.get("limit")
                for event in events
                if event.get("event") == "phase_limits" and event.get("phase") == phase
            ]
            # Read ledgers created before phase_limits had its own event type.
            recorded_phase_limits.extend(
                event.get("phase_limit")
                for event in events
                if event.get("event") == "reserved"
                and event.get("phase") == phase
                and event.get("phase_limit") is not None
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in recorded_phase_limits
            ):
                raise ValueError("external request ledger contains an invalid phase limit")
            if phase_limit is not None and (
                not recorded_phase_limits or phase_limit < min(recorded_phase_limits)
            ):
                stream.write(
                    json.dumps(
                        {"event": "phase_limits", "phase": phase, "limit": phase_limit}
                    )
                    + "\n"
                )
                limits_changed = True
            if limits_changed:
                stream.flush()
                os.fsync(stream.fileno())
            reservations = sum(event.get("event") == "reserved" for event in events)
            if reservations >= effective_limit:
                raise ExternalRequestBudgetExceeded(
                    f"external request budget exhausted ({reservations}/{effective_limit})"
                )
            effective_phase_limit = (
                min([phase_limit, *recorded_phase_limits])
                if phase_limit is not None
                else min(recorded_phase_limits, default=None)
            )
            phase_reservations = sum(
                event.get("event") == "reserved" and event.get("phase") == phase
                for event in events
            )
            if (
                effective_phase_limit is not None
                and phase_reservations >= effective_phase_limit
            ):
                raise ExternalRequestBudgetExceeded(
                    "external request phase budget exhausted "
                    f"({phase}:{phase_reservations}/{effective_phase_limit})"
                )
            sequence = reservations + 1
            event = {
                "event": "reserved",
                "sequence": sequence,
                "phase": phase,
                "metadata": dict(metadata or {}),
                "schema_version": RESOURCE_SCHEMA_VERSION,
            }
            if effective_phase_limit is not None:
                event["phase_limit"] = effective_phase_limit
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            return sequence

    def used(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open(encoding="utf-8") as stream:
            return sum(
                json.loads(line).get("event") == "reserved"
                for line in stream
                if line.strip()
            )

    def reservations(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") == "reserved":
                    rows.append(event)
        return rows


def identity_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy known identity fields only; missing keys stay missing, not invented."""
    if not metadata:
        return {}
    return {key: metadata[key] for key in IDENTITY_METADATA_KEYS if key in metadata}
