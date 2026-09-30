"""Typed records shared by the prefix-commit protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

PREFIX_VERDICT_SCHEMA_VERSION = "prefix_verdict_v1"


class ModelRole(str, Enum):
    WEAK = "weak"
    STRONG = "strong"


class SessionPhase(str, Enum):
    READY = "ready"
    DRAFTING = "drafting"
    AWAITING_VERDICT = "awaiting_verdict"
    AWAITING_CORRECTION = "awaiting_correction"
    COMPLETE = "complete"


class RejectionCode(str, Enum):
    INVALID_ACTION = "invalid_action"
    WRONG_STEP = "wrong_step"
    BROKEN_STATE = "broken_state"
    PREMATURE_TERMINAL = "premature_terminal"
    OTHER = "other"


class RuntimeErrorCode(str, Enum):
    FORMAT_ERROR = "format_error"
    INCOMPLETE_GENERATION = "incomplete_generation"
    REPLAY_DIVERGENCE = "replay_divergence"
    MISSING_OBSERVATION = "missing_observation"


@dataclass(frozen=True, slots=True)
class TaskContext:
    task_id: str
    instruction: str
    system_constraints: tuple[str, ...] = ()
    tool_contracts: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentAction:
    action: Any
    reasoning: str | None = None
    output_tokens: int = 0
    duration_seconds: float = 0.0
    metadata: Mapping[str, Any] | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        if self.output_tokens < 0 or self.input_tokens < 0 or self.cached_input_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.reasoning_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One executed reasoning-action-observation transition."""

    actor: ModelRole
    action: Any
    observation: Any
    reasoning: str | None = None
    output_tokens: int = 0
    terminal: bool = False
    duration_seconds: float = 0.0
    tool_duration_seconds: float = 0.0
    metadata: Mapping[str, Any] | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.output_tokens,
                self.input_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
            )
        ):
            raise ValueError("token counts must be non-negative")
        if self.duration_seconds < 0 or self.tool_duration_seconds < 0:
            raise ValueError("durations must be non-negative")


@dataclass(frozen=True, slots=True)
class DraftHandle:
    base_revision: int
    horizon: int


@dataclass(frozen=True, slots=True)
class PrefixVerdict:
    accepted_steps: int
    rejection_code: RejectionCode | None = None
    confidence: float = 0.0
    # One short sentence: which step failed, what is materially wrong, and what a correct action
    # must change. Fed to the corrector so it repairs the rejected step instead of re-planning.
    reason: str = ""
    # Who should redraft the rejected step: "weak" when the reason is a mechanical instruction
    # the drafting model can follow (do a prerequisite first, fix an argument, drop a claim),
    # "strong" when the fix needs judgement over facts the draft got wrong. None when the
    # verifier did not say; the runtime then falls back to its boundary rule.
    corrector: str | None = None
    # Optional Strong-authored action that replaces the first rejected step. Runtimes may execute
    # it only after validating it against the tools in the current request; otherwise they keep
    # the existing separate correction call.
    replacement_action: Mapping[str, Any] | None = None
    # Evidence-ready mode only: the verifier judged that the committed evidence already answers
    # the question, so Strong writes the final answer and Weak stops. False when not asked.
    answer_ready: bool = False
    schema_version: str = PREFIX_VERDICT_SCHEMA_VERSION
    raw: str = ""
    verifier_duration_seconds: float = 0.0
    verifier_input_tokens: int = 0
    verifier_output_tokens: int = 0
    verifier_cached_input_tokens: int = 0
    verifier_reasoning_tokens: int = 0
    # Runtime diagnostics are not additional fields required in the model's JSON verdict.
    diagnostic_status: str = "model_verdict"
    verification_id: str = ""
    format_retries: int = 0
    # When diagnostic_status == "format_failure", the parser's classification:
    # "missing_keys", "invalid_accepted_steps", "invalid_replacement",
    # "foreign_keys", "unparseable", "truncated", or "undetermined".
    format_failure_kind: str = ""
    # Safe-salvage audit. Fields the parser dropped because they cannot affect the
    # accepted prefix or the reason (foreign keys under a fixed contract, invalid
    # optional values, a smuggled replacement under a non-replacement contract).
    discarded_verdict_fields: tuple[str, ...] = ()
    # A stray reason/replacement emitted on a full accept is canonicalized to
    # null and recorded here instead of costing a format retry.
    reason_on_full_accept: bool = False
    replacement_on_full_accept: bool = False
    # Why a non-null replacement_action was unusable and discarded ("" = kept or
    # absent): "message_replacement", "malformed_replacement", or a runtime
    # guard kind set by the gateway ("oversized_replacement", "unknown_tool",
    # "schema_invalid", "replays_rejected", "mutation_replay"). A discarded
    # replacement never enters the corrector prompt and never executes.
    replacement_validation_failure_kind: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PREFIX_VERDICT_SCHEMA_VERSION:
            raise ValueError("unsupported prefix verdict schema")
        if self.accepted_steps < 0:
            raise ValueError("accepted_steps must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrajectoryDraft:
    base_revision: int
    steps: tuple[AgentStep, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentCheckpoint:
    checkpoint_id: str
    revision: int
    state_hash: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentBranch:
    branch_id: str
    base: EnvironmentCheckpoint


@dataclass(frozen=True, slots=True)
class CommitResult:
    accepted: tuple[AgentStep, ...]
    rejected: tuple[AgentStep, ...]
    correction_required: bool
    revision: int
