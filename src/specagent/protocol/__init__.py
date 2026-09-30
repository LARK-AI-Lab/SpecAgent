"""Current training-free, Weak-led trajectory speculation protocol."""

from specagent.protocol.accounting import (
    InferencePhase,
    InferenceSpan,
    ResourceLedger,
    ResourceSummary,
)
from specagent.protocol.context import (
    ContextPolicy,
    committed_context,
    correction_payload,
    verification_payload,
)
from specagent.protocol.side_effects import action_mutates_state
from specagent.protocol.state import PrefixCommitSession, ProtocolStateError
from specagent.protocol.types import (
    PREFIX_VERDICT_SCHEMA_VERSION,
    AgentAction,
    AgentStep,
    CommitResult,
    DraftHandle,
    EnvironmentBranch,
    EnvironmentCheckpoint,
    ModelRole,
    PrefixVerdict,
    RejectionCode,
    RuntimeErrorCode,
    SessionPhase,
    TaskContext,
    TrajectoryDraft,
)
from specagent.protocol.verifier import parse_prefix_verdict, verifier_messages

__all__ = [
    "AgentAction",
    "AgentStep",
    "CommitResult",
    "ContextPolicy",
    "DraftHandle",
    "EnvironmentBranch",
    "EnvironmentCheckpoint",
    "InferencePhase",
    "InferenceSpan",
    "ModelRole",
    "PREFIX_VERDICT_SCHEMA_VERSION",
    "PrefixCommitSession",
    "PrefixVerdict",
    "RejectionCode",
    "ProtocolStateError",
    "ResourceLedger",
    "ResourceSummary",
    "RuntimeErrorCode",
    "SessionPhase",
    "TaskContext",
    "TrajectoryDraft",
    "action_mutates_state",
    "committed_context",
    "correction_payload",
    "parse_prefix_verdict",
    "verification_payload",
    "verifier_messages",
]
