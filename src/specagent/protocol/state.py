"""Pure state machine for longest-prefix commitment and one-step correction."""

from __future__ import annotations

from dataclasses import replace

from specagent.protocol.types import (
    AgentStep,
    CommitResult,
    DraftHandle,
    ModelRole,
    PrefixVerdict,
    SessionPhase,
)


class ProtocolStateError(RuntimeError):
    pass


class PrefixCommitSession:
    """Own the canonical trajectory; environment branches stay in an adapter."""

    def __init__(self, history: tuple[AgentStep, ...] = ()) -> None:
        self._history = list(history)
        self._revision = len(history)
        self._phase = (
            SessionPhase.COMPLETE if history and history[-1].terminal else SessionPhase.READY
        )
        self._draft: list[AgentStep] = []
        self._horizon: int | None = None

    @property
    def history(self) -> tuple[AgentStep, ...]:
        return tuple(self._history)

    @property
    def draft(self) -> tuple[AgentStep, ...]:
        return tuple(self._draft)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    def begin_draft(self, horizon: int) -> DraftHandle:
        if self._phase is not SessionPhase.READY:
            raise ProtocolStateError(f"cannot begin a draft while phase={self._phase.value}")
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        self._draft = []
        self._horizon = horizon
        self._phase = SessionPhase.DRAFTING
        return DraftHandle(base_revision=self._revision, horizon=horizon)

    def append_weak_step(self, step: AgentStep) -> None:
        if self._phase is not SessionPhase.DRAFTING:
            raise ProtocolStateError("weak steps can only be appended while drafting")
        if step.actor is not ModelRole.WEAK:
            raise ValueError("a speculative draft may only contain Weak steps")
        if self._draft and self._draft[-1].terminal:
            raise ProtocolStateError("cannot append after a terminal step")
        if self._horizon is None or len(self._draft) >= self._horizon:
            raise ProtocolStateError("draft horizon exceeded")
        self._draft.append(step)

    def seal_draft(self) -> tuple[AgentStep, ...]:
        if self._phase is not SessionPhase.DRAFTING:
            raise ProtocolStateError("no active draft to seal")
        if not self._draft:
            raise ProtocolStateError("cannot verify an empty draft")
        self._phase = SessionPhase.AWAITING_VERDICT
        return tuple(self._draft)

    def apply_verdict(self, verdict: PrefixVerdict) -> CommitResult:
        if self._phase is not SessionPhase.AWAITING_VERDICT:
            raise ProtocolStateError("a verdict is only valid for a sealed draft")
        if verdict.accepted_steps > len(self._draft):
            raise ValueError("accepted prefix exceeds the candidate trajectory")
        if verdict.accepted_steps == len(self._draft) and verdict.rejection_code is not None:
            raise ValueError("full acceptance requires a null rejection_code")
        if verdict.accepted_steps < len(self._draft) and verdict.rejection_code is None:
            raise ValueError("a rejected suffix requires a rejection_code")

        accepted = tuple(self._draft[: verdict.accepted_steps])
        rejected = tuple(self._draft[verdict.accepted_steps :])
        self._history.extend(accepted)
        self._revision += len(accepted)
        self._draft = []
        self._horizon = None

        if accepted and accepted[-1].terminal:
            self._phase = SessionPhase.COMPLETE
            correction_required = False
        elif rejected:
            self._phase = SessionPhase.AWAITING_CORRECTION
            correction_required = True
        else:
            self._phase = SessionPhase.READY
            correction_required = False

        return CommitResult(
            accepted=accepted,
            rejected=rejected,
            correction_required=correction_required,
            revision=self._revision,
        )

    def commit_correction(self, step: AgentStep) -> None:
        if self._phase is not SessionPhase.AWAITING_CORRECTION:
            raise ProtocolStateError("no Strong correction is currently required")
        if step.actor is not ModelRole.STRONG:
            raise ValueError("the correction must be produced by the Strong model")
        self._commit_correction(step)

    def commit_routed_correction(self, step: AgentStep) -> None:
        """Commit the one correction authorized by an adaptive router.

        Unlike the historical Strong-only fallback, a routed correction may be
        produced by either model.  The router verdict is the semantic review;
        this method does not reopen or re-verify the correction.
        """
        if self._phase is not SessionPhase.AWAITING_CORRECTION:
            raise ProtocolStateError("no routed correction is currently required")
        if step.actor not in (ModelRole.WEAK, ModelRole.STRONG):
            raise ValueError("a routed correction must be produced by Weak or Strong")
        self._commit_correction(step)

    def _commit_correction(self, step: AgentStep) -> None:
        self._history.append(step)
        self._revision += 1
        self._phase = SessionPhase.COMPLETE if step.terminal else SessionPhase.READY

    def bind_last_draft_observation(
        self,
        observation: object,
        terminal: bool,
        metadata: dict | None = None,
    ) -> AgentStep:
        """Bind the real observation for an already-executed draft interaction.

        Paper-style observed-K adapters keep customer replies in the candidate
        until the simulator answers.  The reply is already externally visible;
        this only completes its action-observation pair before verification.
        """
        if self._phase is not SessionPhase.DRAFTING or not self._draft:
            raise ProtocolStateError("no active draft step to bind")
        last = self._draft[-1]
        if last.observation is not None:
            raise ProtocolStateError("the last draft step already has an observation")
        extra = dict(last.metadata or {})
        extra.pop("pending", None)
        if metadata:
            extra.update(metadata)
        bound = replace(
            last,
            observation=observation,
            terminal=terminal,
            metadata=extra or None,
        )
        self._draft[-1] = bound
        return bound

    def reopen_for_redraft(self) -> None:
        """Return to drafting instead of demanding a correction from Strong.

        apply_verdict has already committed the accepted prefix and dropped the
        rejected suffix; this only says the replacement will be drafted again
        rather than corrected. The redraft is sealed and verified like any other
        chunk, so nothing reaches history without a verdict -- the runtime is
        choosing who writes the replacement, not skipping verification.
        """
        if self._phase is not SessionPhase.AWAITING_CORRECTION:
            raise ProtocolStateError("only a rejected draft can be reopened for a redraft")
        self._phase = SessionPhase.READY

    def bind_last_observation(
        self,
        observation: object,
        terminal: bool,
        metadata: dict | None = None,
    ) -> AgentStep:
        """Fill in the observation for a pending action that was executed after accept."""
        if not self._history:
            raise ProtocolStateError("no committed step to bind")
        last = self._history[-1]
        extra = dict(last.metadata or {})
        extra.pop("pending", None)
        if metadata:
            extra.update(metadata)
        tool_dur = last.tool_duration_seconds
        if metadata and "tool_duration_seconds" in metadata:
            tool_dur = float(metadata["tool_duration_seconds"])
        bound = replace(
            last,
            observation=observation,
            terminal=terminal,
            tool_duration_seconds=tool_dur,
            metadata=extra or None,
        )
        self._history[-1] = bound
        if terminal:
            self._phase = SessionPhase.COMPLETE
        return bound

    def abort_draft(self) -> tuple[AgentStep, ...]:
        if self._phase not in (SessionPhase.DRAFTING, SessionPhase.AWAITING_VERDICT):
            raise ProtocolStateError("no draft can be aborted in the current phase")
        discarded = tuple(self._draft)
        self._draft = []
        self._horizon = None
        self._phase = SessionPhase.READY
        return discarded
