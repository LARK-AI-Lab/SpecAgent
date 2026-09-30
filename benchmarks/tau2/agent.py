"""tau2 agent implementing the SpecAgent prefix-commit (Weak/Strong) protocol.

Weak drafts up to `horizon` steps per chunk. Read-only tool calls execute
immediately (see Tau2DeferredExecutionGate.run_read). The first step that
cannot be taken back seals the chunk as a pending step that has NOT run: a
state-changing tool call (staged in the gate, including
transfer_to_human_agents) or a text reply to the customer. This mirrors the
"at most one trailing pending step per chunk" shape, which is the only shape
protocol.state.PrefixCommitSession.bind_last_observation supports (it can
only patch history[-1]).

Strong is asked to verify a chunk only when it ends in a staged tool call.
A reply is sent without review, and a chunk of `horizon` reads commits
without review; both are counted as auto-accepted. The exception is a
stuck Weak: once unreviewed chunks keep repeating themselves (see
SpecAgentTau2TCState.unverified_chunks), the next one goes to Strong.
Strong rules on the whole candidate once and returns
{reason, accepted_steps, replacement_action}:

  full accept, pending tool  -> execute it for real and return it
  full accept, reply         -> send it; the customer's next message becomes
                                its observation
  full accept, reads only    -> the horizon was hit; Weak keeps drafting in
                                the same turn (nothing is sent)
  reject at j                -> steps j.. are removed from the Weak prompt
                                (executed reads stay in tau2's trajectory
                                because they did happen, but they leave the
                                public history), the pending step never runs.
                                A schema-valid replacement_action executes
                                directly; otherwise Strong takes one repair
                                turn from the retained prefix.

The verifier sees the customer's opening request as `task.instruction` and
every later customer message as the observation of the reply it answered, so
its confirmed_history is the full dialogue.

tau2 gives agents no reference to the live Environment (see
tau2.runner.build.build_agent), so "execute a tool call for real" can only
happen through a bound callback the orchestrator hands to this agent -- the
gate. Mutation classification does NOT need that callback: it reads
tau2's own `__mutates_state__` marker straight off each Tool's underlying
function (self.tools is available to every agent).

KEY INVARIANT -- who appends what, and where. Every executed tool call must
land in exactly one place in each of the two logs (state.raw, the LLM
prompt; and orchestrator.trajectory, the eval record):

  read-only call   executed by gate.run_read, sealed inside the chunk and
                   never returned to the orchestrator. WE append it to both
                   state.raw and (by splicing) the trajectory.
  accepted pending / Strong repair
                   executed by gate.resolve / gate.run_resolved, then
                   RETURNED to the orchestrator. It routes the message to
                   ENV, take_resolved() hands back our already-computed
                   result instead of re-executing, and it feeds the
                   ToolMessage back to us next turn. So we append ONLY the
                   assistant message; appending the result here would
                   duplicate it in both logs.
  rejected pending never executed: we pop the proposing assistant message
                   back off state.raw so no tool_call is left dangling.
  replacement      same return-to-orchestrator path as a repair
                   (run_verifier_replacement + take_resolved), counted as
                   verifier_replacements, never as a Strong repair turn.

The orchestrator.py patch (apply_orchestrator_patch.py in this directory)
MUST be applied to tau2-bench or this agent will double-execute every
accepted mutating call.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from baseline_agent import AGENT_INSTRUCTION, FALLBACK_CONTENT, SYSTEM_PROMPT, message_to_api
from metrics_interface import (
    begin_episode,
    begin_logical_request,
    make_tau2_span,
    mark_span_recorded,
    record_failed_transport,
    record_result_span,
    span_id_kwargs,
    span_ids_from_agent,
    usage_fields_from_result,
)
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from openai_client import OpenAIChatClient
from pydantic import BaseModel
from reflective_prefix import (
    PROTOCOL_NAME,
    VERIFIER_CONTRACT,
    correction_messages,
    validate_replacement_action,
    verifier_messages,
)
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.environment.toolkit import MUTATES_STATE_ATTR

from specagent.evals.trial_shard_io import TrialIOError, configure_from_stats_env
from specagent.protocol.accounting import InferencePhase, ResourceLedger
from specagent.protocol.context import ContextPolicy, verification_payload
from specagent.protocol.evidence_review import prune_payload
from specagent.protocol.resource_estimate import paper_resources_from_tau_agent
from specagent.protocol.state import PrefixCommitSession
from specagent.protocol.types import (
    AgentStep,
    ModelRole,
    PrefixVerdict,
    RejectionCode,
    SessionPhase,
    TaskContext,
)
from specagent.protocol.verifier import parse_prefix_verdict

# A blip on one model call would otherwise fail the whole task: tau2 retries the
# entire simulation from scratch, throwing away every step taken so far.
# Retrying the single call is far cheaper.
_TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
_MAX_TRANSPORT_RETRIES = 4
_RETRY_BASE_DELAY = 2.0

# A read-only chunk produces no message for the orchestrator, so the turn has to
# draft again. Cap that: a Weak model that only ever reads would otherwise spin
# here forever, and the orchestrator's max_steps cannot see inside this call.
_MAX_CHUNKS_PER_TURN = 8

# Tools that are never run on Weak's word alone even though tau2 does not mark
# them as mutating: tau2's flag only tracks the DB, but to the customer a
# transfer is the end of the conversation.
_IRREVOCABLE_TOOL_NAMES = frozenset({"transfer_to_human_agents"})

# After this many consecutive unreviewed chunks that repeat one already
# committed, the next chunk goes to Strong. Progress does not repeat itself,
# so a Weak that is getting somewhere never reaches the budget.
_UNVERIFIED_REPEAT_BUDGET = 2

# How many recent unreviewed chunks a repeat is looked for among, and how much
# word overlap counts as "the same reply again". A stuck agent rewords itself
# every turn, so replies cannot be compared literally. The threshold is
# deliberately loose because the two errors are not symmetric: a false positive
# costs one Strong verification, while a false negative lets a livelock run to
# the step cap.
_UNVERIFIED_SIG_WINDOW = 6
_REPEAT_SIMILARITY = 0.6

# What reasoning the verifier payload is built with. Candidate-step reasoning is
# then pruned from the payload before it is sent (see _strong_verdict).
_CONTEXT_POLICY = ContextPolicy.MIXED_THINKING

MESSAGE_ACTION_TYPE = "message"


def _tool_mutates_state(tool: Tool) -> bool:
    return bool(getattr(getattr(tool, "_func", None), MUTATES_STATE_ATTR, True))


class Tau2DeferredExecutionGate:
    """Bound once per episode by the orchestrator.py patch. Lets the agent
    execute tool calls for real from inside generate_next_message (reads,
    always; the single staged mutating call, only after Strong accepts it)
    and tells the patched orchestrator when a returned message's tool
    results are already resolved so it does not re-execute them.
    """

    def __init__(self) -> None:
        self._execute: Callable[[list[ToolCall]], list[ToolMessage]] | None = None
        self._orchestrator: Any = None
        self._resolved: dict[str, ToolMessage] = {}
        self._pending: ToolCall | None = None
        self.stats: dict[str, int] = {
            "tool_calls_executed": 0,
            "reads_executed": 0,
            "corrections_executed": 0,
            "staged": 0,
            "committed": 0,
            "discarded": 0,
            "discarded_at_stop": 0,
            "commit_errors": 0,
            "rejected_reads": 0,
            "horizon_continuations": 0,
            "verifier_format_errors": 0,
            "transport_retries": 0,
            "read_loop_exhausted": 0,
            # What a Strong repair turn produced.
            "correction_read": 0,
            "correction_mutation": 0,
            "correction_reply": 0,
            "corrector_strong": 0,
            # Verifier-authored tool replacement executed through the gate. Not
            # a Strong repair turn.
            "verifier_replacements": 0,
            "replacement_guard_failed": 0,
            "full_accept_contract_violation": 0,
            # Chunks committed without a Strong call (reads only, or a reply). A
            # chunk the repeat budget sent to Strong anyway is not one of these --
            # it is counted under read_budget_escalations.
            "auto_accepted_chunks": 0,
            "read_budget_escalations": 0,
        }

    def bind(
        self, *, execute: Callable[[list[ToolCall]], list[ToolMessage]], orchestrator: Any
    ) -> None:
        """Bind to the orchestrator OBJECT, not to its trajectory list.

        Orchestrator.initialize() REASSIGNS self.trajectory (to message_history
        or to [first_message]) after __init__ has run, so capturing the list
        object here would leave us splicing reads into an orphaned list that
        never reaches results.json. Always read .trajectory lazily.
        """
        self._execute = execute
        self._orchestrator = orchestrator

    @property
    def bound(self) -> bool:
        return self._execute is not None

    def _run(self, tool_call: ToolCall) -> ToolMessage:
        if self._execute is None:
            raise RuntimeError("Tau2DeferredExecutionGate is not bound to an orchestrator")
        # Tool WALL TIME is not measured here: self._execute is the
        # orchestrator's _execute_tool_calls, which the orchestrator patch has
        # already wrapped to accumulate into agent.tool_seconds. That single
        # hook covers the baseline agent too, so both report tool time the
        # same way. Here we only count executions.
        self.stats["tool_calls_executed"] += 1
        return self._execute([tool_call])[0]

    def run_read(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a read-only call inline, mid-draft.

        This call is sealed inside the chunk and is NEVER returned to the
        orchestrator, so step() will never see it -- meaning we must splice
        the agent+tool pair into the trajectory ourselves, and must NOT mark
        it resolved (nothing will ever come back to claim it).

        The announcing message is built BEFORE the call runs, and that ordering
        is load-bearing: Orchestrator.get_trajectory() SORTS by timestamp, so
        list order is irrelevant. Executing first would stamp the ToolMessage
        earlier than the assistant message announcing it, and after sorting the
        result would precede its own call -- which makes the evaluator's
        Environment.set_state replay fail with "Tool message not expected" and
        scores the whole task as an infrastructure error.
        """
        agent_message = AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        result = self._run(tool_call)
        trajectory = getattr(self._orchestrator, "trajectory", None)
        if trajectory is not None:
            trajectory.append(agent_message)
            trajectory.append(result)
        self.stats["reads_executed"] += 1
        return result

    def run_resolved(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a call that IS returned to the orchestrator (a Strong
        repair). Do not splice -- step() appends it through the normal
        flow -- but do mark it resolved so step() reuses this result instead
        of executing the same call a second time.
        """
        result = self._run(tool_call)
        self._resolved[tool_call.id] = result
        self.stats["corrections_executed"] += 1
        return result

    def run_verifier_replacement(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a verifier replacement_action that IS returned to the
        orchestrator. Same take_resolved pairing as a repair, but counted
        separately so it is not mistaken for a Strong repair turn.
        """
        result = self._run(tool_call)
        self._resolved[tool_call.id] = result
        self.stats["verifier_replacements"] += 1
        return result

    def stage(self, tool_call: ToolCall) -> None:
        self._pending = tool_call
        self.stats["staged"] += 1

    def resolve(self, *, accept: bool) -> ToolMessage | None:
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        if not accept:
            self.stats["discarded"] += 1
            return None
        # Like a repair, the accepted pending call IS returned to the
        # orchestrator, so mark it resolved and let step() do the appending.
        result = self._run(pending)
        self._resolved[pending.id] = result
        self.stats["committed"] += 1
        if result.error:
            self.stats["commit_errors"] += 1
        return result

    def defers(self, message: Message) -> bool:
        tool_calls = getattr(message, "tool_calls", None)
        return bool(tool_calls) and all(tc.id in self._resolved for tc in tool_calls)

    def take_resolved(self, message: Message) -> list[ToolMessage]:
        return [self._resolved.pop(tc.id) for tc in message.tool_calls]

    def discard_at_stop(self) -> None:
        if self._pending is not None:
            self.stats["discarded_at_stop"] += 1
            self._pending = None


class SpecAgentTau2TCState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    raw: list[dict] = []
    session: Any = None
    # The customer's opening request, plus any later customer message that
    # could not be bound as the observation of a preceding reply.
    instruction: str = ""
    # How many times in a row an unreviewed chunk has repeated one we have
    # already committed unreviewed -- reads and replies alike. Lives on the
    # state, not on the turn: a reply loop spends one chunk per turn, so a
    # per-turn counter would reset before it could ever reach the budget.
    #
    # It counts *repeats* rather than chunks because counting chunks makes the
    # boundary fire on ordinary progress, sending every few replies to a
    # verifier that is otherwise only asked about mutations. A stuck agent
    # still repeats itself, so the pathology the budget exists for still trips it.
    unverified_chunks: int = 0
    # Signatures of the recent unreviewed chunks, for that repeat test.
    unverified_sigs: list[Any] = []


def _is_message_step(step: AgentStep) -> bool:
    return isinstance(step.action, dict) and step.action.get("type") == MESSAGE_ACTION_TYPE


def _chunk_signature(step: AgentStep) -> tuple[str, Any]:
    """What makes two unreviewed chunks the same thing said twice."""
    if _is_message_step(step):
        text = str((step.action or {}).get("content") or "")
        words = frozenset(w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2)
        return ("message", words)
    action = step.action or {}
    return ("tool", json.dumps({"tool": action.get("tool"), "arguments": action.get("arguments")},
                               sort_keys=True, ensure_ascii=False))


def _repeats_earlier_chunk(signature: tuple[str, Any], seen: list[Any]) -> bool:
    kind, body = signature
    for entry in seen:
        if not (isinstance(entry, tuple) and len(entry) == 2) or entry[0] != kind:
            continue
        other = entry[1]
        if kind == "tool":
            if other == body:
                return True
            continue
        if not body or not other:
            continue
        if len(body & other) / len(body | other) >= _REPEAT_SIMILARITY:
            return True
    return False


def _reasoning_of(message: AssistantMessage) -> str | None:
    return (message.raw_data or {}).get("reasoning_content") or None


def _parse_tool_calls(raw_calls: list[dict]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for call in raw_calls:
        try:
            args = json.loads(call["arguments"]) if call["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool_calls.append(ToolCall(
            id=call["id"] or f"call_{uuid.uuid4().hex[:12]}",
            name=call["name"], arguments=args, requestor="assistant",
        ))
    return tool_calls


class SpecAgentTau2Agent(HalfDuplexAgent):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task | None = None,
        domain: str = "",
        horizon: int = 6,
        weak: dict | None = None,
        strong: dict | None = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        weak = dict(weak or {})
        strong = dict(strong or {})
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.domain = domain
        self.horizon = horizon
        self.protocol = PROTOCOL_NAME

        self.weak_client = OpenAIChatClient(
            base_url=weak["base_url"], api_key=weak.get("api_key", "EMPTY"),
            model=weak["model"], thinking_kwarg=weak.get("thinking_kwarg"),
            auth_scheme=weak.get("auth_scheme", "bearer"),
        )
        self.weak_thinking = weak.get("thinking")
        self.weak_max_output_tokens = weak.get("max_output_tokens")

        self.strong_client = OpenAIChatClient(
            base_url=strong["base_url"], api_key=strong.get("api_key", "EMPTY"),
            model=strong["model"], thinking_kwarg=strong.get("thinking_kwarg"),
            auth_scheme=strong.get("auth_scheme", "bearer"),
        )
        # The Strong repair turn uses the Strong role's thinking setting; the
        # verifier emits one short JSON object, so its thinking is always off
        # and its cost stays prefill-dominated.
        self.strong_thinking = strong.get("thinking")
        self.strong_max_output_tokens = strong.get("max_output_tokens")
        self.verifier_max_output_tokens = strong.get(
            "verifier_max_output_tokens", strong.get("max_output_tokens")
        )
        self._tool_contracts = {t.name: t.openai_schema for t in tools}

        self._loop = asyncio.new_event_loop()
        mutating = {t.name for t in tools if _tool_mutates_state(t)}
        # Steps that must not run on Weak's word alone.
        self._irrevocable_tools = mutating | {
            t.name for t in tools if t.name in _IRREVOCABLE_TOOL_NAMES
        }
        self.gate = Tau2DeferredExecutionGate()
        # exact attribute name the orchestrator.py patch looks for via getattr
        self.deferred_execution_gate = self.gate

        self.task_id: str | None = task.id if task is not None else None
        self.ledger = ResourceLedger()
        self._started_at = 0.0
        # Accumulated by the orchestrator patch's _execute_tool_calls wrapper.
        self.tool_seconds: float = 0.0
        self._attempt_id = ""
        self._request_id = ""
        self._transport_attempt_index = 1
        self._last_attempt_duration = 0.0

    @property
    def system_prompt(self) -> str:
        # Byte-identical to the baseline agent's, so a SpecAgent-vs-baseline delta
        # is the protocol and not a prompt difference.
        return SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION, domain_policy=self.domain_policy
        )

    # -- lifecycle -----------------------------------------------------

    def get_init_state(
        self, message_history: list[Message] | None = None
    ) -> SpecAgentTau2TCState:
        raw: list[dict] = []
        for m in message_history or []:
            raw.extend(message_to_api(m))
        # NOTE: message_history (solo/ticket pre-seed) only informs the LLM
        # prompt here; it cannot be replayed into AgentStep history since
        # tau2 Messages don't carry the actor/mutation metadata a PrefixCommitSession
        # step needs. Standard text-mode tasks start with empty history, so
        # this limitation does not affect the normal eval path.
        self.ledger = ResourceLedger()
        self.tool_seconds = 0.0
        self._started_at = time.perf_counter()
        begin_episode(self)
        return SpecAgentTau2TCState(raw=raw, session=PrefixCommitSession())

    def stop(
        self,
        message: ValidAgentInputMessage | None = None,
        state: SpecAgentTau2TCState | None = None,
    ) -> None:
        self.gate.discard_at_stop()
        stats_path = os.environ.get("SPECAGENT_STATS")
        if not stats_path:
            return
        if self.task_id is None:
            raise TrialIOError(
                "SPECAGENT_STATS set but task_id is missing; refusing to fake a ledger"
            )
        summary = self.ledger.summarize()
        # to_dict(), NOT dataclasses.asdict(): asdict only emits declared fields
        # and would silently drop every derived latency/aggregate metric
        # (llm_call_seconds, the weak/strong/verify splits, the calls[] list).
        resources = summary.to_dict()
        resources["strong_generation_output_tokens"] = summary.strong_generation_output_tokens
        wall = time.perf_counter() - self._started_at
        llm_seconds = resources["llm_call_seconds"]
        tool_seconds = self.tool_seconds
        trial_id = getattr(self, "trial_id", None) or self._attempt_id
        record = {
            "task_id": self.task_id,
            "attempt_id": self._attempt_id,
            "benchmark": "tau2",
            "domain": self.domain,
            "wall_clock_seconds": wall,
            "llm_call_seconds": llm_seconds,
            "tool_seconds": tool_seconds,
            # Everything that is neither inference nor tool execution: env
            # reset, orchestrator scheduling, our own bookkeeping. Weak and
            # Strong never overlap here, so this is never negative.
            "overhead_seconds": wall - llm_seconds - tool_seconds,
            "gate": dict(self.gate.stats),
            "resources": resources,
            "paper_resources": paper_resources_from_tau_agent(self),
            "trial_id": trial_id,
            "protocol": {"name": self.protocol, "horizon": self.horizon},
        }
        if not trial_id:
            raise TrialIOError("stats ledger missing trial_id; refusing to fake a ledger")
        writer = configure_from_stats_env(stats_path)
        if writer is None:
            raise TrialIOError("SPECAGENT_STATS set but shard writer was not configured")
        writer.write_stats(record, task_id=self.task_id, trial_id=trial_id)
        writer.archive_trial(task_id=self.task_id, trial_id=trial_id)

    # -- main entrypoint -------------------------------------------------

    def _absorb_input(
        self, message: ValidAgentInputMessage | None, state: SpecAgentTau2TCState
    ) -> None:
        """Append the orchestrator's message to the Weak prompt and to the protocol history.

        A customer message is the observation of the reply that preceded it; the very first one
        is the task instruction. Tool messages answer an accepted pending call or a repair,
        whose observation was already bound when the gate executed it.
        """
        if isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                state.raw.extend(message_to_api(tm))
            return
        if message is None:
            return
        state.raw.extend(message_to_api(message))
        if not isinstance(message, UserMessage):
            return
        text = message.content or ""
        history = state.session.history
        if history and _is_message_step(history[-1]) and history[-1].observation is None:
            state.session.bind_last_observation(text, terminal=False)
        elif not state.instruction:
            state.instruction = text
        else:
            state.instruction += f"\n\n[Later user message]\n{text}"

    def generate_next_message(
        self, message: ValidAgentInputMessage | None, state: SpecAgentTau2TCState
    ) -> tuple[AssistantMessage, SpecAgentTau2TCState]:
        self._absorb_input(message, state)
        session: PrefixCommitSession = state.session
        if session.phase is SessionPhase.AWAITING_CORRECTION:
            # A rejection carried over from a previous call whose repair never
            # completed; the verdict that produced it is gone.
            return self._strong_correction(
                session, state, reason="a previous draft step was withheld",
            )

        for _ in range(_MAX_CHUNKS_PER_TURN):
            candidate, draft_stats, last_message, last_tool_call, raw_marks, steps_per_call = (
                self._draft_chunk(session, state)
            )
            try:
                verdict = self._strong_verdict(session, candidate, state)
                if verdict.diagnostic_status == "runtime_auto_accept":
                    # A reads-only chunk or a reply that nobody reviewed. Both share
                    # one budget: a reply that never mutates anything is exactly as
                    # unreviewed as a read that never acts, and without this a stuck
                    # reply loop has no backstop.
                    #
                    # The budget counts repeats, not chunks: an agent making progress
                    # says something new each turn and must not be interrupted, while
                    # a stuck one says the same thing again.
                    signature = _chunk_signature(candidate[-1])
                    if _repeats_earlier_chunk(signature, state.unverified_sigs):
                        state.unverified_chunks += 1
                    else:
                        state.unverified_chunks = 0
                    state.unverified_sigs.append(signature)
                    del state.unverified_sigs[:-_UNVERIFIED_SIG_WINDOW]
                    if state.unverified_chunks >= _UNVERIFIED_REPEAT_BUDGET:
                        # Past the budget the boundary fires, and firing means
                        # *verifying*, not rejecting: an honest repeat still commits
                        # if Strong accepts it.
                        self.gate.stats["read_budget_escalations"] += 1
                        self.gate.stats["auto_accepted_chunks"] -= 1
                        verdict = self._strong_verdict(session, candidate, state, force=True)
                        state.unverified_chunks = 0
                        state.unverified_sigs.clear()
                else:
                    # Anything Strong actually ruled on resets the run: the budget
                    # counts consecutive repeats, not lifetime ones.
                    state.unverified_chunks = 0
                    state.unverified_sigs.clear()
            except BaseException:
                # Verification never completed, so nothing in this chunk is
                # committed -- but the Weak tokens were still spent. Record them
                # as discarded before propagating, otherwise a failed task
                # reports zero Weak cost and we under-count the price of failure.
                self._record_draft_spans(
                    draft_stats,
                    PrefixVerdict(accepted_steps=0, rejection_code=RejectionCode.OTHER,
                                  confidence=0.0, raw="verification did not complete"),
                    steps_per_call,
                )
                self.gate.discard_at_stop()  # drop any staged call; it must not leak
                raise
            self._record_draft_spans(draft_stats, verdict, steps_per_call)
            result = session.apply_verdict(verdict)

            last = candidate[-1]
            pending_tool = bool((last.metadata or {}).get("pending")) and not _is_message_step(last)

            if verdict.accepted_steps == len(candidate):
                if pending_tool:
                    tool_result = self.gate.resolve(accept=True)
                    assert tool_result is not None and last_tool_call is not None
                    session.bind_last_observation(
                        tool_result.content, terminal=False,
                        metadata={"error": True} if tool_result.error else None,
                    )
                    # Deliberately NOT appending the tool result to state.raw: this
                    # message is returned to the orchestrator, which routes it to
                    # ENV and feeds the ToolMessage back on the next call. Appending
                    # here would duplicate it.
                    return last_message, state
                if _is_message_step(last):
                    # Sent by returning it; the customer's answer binds later as its
                    # observation.
                    session.bind_last_observation(None, terminal=False, metadata={"pending": False})
                    return last_message, state
                # Reads only: the horizon closed the chunk. Nothing is sendable, so Weak
                # keeps drafting from the extended history instead of handing a read back
                # to the orchestrator (which would execute it a second time).
                self.gate.stats["horizon_continuations"] += 1
                continue

            # Rejected at verdict.accepted_steps. The pending step never runs; the rejected
            # suffix leaves the Weak prompt so the repair starts from the clean prefix.
            if pending_tool:
                self.gate.resolve(accept=False)
            rejected = candidate[verdict.accepted_steps:]
            del state.raw[raw_marks[verdict.accepted_steps]:]
            self.gate.stats["rejected_reads"] += sum(
                1 for step in rejected if not (step.metadata or {}).get("pending")
            )
            assert result.correction_required
            return self._redirect(session, state, verdict=verdict)

        # Weak only ever read, for _MAX_CHUNKS_PER_TURN chunks running. Every
        # chunk was accepted, so the session is READY and a Strong repair
        # would be illegal here (commit_correction requires AWAITING_CORRECTION).
        # Emit a plain message so the turn terminates instead of spinning.
        self.gate.stats["read_loop_exhausted"] += 1
        fallback = AssistantMessage(role="assistant", content=FALLBACK_CONTENT, tool_calls=None)
        state.raw.extend(message_to_api(fallback))
        return fallback, state

    def _run_model(
        self,
        make_coro: Callable[[], Any],
        *,
        what: str,
        role: ModelRole,
        phase: InferencePhase,
    ) -> Any:
        """Run one model call on the agent's loop, retrying transient transport
        failures with exponential backoff. See _TRANSIENT_ERRORS.

        Each logical call gets one request_id; every HTTP attempt (including
        failures) is a span. FLOPs/HBM stay unknown via metrics_interface.
        """
        delay = _RETRY_BASE_DELAY
        last: BaseException | None = None
        begin_logical_request(self)
        for attempt in range(_MAX_TRANSPORT_RETRIES):
            self._transport_attempt_index = attempt + 1
            ids = span_ids_from_agent(self)
            started = time.perf_counter()
            try:
                result = self._loop.run_until_complete(make_coro())
                self._last_attempt_duration = time.perf_counter() - started
                try:
                    result._tau2_span_ids = ids
                except Exception:
                    pass
                return result
            except _TRANSIENT_ERRORS as exc:
                last = exc
                self.gate.stats["transport_retries"] += 1
                elapsed = time.perf_counter() - started
                self._last_attempt_duration = elapsed
                record_failed_transport(
                    self, role=role, phase=phase, duration_seconds=elapsed, ids=ids,
                )
                if attempt == _MAX_TRANSPORT_RETRIES - 1:
                    break
                time.sleep(delay)
                delay *= 2
            except BaseException as exc:
                elapsed = time.perf_counter() - started
                self._last_attempt_duration = elapsed
                status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                record_failed_transport(
                    self, role=role, phase=phase, duration_seconds=elapsed, ids=ids,
                    status=status,
                )
                raise
        raise RuntimeError(
            f"{what} failed after {_MAX_TRANSPORT_RETRIES} attempts: {last!r}"
        ) from last

    # -- weak drafting -----------------------------------------------------

    def _weak_call(
        self, state: SpecAgentTau2TCState
    ) -> tuple[AssistantMessage, list[ToolCall], dict]:
        api_messages = [{"role": "system", "content": self.system_prompt}] + state.raw
        tools = [t.openai_schema for t in self.tools]
        t0 = time.perf_counter()
        result = self._run_model(
            lambda: self.weak_client.complete_with_tools(
                api_messages, tools,
                max_output_tokens=self.weak_max_output_tokens, thinking=self.weak_thinking,
            ),
            what="weak draft call",
            role=ModelRole.WEAK,
            phase=InferencePhase.DRAFT,
        )
        dur = float(getattr(self, "_last_attempt_duration", time.perf_counter() - t0))
        usage = usage_fields_from_result(result)
        ids = getattr(result, "_tau2_span_ids", None) or span_ids_from_agent(self)
        stat = {
            **usage,
            "duration_seconds": dur,
            "request_id": ids.get("request_id") or self._request_id,
            "transport_attempt_id": ids.get("transport_attempt_id"),
            "transport_attempt_index": (
                ids.get("transport_attempt_index") or self._transport_attempt_index
            ),
        }
        content = result.content or ""
        # Every tool call Weak proposes in one generation is kept (parallel tool
        # calls); _draft_chunk splits them into one step each.
        tool_calls = _parse_tool_calls(result.tool_calls or [])
        if tool_calls:
            content = ""
        elif not content:
            content = FALLBACK_CONTENT
        assistant_message = AssistantMessage(
            role="assistant",
            content=(content or None) if not tool_calls else None,
            tool_calls=tool_calls or None,
            usage={"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens},
            generation_time_seconds=dur,
            raw_data={"reasoning_content": result.reasoning or ""},
        )
        return assistant_message, tool_calls, stat

    def _draft_chunk(
        self, session: PrefixCommitSession, state: SpecAgentTau2TCState,
    ) -> tuple[
        tuple[AgentStep, ...], list[dict], AssistantMessage, ToolCall | None,
        list[int], list[int],
    ]:
        """Draft one chunk. Returns the sealed candidate, per-call usage (one entry per
        Weak inference call), the last assistant message, the staged tool call if any,
        ``raw_marks[i]`` = len(state.raw) before step i was appended (so a rejected
        suffix can be cut out of the prompt exactly), and ``steps_per_call[j]`` = how
        many candidate steps Weak inference call j produced (more than 1 when Weak
        proposes several tool calls at once)."""
        session.begin_draft(self.horizon)
        draft_stats: list[dict] = []
        raw_marks: list[int] = []
        steps_per_call: list[int] = []
        last_message: AssistantMessage | None = None
        last_tool_call: ToolCall | None = None
        while True:
            assistant_message, tool_calls, stat = self._weak_call(state)
            draft_stats.append(stat)
            n_steps_this_call = 0
            seal_chunk = False

            if not tool_calls:
                # A reply to the customer always seals the chunk. It is pending
                # until it is returned, which is what the verifier is told if the
                # repeat budget sends it to Strong.
                step = AgentStep(
                    actor=ModelRole.WEAK,
                    action={
                        "type": MESSAGE_ACTION_TYPE, "content": assistant_message.content or "",
                    },
                    observation=None,
                    metadata={"pending": True},
                    reasoning=_reasoning_of(assistant_message),
                    output_tokens=stat["output_tokens"],
                    duration_seconds=stat["duration_seconds"],
                )
                raw_marks.append(len(state.raw))
                session.append_weak_step(step)
                state.raw.extend(message_to_api(assistant_message))
                last_message, last_tool_call = assistant_message, None
                steps_per_call.append(1)
                break

            # A multi-call batch is split into independent single-tool-call messages --
            # each with its own raw_marks entry -- so a verdict's prefix cut can land
            # inside the batch. Cutting a shared multi-call announcement mid-batch would
            # otherwise leave dangling tool_call_ids (if cut mid-batch) or re-send
            # already-accepted steps' messages (if the cut were forced to the batch start).
            for i, tool_call in enumerate(tool_calls):
                action = {"tool": tool_call.name, "arguments": tool_call.arguments}
                call_message = assistant_message if len(tool_calls) == 1 else AssistantMessage(
                    role="assistant", content=None, tool_calls=[tool_call],
                    usage={
                        "prompt_tokens": stat["input_tokens"] if i == 0 else 0,
                        "completion_tokens": stat["output_tokens"] if i == 0 else 0,
                    },
                    generation_time_seconds=stat["duration_seconds"] if i == 0 else 0.0,
                    raw_data=assistant_message.raw_data,
                )
                # The whole batch is one Weak generation; its cost cannot be split
                # per call, so it is attributed to the batch's first step only (see
                # _record_draft_spans, which charges per Weak call via steps_per_call,
                # not by summing these fields back up).
                step_output_tokens = stat["output_tokens"] if i == 0 else 0
                step_duration = stat["duration_seconds"] if i == 0 else 0.0
                step_reasoning = _reasoning_of(assistant_message) if i == 0 else None

                if tool_call.name in self._irrevocable_tools:
                    self.gate.stage(tool_call)
                    step = AgentStep(
                        actor=ModelRole.WEAK, action=action, observation=None,
                        metadata={"pending": True}, reasoning=step_reasoning,
                        output_tokens=step_output_tokens, duration_seconds=step_duration,
                    )
                    raw_marks.append(len(state.raw))
                    session.append_weak_step(step)
                    # Only the assistant message goes in: the call has NOT run, so
                    # there is no tool result to record. Staging always seals the
                    # chunk, so no later weak call in this chunk sees the dangling
                    # tool_call. If Strong accepts, the orchestrator feeds the real
                    # ToolMessage back next turn; if it rejects, generate_next_message
                    # cuts this message back out.
                    state.raw.extend(message_to_api(call_message))
                    last_message, last_tool_call = call_message, tool_call
                    n_steps_this_call += 1
                    seal_chunk = True
                    break

                tool_result = self.gate.run_read(tool_call)
                step = AgentStep(
                    actor=ModelRole.WEAK, action=action, observation=tool_result.content,
                    metadata={"error": True} if tool_result.error else None,
                    reasoning=step_reasoning, output_tokens=step_output_tokens,
                    duration_seconds=step_duration,
                )
                raw_marks.append(len(state.raw))
                session.append_weak_step(step)
                state.raw.extend(message_to_api(call_message))
                state.raw.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "content": tool_result.content or "",
                })
                last_message, last_tool_call = call_message, tool_call
                n_steps_this_call += 1
                if len(session.draft) >= self.horizon:
                    seal_chunk = True
                    break

            steps_per_call.append(n_steps_this_call)
            if seal_chunk:
                break

        candidate = session.seal_draft()
        return candidate, draft_stats, last_message, last_tool_call, raw_marks, steps_per_call

    def _record_draft_spans(
        self, draft_stats: list[dict], verdict: PrefixVerdict, steps_per_call: list[int],
    ) -> None:
        # A Weak call is charged as committed if the FIRST candidate step it
        # produced made it into the accepted prefix.
        offset = 0
        for stat, n in zip(draft_stats, steps_per_call, strict=True):
            committed = offset < verdict.accepted_steps
            offset += n
            draft_ids = {
                "request_id": stat.get("request_id") or self._request_id,
                "transport_attempt_id": stat.get("transport_attempt_id"),
                "transport_attempt_index": int(stat.get("transport_attempt_index") or 1),
                "parent_task_id": self.task_id,
                "parent_episode_id": self._attempt_id or None,
            }
            self.ledger.add(make_tau2_span(
                role=ModelRole.WEAK, phase=InferencePhase.DRAFT,
                duration_seconds=stat["duration_seconds"],
                committed_output_tokens=stat["output_tokens"] if committed else 0,
                discarded_output_tokens=0 if committed else stat["output_tokens"],
                usage_status=stat["usage_status"],
                cache_provenance=stat.get("cache_provenance", "unspecified"),
                input_tokens=stat["input_tokens"],
                output_tokens=stat["output_tokens"],
                cached_input_tokens=stat["cached_input_tokens"],
                reasoning_tokens=stat["reasoning_tokens"],
                input_tokens_status=stat.get("input_tokens_status"),
                output_tokens_status=stat.get("output_tokens_status"),
                cached_tokens_status=stat.get("cached_tokens_status"),
                reasoning_tokens_status=stat.get("reasoning_tokens_status"),
                **span_id_kwargs(draft_ids),
            ))
            mark_span_recorded(self, draft_ids)

    # -- strong verification / repair ------------------------------------

    def _strong_verdict(
        self, session: PrefixCommitSession, candidate: tuple[AgentStep, ...],
        state: SpecAgentTau2TCState, *, force: bool = False,
    ) -> PrefixVerdict:
        """Verify the candidate, or auto-accept it when it does not end in a staged call.

        `force` overrides that gate. The unreviewed-chunk budget uses it: a chunk that is
        out of scope is exactly why the budget was counting it, so re-applying the gate there
        would return the same auto-accept forever.
        """
        self._request_id = ""
        last = candidate[-1]
        pending = bool((last.metadata or {}).get("pending"))
        if not force and not (pending and not _is_message_step(last)):
            self.gate.stats["auto_accepted_chunks"] += 1
            return PrefixVerdict(
                accepted_steps=len(candidate), rejection_code=None, confidence=1.0,
                reason="", raw="not in verify scope; committed unreviewed",
                diagnostic_status="runtime_auto_accept",
            )
        task = TaskContext(
            task_id=self.task_id or "tau2-task",
            instruction=state.instruction,
            tool_contracts=self._tool_contracts,
        )
        # The verifier always rules on the whole chunk, executed reads included:
        # the reads' observations are real, but the verifier can still decline to
        # commit them.
        payload = verification_payload(
            session.history, candidate, policy=_CONTEXT_POLICY, task=task
        )
        # The candidate's own reasoning is the drafter arguing for the step under
        # review, so the verifier judges the actions without it.
        payload, pruned = prune_payload(
            payload, history_reasoning=True, candidate_reasoning=False,
            compaction_summaries=True,
        )
        payload.setdefault("request_evidence_metadata", {})["pruned_context"] = pruned
        last_error = "unknown format error"
        for attempt in range(2):
            messages = verifier_messages(
                payload, domain=self.domain, domain_policy=self.domain_policy,
                retry=attempt > 0,
            )
            t0 = time.perf_counter()
            # temperature 0: a sampled verdict would add noise to every decision.
            result = self._run_model(
                lambda messages=messages: self.strong_client.complete(
                    messages, max_output_tokens=self.verifier_max_output_tokens,
                    thinking=False, temperature=0.0,
                ),
                what="strong verification call",
                role=ModelRole.STRONG,
                phase=InferencePhase.VERIFICATION,
            )
            dur = float(getattr(self, "_last_attempt_duration", time.perf_counter() - t0))
            record_result_span(
                self, result, role=ModelRole.STRONG, phase=InferencePhase.VERIFICATION,
                duration_seconds=dur,
            )
            try:
                verdict = parse_prefix_verdict(
                    result.content, candidate_length=len(candidate), contract=VERIFIER_CONTRACT,
                )
            except ValueError as exc:
                last_error = str(exc)
                continue
            if verdict.diagnostic_status == "full_accept_contract_violation":
                self.gate.stats["full_accept_contract_violation"] += 1
            return replace(
                verdict, verifier_duration_seconds=dur, raw=result.content or "",
                format_retries=attempt,
            )

        self.gate.stats["verifier_format_errors"] += 1
        # Fail closed on the step that has not happened, not on the ones that
        # have: keep every executed read before the pending step. Rejecting
        # those would erase real observations from the Weak prompt, while the
        # unverified pending step is what actually needs a decision.
        executed = sum(1 for step in candidate if not (step.metadata or {}).get("pending"))
        return PrefixVerdict(
            accepted_steps=executed,
            rejection_code=None if executed == len(candidate) else RejectionCode.OTHER,
            confidence=0.0,
            reason="the reviewer could not produce a verdict for this step",
            raw=last_error,
            diagnostic_status="format_failure",
            format_retries=1,
        )

    def _redirect(
        self,
        session: PrefixCommitSession,
        state: SpecAgentTau2TCState,
        *,
        verdict: PrefixVerdict,
    ) -> tuple[AssistantMessage, SpecAgentTau2TCState]:
        """Reject path: a schema-valid replacement executes once, else one Strong repair turn.

        The repair turn does not see the rejected suffix (already stripped from
        state.raw), Weak reasoning, or a guard-failed action payload.
        """
        if verdict.diagnostic_status == "model_verdict":
            code = verdict.rejection_code.value if verdict.rejection_code else "unspecified"
            self.gate.stats[f"reject_{code}"] = self.gate.stats.get(f"reject_{code}", 0) + 1
        action, _guard_error = validate_replacement_action(verdict.replacement_action, self.tools)
        if verdict.replacement_action is not None and action is not None:
            return self._execute_verifier_replacement(session, state, action)
        if verdict.replacement_action is not None:
            self.gate.stats["replacement_guard_failed"] += 1
        self.gate.stats["corrector_strong"] += 1
        return self._strong_correction(session, state, reason=verdict.reason)

    def _execute_verifier_replacement(
        self,
        session: PrefixCommitSession,
        state: SpecAgentTau2TCState,
        action: Mapping[str, Any],
    ) -> tuple[AssistantMessage, SpecAgentTau2TCState]:
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=str(action["tool"]),
            arguments=dict(action["arguments"]),
            requestor="assistant",
        )
        assistant_message = AssistantMessage(
            role="assistant", content=None, tool_calls=[tool_call],
        )
        tool_result = self.gate.run_verifier_replacement(tool_call)
        step = AgentStep(
            actor=ModelRole.STRONG,
            action={"tool": tool_call.name, "arguments": tool_call.arguments},
            observation=tool_result.content,
            metadata={
                "source": "verifier_replacement",
                **({"error": True} if tool_result.error else {}),
            },
        )
        session.commit_correction(step)
        state.raw.extend(message_to_api(assistant_message))
        return assistant_message, state

    def _strong_correction(
        self, session: PrefixCommitSession, state: SpecAgentTau2TCState, *, reason: str,
    ) -> tuple[AssistantMessage, SpecAgentTau2TCState]:
        messages = correction_messages(
            conversation=list(state.raw), reason=reason, domain_policy=self.domain_policy,
        )
        tools = [t.openai_schema for t in self.tools]
        t0 = time.perf_counter()
        result = self._run_model(
            lambda: self.strong_client.complete_with_tools(
                messages, tools,
                max_output_tokens=self.strong_max_output_tokens, thinking=self.strong_thinking,
            ),
            what="strong correction call",
            role=ModelRole.STRONG,
            phase=InferencePhase.CORRECTION,
        )
        dur = float(getattr(self, "_last_attempt_duration", time.perf_counter() - t0))
        record_result_span(
            self, result, role=ModelRole.STRONG, phase=InferencePhase.CORRECTION,
            duration_seconds=dur,
        )

        if result.tool_calls:
            tool_calls = _parse_tool_calls(result.tool_calls)
            assistant_message = AssistantMessage(
                role="assistant", content=None, tool_calls=tool_calls,
                usage={"prompt_tokens": result.input_tokens,
                       "completion_tokens": result.output_tokens},
                generation_time_seconds=dur,
                raw_data={"reasoning_content": result.reasoning or ""},
            )
            # A repair is the reviewed fallback itself: execute immediately for
            # real (mutating or not). run_resolved (not run_read) because this
            # message IS returned to the orchestrator: step() appends it and
            # reuses this result rather than executing the call a second time.
            tool_results: list[ToolMessage] = []
            for tool_call in tool_calls:
                if tool_call.name in self._irrevocable_tools:
                    self.gate.stats["correction_mutation"] += 1
                else:
                    self.gate.stats["correction_read"] += 1
                tool_results.append(self.gate.run_resolved(tool_call))
            if len(tool_calls) == 1:
                action: Any = {"tool": tool_calls[0].name, "arguments": tool_calls[0].arguments}
                observation: Any = tool_results[0].content
                has_error = bool(tool_results[0].error)
            else:
                # commit_correction takes exactly one AgentStep per repair
                # (PrefixCommitSession enforces this), so a multi-call repair
                # batches into one step with a `{"parallel": [...]}` action.
                # AgentStep.action/.observation are untyped, and every consumer
                # of committed history dumps a committed action verbatim.
                action = {"parallel": [{"tool": tc.name, "arguments": tc.arguments}
                                       for tc in tool_calls]}
                observation = [tr.content for tr in tool_results]
                has_error = any(tr.error for tr in tool_results)
            session.commit_correction(AgentStep(
                actor=ModelRole.STRONG, action=action, observation=observation,
                metadata={"error": True} if has_error else None,
                reasoning=result.reasoning or None,
                output_tokens=result.output_tokens, duration_seconds=dur,
            ))
            # As above: the orchestrator feeds the ToolMessage back next turn,
            # so only the assistant message goes into state.raw here.
            state.raw.extend(message_to_api(assistant_message))
            return assistant_message, state

        self.gate.stats["correction_reply"] += 1
        content = result.content or FALLBACK_CONTENT
        assistant_message = AssistantMessage(
            role="assistant", content=content, tool_calls=None,
            usage={"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens},
            generation_time_seconds=dur, raw_data={"reasoning_content": result.reasoning or ""},
        )
        session.commit_correction(AgentStep(
            actor=ModelRole.STRONG, action={"type": MESSAGE_ACTION_TYPE, "content": content},
            observation=None, reasoning=result.reasoning or None,
            output_tokens=result.output_tokens, duration_seconds=dur,
        ))
        state.raw.extend(message_to_api(assistant_message))
        return assistant_message, state


def create_specagent_tau2_agent(tools, domain_policy, **kwargs) -> SpecAgentTau2Agent:
    """Factory registered as 'specagent'. `llm_args` (threaded through from
    tau2.data_model.simulation.TextRunConfig.llm_args_agent) must supply
    domain, horizon, weak={base_url, api_key, model, ...},
    strong={base_url, api_key, model, ...} -- see run_eval.py.
    """
    llm_args = dict(kwargs.get("llm_args") or {})
    return SpecAgentTau2Agent(
        tools=tools, domain_policy=domain_policy, task=kwargs.get("task"), **llm_args
    )
