"""Baseline tau2 agent: a single LLM (Weak-only or Strong-only) drives the
whole conversation directly, with no prefix-commit protocol involved.

Bypasses tau2's litellm-backed `generate()` (see llm_agent.py) in favor of a
raw OpenAI-compatible HTTP client (openai_client.py), shared with
agent.py so weak/strong calls, thinking-mode handling, and token
accounting are identical between baseline and protocol runs. Token/time
accounting is written only when SPECAGENT_BASELINE_STATS is set -- tau2's own
`usage` tracking on AssistantMessage is populated manually below, not by
litellm.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from metrics_interface import (
    begin_episode,
    record_result_span,
    run_until_complete_dispatched,
)
from openai_client import OpenAIChatClient
from pydantic import BaseModel
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

from specagent.evals.trial_shard_io import TrialIOError, configure_from_stats_env
from specagent.protocol.accounting import InferencePhase, ResourceLedger
from specagent.protocol.resource_estimate import paper_resources_from_tau_agent
from specagent.protocol.types import ModelRole

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

FALLBACK_CONTENT = (
    "I'm sorry, I ran into an internal error processing that. Could you repeat your last message?"
)


class SpecAgentTau2State(BaseModel):
    """Running OpenAI-format message list (system prompt held separately)."""

    model_config = {"arbitrary_types_allowed": True}

    raw: list[dict] = []


def _tool_call_to_api(tc: ToolCall) -> dict:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
    }


def message_to_api(message: Message) -> list[dict]:
    """Convert one tau2 Message into 0+ OpenAI-format dicts for the client."""
    if isinstance(message, UserMessage):
        return [{"role": "user", "content": message.content or ""}]
    if isinstance(message, ToolMessage):
        return [{"role": "tool", "tool_call_id": message.id, "content": message.content or ""}]
    if isinstance(message, MultiToolMessage):
        return [
            {"role": "tool", "tool_call_id": tm.id, "content": tm.content or ""}
            for tm in message.tool_messages
        ]
    if isinstance(message, AssistantMessage):
        return [
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [_tool_call_to_api(tc) for tc in (message.tool_calls or [])] or None,
                "reasoning_content": (message.raw_data or {}).get("reasoning_content", ""),
            }
        ]
    raise TypeError(f"Unsupported message type for tau2 agent: {type(message)!r}")


class BaselineTau2Agent(HalfDuplexAgent):
    """Direct single-model tau2 agent (weak-only or strong-only baseline)."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task | None = None,
        **llm_args: Any,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        llm_args = dict(llm_args)
        # Which side of the comparison this baseline is. Drives ModelRole so a
        # baseline run's resources dict is directly comparable to a SpecAgent run's:
        # a weak-only baseline's tokens land in weak_*, a strong-only
        # baseline's in strong_act_*.
        role = llm_args.pop("role", "weak")
        self.role = ModelRole.STRONG if role == "strong" else ModelRole.WEAK
        self.model = llm_args.pop("model")
        base_url = llm_args.pop("base_url")
        api_key = llm_args.pop("api_key", "EMPTY")
        thinking_kwarg = llm_args.pop("thinking_kwarg", None)
        auth_scheme = llm_args.pop("auth_scheme", "bearer")
        self.thinking = llm_args.pop("thinking", None)
        self.max_output_tokens = llm_args.pop("max_output_tokens", None)
        self.client = OpenAIChatClient(
            base_url=base_url,
            api_key=api_key,
            model=self.model,
            thinking_kwarg=thinking_kwarg,
            auth_scheme=auth_scheme,
        )
        # tau2's HalfDuplexAgent contract is synchronous; keep one persistent
        # event loop per agent instance so the AsyncOpenAI client's internal
        # httpx client always sees the same loop across calls.
        self._loop = asyncio.new_event_loop()
        self.task_id: str | None = task.id if task is not None else None
        self.ledger = ResourceLedger()
        self._started_at = 0.0
        self._turns = {"tool_call_turns": 0, "text_turns": 0}
        # Accumulated by the orchestrator patch's _execute_tool_calls wrapper.
        self.tool_seconds: float = 0.0
        self._attempt_id = ""
        self._request_id = ""
        self._transport_attempt_index = 1
        self._last_attempt_duration = 0.0

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION, domain_policy=self.domain_policy
        )

    def get_init_state(self, message_history: list[Message] | None = None) -> SpecAgentTau2State:
        raw: list[dict] = []
        for m in message_history or []:
            raw.extend(message_to_api(m))
        self.ledger = ResourceLedger()
        self._turns = {"tool_call_turns": 0, "text_turns": 0}
        self.tool_seconds = 0.0
        self._started_at = time.perf_counter()
        begin_episode(self)
        return SpecAgentTau2State(raw=raw)

    def generate_next_message(
        self, message: ValidAgentInputMessage | None, state: SpecAgentTau2State
    ) -> tuple[AssistantMessage, SpecAgentTau2State]:
        if isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                state.raw.extend(message_to_api(tm))
        elif message is not None:
            state.raw.extend(message_to_api(message))

        api_messages = [{"role": "system", "content": self.system_prompt}] + state.raw
        tools = [t.openai_schema for t in self.tools]
        result, dur, ids = run_until_complete_dispatched(
            self,
            self._loop,
            self.client.complete_with_tools(
                api_messages,
                tools,
                max_output_tokens=self.max_output_tokens,
                thinking=self.thinking,
            ),
            role=self.role,
            phase=InferencePhase.ACT,
        )
        # Phase ACT: this baseline commits every token it generates (nothing is
        # speculated, so nothing is ever discarded).
        record_result_span(
            self,
            result,
            role=self.role,
            phase=InferencePhase.ACT,
            duration_seconds=dur,
            committed=True,
            ids=ids,
        )

        tool_calls: list[ToolCall] | None = None
        content = result.content or ""
        if result.tool_calls:
            self._turns["tool_call_turns"] += 1
            tool_calls = []
            for call in result.tool_calls:
                try:
                    args = json.loads(call["arguments"]) if call["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=call["id"] or f"call_{uuid.uuid4().hex[:12]}",
                        name=call["name"],
                        arguments=args,
                        requestor="assistant",
                    )
                )
            content = ""
        else:
            self._turns["text_turns"] += 1
            if not content:
                content = FALLBACK_CONTENT

        assistant_message = AssistantMessage(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls,
            usage={"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens},
            generation_time_seconds=dur,
            raw_data={"reasoning_content": result.reasoning or ""},
        )
        state.raw.extend(message_to_api(assistant_message))
        return assistant_message, state

    def stop(
        self,
        message: ValidAgentInputMessage | None = None,
        state: SpecAgentTau2State | None = None,
    ) -> None:
        stats_path = os.environ.get("SPECAGENT_BASELINE_STATS")
        if not stats_path:
            return
        if self.task_id is None:
            raise TrialIOError(
                "SPECAGENT_BASELINE_STATS set but task_id is missing; refusing to fake a ledger"
            )
        # Same record shape as agent.stop() so baseline and SpecAgent runs can
        # be diffed field-for-field.
        resources = self.ledger.summarize().to_dict()
        wall = time.perf_counter() - self._started_at
        llm_seconds = resources["llm_call_seconds"]
        record = {
            "task_id": self.task_id,
            "wall_clock_seconds": wall,
            "llm_call_seconds": llm_seconds,
            "tool_seconds": self.tool_seconds,
            "overhead_seconds": wall - llm_seconds - self.tool_seconds,
            "turns": dict(self._turns),
            "resources": resources,
            "paper_resources": paper_resources_from_tau_agent(self),
            "trial_id": getattr(self, "trial_id", None) or self._attempt_id,
        }
        trial_id = getattr(self, "trial_id", None) or getattr(self, "_attempt_id", None)
        if not trial_id:
            raise TrialIOError("stats ledger missing trial_id; refusing to fake a ledger")
        writer = configure_from_stats_env(stats_path)
        if writer is None:
            raise TrialIOError("SPECAGENT_BASELINE_STATS set but shard writer was not configured")
        writer.write_stats(record, task_id=self.task_id, trial_id=trial_id)
        writer.archive_trial(task_id=self.task_id, trial_id=trial_id)


def create_baseline_tau2_agent(tools, domain_policy, **kwargs) -> BaselineTau2Agent:
    """Factory registered as e.g. 'specagent_weak' / 'specagent_strong'.

    kwargs follow tau2.runner.build.build_agent's call convention: llm (str,
    unused -- the model name lives in llm_args["model"]), llm_args (dict:
    base_url, model, api_key, thinking_kwarg, thinking, max_output_tokens),
    task (Task), plus unused audio/solo params.
    """
    llm_args = dict(kwargs.get("llm_args") or {})
    return BaselineTau2Agent(
        tools=tools, domain_policy=domain_policy, task=kwargs.get("task"), **llm_args
    )
