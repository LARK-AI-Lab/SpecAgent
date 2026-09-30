"""Apply the SpecAgent deferred-execution-gate hook to tau2's Orchestrator.

Usage: python apply_orchestrator_patch.py [path/to/tau2/orchestrator/orchestrator.py]

Without an argument, patches the orchestrator of the importable `tau2` package.
Idempotent (no-ops if already applied).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _default_target() -> Path:
    spec = importlib.util.find_spec("tau2")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("tau2 is not importable; pass the orchestrator.py path explicitly")
    return Path(next(iter(spec.submodule_search_locations))) / "orchestrator" / "orchestrator.py"


TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_target()

# Present only in a patched orchestrator.
MARKER = "deferred_execution_gate"

INIT_OLD = '''        # Validate mode compatibility
        self._validate_mode_compatibility()

    def _validate_mode_compatibility(self):'''

INIT_NEW = '''        # Validate mode compatibility
        self._validate_mode_compatibility()

        # SpecAgent integration: bind the agent's deferred-execution gate
        # (if any) so it can execute tool calls for real from inside
        # generate_next_message and tell us not to re-execute them here.
        # Bind `self`, NOT self.trajectory -- initialize() reassigns that list
        # later, so the gate must read it lazily or reads it splices in are
        # silently lost. See benchmarks/tau2/agent.py.
        # Task-level tool-time accounting. Wrapping here (rather than inside
        # the gate) is the single point every tool execution passes through,
        # so the baseline agent gets the same numbers as the SpecAgent agent.
        # Feeds overhead_seconds = wall - llm_call_seconds - tool_seconds.
        if hasattr(self.agent, "tool_seconds"):
            import time as _specagent_time

            _specagent_exec = self._execute_tool_calls

            def _specagent_timed_execute(tool_calls):
                _t0 = _specagent_time.perf_counter()
                try:
                    return _specagent_exec(tool_calls)
                finally:
                    self.agent.tool_seconds += _specagent_time.perf_counter() - _t0

            self._execute_tool_calls = _specagent_timed_execute

        gate = getattr(self.agent, "deferred_execution_gate", None)
        if gate is not None:
            gate.bind(execute=self._execute_tool_calls, orchestrator=self)

    def _validate_mode_compatibility(self):'''

STEP_OLD = '''        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            tool_results = self._execute_tool_calls(self.message.tool_calls)
            assert len(self.message.tool_calls) == len(tool_results), (
                "Number of tool calls and tool messages should be the same"
            )'''

STEP_NEW = '''        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            # SpecAgent integration: if the agent already executed these
            # calls for real from inside generate_next_message (an accepted
            # staged mutation, or a Strong correction), reuse those results
            # instead of double-executing them.
            gate = getattr(self.agent, "deferred_execution_gate", None)
            if self.from_role == Role.AGENT and gate is not None and gate.defers(self.message):
                tool_results = gate.take_resolved(self.message)
            else:
                tool_results = self._execute_tool_calls(self.message.tool_calls)
            assert len(self.message.tool_calls) == len(tool_results), (
                "Number of tool calls and tool messages should be the same"
            )'''


def apply_patch() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("already applied, no-op")
        return
    if text.count(INIT_OLD) != 1:
        raise SystemExit(
            f"expected exactly one match for __init__ anchor, found {text.count(INIT_OLD)}"
        )
    if text.count(STEP_OLD) != 1:
        raise SystemExit(
            f"expected exactly one match for step() anchor, found {text.count(STEP_OLD)}"
        )
    text = text.replace(INIT_OLD, INIT_NEW, 1)
    text = text.replace(STEP_OLD, STEP_NEW, 1)
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET}")


if __name__ == "__main__":
    apply_patch()
