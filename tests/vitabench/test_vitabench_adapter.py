"""VitaBench adapter: tool labels, session patch, driver commands, and the gateway setting.

Tests that read the benchmark need a VitaBench checkout at commit 742e240 in $VITABENCH_DIR and
are skipped without one.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "benchmarks" / "vitabench", ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import apply_session_patch  # noqa: E402
import run_eval  # noqa: E402
import tool_classes  # noqa: E402

from specagent import server  # noqa: E402
from specagent.protocol import side_effects  # noqa: E402
from specagent.protocol.side_effects import action_mutates_state  # noqa: E402
from specagent.runtime.gateway import SpeculativeToolGateway  # noqa: E402

VITA_ROOT = Path(os.environ.get("VITABENCH_DIR") or "/nonexistent").expanduser()
needs_vita = pytest.mark.skipif(
    not (VITA_ROOT / "src" / "vita").is_dir(), reason="needs a VitaBench checkout in $VITABENCH_DIR"
)
# The labels of the pinned checkout, as the paper run's gateway received them.
PINNED_CLASSES = ROOT / "tests" / "golden" / "vitabench_tool_classes.json"


@pytest.fixture(autouse=True)
def _no_class_file(monkeypatch):
    monkeypatch.delenv(side_effects.TOOL_CLASS_FILE_ENV, raising=False)
    side_effects._OFFICIAL_CACHE.clear()
    yield
    side_effects._OFFICIAL_CACHE.clear()


# --- tool classification ------------------------------------------------------------------------


def _source_table() -> dict[str, str]:
    """A second, inline extraction of the @is_tool labels, so the extractor is not checked
    against itself."""
    table: dict[str, str] = {}
    for path in (VITA_ROOT / "src" / "vita").rglob("*.py"):
        for match in re.finditer(
            r"@is_tool\(\s*(?:tool_type\s*=\s*)?ToolType\.(\w+)[^)]*\)\s*\n\s*(?:async\s+)?def\s+(\w+)",
            path.read_text(encoding="utf-8"),
        ):
            tool_type, name = match.groups()
            table[name] = tool_type.lower()
    return table


@needs_vita
def test_tool_labels_match_the_vita_source_and_the_pinned_table() -> None:
    expected = _source_table()
    assert expected, "no @is_tool decorators found; vita layout changed?"
    got = tool_classes.scan(VITA_ROOT)
    assert got == expected
    counts = {kind: sum(1 for v in got.values() if v == kind) for kind in set(got.values())}
    assert counts == {"write": 27, "read": 33, "generic": 6}
    assert tool_classes.mutates_map(VITA_ROOT) == json.loads(PINNED_CLASSES.read_text())


def test_pinned_labels_drive_gating(tmp_path, monkeypatch) -> None:
    """WRITE gates, every other label commits unreviewed, unlisted names keep the heuristics."""
    table = json.loads(PINNED_CLASSES.read_text())
    class_file = tmp_path / "tool_classes.json"
    class_file.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setenv(side_effects.TOOL_CLASS_FILE_ENV, str(class_file))

    known = set(table) | {"claim_done", "never_seen_tool_xyz"}
    wrong = [
        name
        for name, expected in table.items()
        if action_mutates_state({"tool": name, "arguments": {}}, known_tools=known) != expected
    ]
    assert wrong == []
    assert sum(table.values()) == 27
    # Generic tools such as `weather` would be gated by the name heuristics alone.
    monkeypatch.delenv(side_effects.TOOL_CLASS_FILE_ENV)
    assert action_mutates_state({"tool": "weather"}) is True
    monkeypatch.setenv(side_effects.TOOL_CLASS_FILE_ENV, str(class_file))
    assert action_mutates_state({"tool": "weather"}) is False
    assert action_mutates_state({"tool": "claim_done"}, known_tools=known) is True
    assert action_mutates_state({"tool": "never_seen_tool_xyz"}, known_tools=known) is True


def test_tool_report_lists_unlabeled() -> None:
    line = tool_classes.report({"a": "write", "b": "read", "c": "generic", "d": "think"})
    assert "total=4" in line and "write=1" in line and "read=1" in line
    assert "c(generic)" in line and "d(think)" in line


# --- session patch ------------------------------------------------------------------------------


PATCHED_FILES = (
    "utils/llm_utils.py",
    "agent/llm_agent.py",
    "cli.py",
    "evaluator/evaluator.py",
    "run.py",
)


def _mini_vita_root(tmp_path: Path) -> Path:
    """The files the patch touches, from git HEAD (the checkout on disk may be patched)."""
    root = tmp_path / "vita"
    for rel in PATCHED_FILES:
        dst = root / "src" / "vita" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(
            subprocess.run(
                ["git", "-C", str(VITA_ROOT), "show", f"HEAD:src/vita/{rel}"],
                capture_output=True, text=True, check=True,
            ).stdout,
            encoding="utf-8",
        )
    return root


@needs_vita
def test_session_patch_is_anchored_and_idempotent(tmp_path) -> None:
    root = _mini_vita_root(tmp_path)
    assert apply_session_patch.apply(root) is True
    assert apply_session_patch.apply(root) is False
    text = {rel: (root / "src" / "vita" / rel).read_text(encoding="utf-8")
            for rel in PATCHED_FILES}
    utils = text["utils/llm_utils.py"]
    assert utils.count("x-specagent-session-id") == 2
    assert utils.count("SPECAGENT_SESSION_ID: contextvars.ContextVar") == 1
    assert utils.count("SPECAGENT_SESSION_ID.get()") == 1
    assert utils.count("VITA_LLM_TIMEOUT_S") == 1
    assert utils.count("import os\n") == 1
    assert text["agent/llm_agent.py"].count("SPECAGENT_SESSION_ID.set") == 2
    assert text["cli.py"].count('if k != "model"') == 3
    assert text["evaluator/evaluator.py"].count("VITA_DEFER_EVAL") == 2
    run_py = text["run.py"]
    assert "_SPEC_EVAL_PENDING" in run_py and "_spec_drain_evals(save_to)" in run_py
    assert "if _SPEC_ASYNC_EVAL:" in run_py
    for rel, source in text.items():
        compile(source, rel, "exec")


@needs_vita
def test_session_patch_fails_closed_on_drift(tmp_path) -> None:
    root = _mini_vita_root(tmp_path)
    utils = root / "src" / "vita" / "utils" / "llm_utils.py"
    utils.write_text(
        utils.read_text(encoding="utf-8").replace(
            '            headers = models[model]["headers"]', "            headers = {}"
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="headers anchor"):
        apply_session_patch.apply(root)


@needs_vita
def test_session_header_reaches_request_and_is_per_agent(tmp_path, monkeypatch) -> None:
    """A patched checkout stamps a distinct x-specagent-session-id per agent instance."""
    for module in ("pydantic", "loguru", "docstring_parser", "requests", "yaml"):
        pytest.importorskip(module)
    root = tmp_path / "vita_pkg"
    shutil.copytree(VITA_ROOT / "src" / "vita", root / "src" / "vita")
    apply_session_patch.apply(root)
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "default:\n  temperature: 0.0\n  headers: {Content-Type: application/json}\n"
        "models:\n  - name: agent\n    base_url: http://127.0.0.1:1/v1/chat/completions\n"
        "    headers: {Authorization: Bearer none}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VITA_MODEL_CONFIG_PATH", str(models_yaml))
    monkeypatch.syspath_prepend(str(root / "src"))
    for module in [name for name in sys.modules if name == "vita" or name.startswith("vita.")]:
        sys.modules.pop(module)

    captured: list[dict] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}, "index": 0}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **kw: captured.append(kw) or _Resp())
    llm_utils = importlib.import_module("vita.utils.llm_utils")
    llm_agent = importlib.import_module("vita.agent.llm_agent")

    def agent():
        return llm_agent.LLMAgent(tools=[], domain_policy="policy {time}", llm="agent",
                                  time="2025-06-21 11:20:00", language="english")

    agent()
    first = llm_utils.SPECAGENT_SESSION_ID.get()
    assert isinstance(first, str) and first.startswith("vita-")
    llm_utils.generate(model="agent", messages=[], tools=[])
    assert captured[-1]["headers"]["x-specagent-session-id"] == first
    agent()
    second = llm_utils.SPECAGENT_SESSION_ID.get()
    assert second != first
    llm_utils.generate(model="agent", messages=[], tools=[])
    assert captured[-1]["headers"]["x-specagent-session-id"] == second


# --- driver -------------------------------------------------------------------------------------


def _args(*argv: str) -> argparse.Namespace:
    return run_eval.build_arg_parser().parse_args(
        ["--benchmark-root", "/vita", "--output-dir", "/out", *argv]
    )


def test_driver_defaults_are_the_paper_setting() -> None:
    args = _args("--mode", "specagent")
    command = run_eval.vita_command(args, {"agent": "a", "user": "u", "evaluator": "e"}, "s.json")
    flags = dict(zip(command[4::2], command[5::2], strict=True))
    assert flags["--domain"] == "delivery"
    assert flags["--max-steps"] == "100"
    assert flags["--max-concurrency"] == "8"
    assert flags["--num-trials"] == "1"
    assert "--task-ids" not in command and "--num-tasks" not in command
    assert args.async_eval is True


def test_specagent_mode_starts_the_release_server_in_the_vitabench_setting(tmp_path) -> None:
    args = _args("--mode", "specagent", "--gateway-arg=--horizon=4")
    weak = {"base_url": "http://w:30000", "model": "weak", "auth_scheme": "bearer"}
    strong = {"base_url": "http://s/v1/chat/completions", "model": "strong",
              "auth_scheme": "raw"}
    command = run_eval.gateway_command(args, tmp_path, weak, strong, tmp_path / "classes.json")
    assert command[1:3] == ["-m", "specagent.server"]
    assert command[command.index("--setting") + 1] == "vitabench"
    assert command[command.index("--weak-base-url") + 1] == "http://w:30000/v1"
    assert command[command.index("--strong-base-url") + 1] == "http://s/v1"
    assert command[command.index("--strong-auth-scheme") + 1] == "raw"
    assert command[command.index("--tool-class-file") + 1] == str(tmp_path / "classes.json")
    assert command[-1] == "--horizon=4"
    assert not any("key" in part for part in command)
    # The server parses the same flags into the VitaBench setting plus the override.
    parsed = server.build_gateway_config(server.build_arg_parser().parse_args(command[3:]))
    assert parsed.review_replies is False and parsed.weak_max_tokens == 8192
    assert parsed.horizon == 4


def test_baselines_send_the_paper_request_fields() -> None:
    assert run_eval.BASELINE_AGENT_EXTRA["weak"]["max_tokens"] == 8192
    assert run_eval.BASELINE_AGENT_EXTRA["weak"]["reasoning_effort"] == "high"
    for mode in ("weak", "strong"):
        assert run_eval.BASELINE_AGENT_EXTRA[mode]["chat_template_kwargs"] == {
            "enable_thinking": True
        }


def test_completions_url_normalization() -> None:
    assert run_eval._completions_url("http://x:1/v1") == "http://x:1/v1/chat/completions"
    assert run_eval._completions_url("http://x:1/") == "http://x:1/v1/chat/completions"
    assert run_eval._api_base("http://x:1/v1/chat/completions") == "http://x:1/v1"


def test_models_yaml_routes_only_the_agent_and_dedupes_shared_models() -> None:
    from configure_models import build_models_yaml

    config, names = build_models_yaml(
        agent_name="specagent", agent_base_url="http://gw/v1/chat/completions",
        user_name="judge-model", user_base_url="http://j/v1/chat/completions",
        evaluator_name="judge-model", evaluator_base_url="http://j/v1/chat/completions",
    )
    assert names == {"agent": "specagent", "user": "judge-model",
                     "evaluator": "judge-model::evaluator"}
    entries = {entry["name"]: entry for entry in config["models"]}
    assert entries["specagent"]["base_url"] == "http://gw/v1/chat/completions"
    assert entries["judge-model"]["temperature"] == 0.7
    assert entries["judge-model::evaluator"]["model"] == "judge-model"


# --- the gateway in the VitaBench setting ------------------------------------------------------


def _vita_tool(name: str) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": name,
                         "parameters": {"type": "object", "properties": {}}}}


def _completion(*calls: tuple[str, str], content: str | None = None) -> dict:
    tool_calls = [{"id": call_id, "type": "function",
                   "function": {"name": name, "arguments": "{}"}} for call_id, name in calls]
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                                                 "tool_calls": tool_calls or None},
                         "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


class _Backend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.models: list[str] = []

    def complete(self, request, *, model, **kwargs):
        self.models.append(model)
        return self.responses.popleft()


def _gateway(backend, monkeypatch) -> SpeculativeToolGateway:
    monkeypatch.setenv(side_effects.TOOL_CLASS_FILE_ENV, str(PINNED_CLASSES))
    args = server.build_arg_parser().parse_args(
        ["--weak-base-url", "http://w/v1", "--weak-model", "weak",
         "--strong-base-url", "http://s/v1", "--strong-model", "strong",
         "--setting", "vitabench"]
    )
    return SpeculativeToolGateway(backend, server.build_gateway_config(args))


def _request(*names: str) -> dict:
    return {"model": "specagent", "tools": [_vita_tool(name) for name in names],
            "messages": [{"role": "system", "content": "policy"},
                         {"role": "assistant", "content": "Hello, how can I help you?"},
                         {"role": "user", "content": "Order noodles."}]}


def test_generic_tool_and_reply_commit_unreviewed(monkeypatch) -> None:
    backend = _Backend([_completion(("c1", "weather"))])
    decision = _gateway(backend, monkeypatch).handle(_request("weather"))
    assert decision.source == "weak_read"
    backend = _Backend([_completion(content="Which store?")])
    decision = _gateway(backend, monkeypatch).handle(_request("weather"))
    assert (decision.source, decision.boundary) == ("runtime_auto_accept", "message")
    assert backend.models == ["weak"]


def test_write_labelled_call_is_reviewed(monkeypatch) -> None:
    verdict = _completion(content=json.dumps(
        {"reason": None, "accepted_steps": 1, "replacement_action": None}))
    backend = _Backend([_completion(("c1", "create_delivery_order")), verdict])
    decision = _gateway(backend, monkeypatch).handle(_request("create_delivery_order"))
    assert decision.source == "weak_verified"
    assert backend.models == ["weak", "strong"]
