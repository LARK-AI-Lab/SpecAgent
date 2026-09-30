"""AutomationBench adapter tests that need verifiers + automationbench.

They skip unless both packages are importable (run them inside the AutomationBench
environment). They exercise the real ``AutomationBenchEnv`` and the recording client end to end
with a stub transport: no network, no model calls.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers")
pytest.importorskip("automationbench")
httpx = pytest.importorskip("httpx")
datasets = pytest.importorskip("datasets")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT / "benchmarks" / "automationbench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import solo_client  # noqa: E402
import specagent_adapter as adapter  # noqa: E402
from automationbench.rubric import create_rubric  # noqa: E402
from automationbench.runner import AutomationBenchEnv  # noqa: E402

from specagent.protocol.accounting import InferencePhase, QuantityStatus  # noqa: E402
from specagent.protocol.types import ModelRole  # noqa: E402


def _completion_body(*, content="done", tool_calls=None, usage=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _response(**kwargs):
    return httpx.Response(
        200,
        content=json.dumps(_completion_body(**kwargs)).encode(),
        headers={"content-type": "application/json"},
    )


_USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}


def _gmail_tool_call(to="bob@example.com"):
    return [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "gmail_send_email",
                "arguments": json.dumps({"to": to, "subject": "hi", "body": "hello"}),
            },
        }
    ]


class _FakeAPI:
    """Stands in for ``AsyncOpenAI``: ``post`` and ``base_url`` are all verifiers uses."""

    def __init__(self, script):
        self._script = list(script)
        self.posts = []
        self.base_url = "http://fake.local/v1"

    async def post(self, path, *, cast_to=None, body=None, options=None, **kw):
        self.posts.append({"path": path, "body": body, "options": options})
        if not self._script:
            raise RuntimeError("fake transport: script exhausted")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _state(task_name="task_a", trajectory_id="traj-1", example_id=0):
    return {
        "info": {
            "task_name": task_name,
            "initial_state": {},
            "assertions": [],
            "zapier_tools": [],
        },
        "trajectory_id": trajectory_id,
        "example_id": example_id,
    }


def _recorder(role=ModelRole.WEAK):
    return adapter.AttemptRecorder(role=role, phase=InferencePhase.ACT, model="fake")


def _client(script, recorder=None, authorization=None):
    rec = recorder or _recorder()
    client = solo_client.SpecAgentChatClient(
        _FakeAPI(script), recorder=rec, authorization=authorization
    )
    return client, rec


def _run(coro):
    return asyncio.run(coro)


def _patch_retry(monkeypatch, attempts=3):
    import automationbench.clients as ab_clients

    monkeypatch.setattr(ab_clients, "_RETRY_MAX_ATTEMPTS", attempts)
    monkeypatch.setattr(ab_clients, "_retry_delay", lambda *a, **k: 0.0)


def _ask(client, state):
    return client.get_native_response(
        [{"role": "user", "content": "hi"}], "fake", {}, None, state=state
    )


# Transport-level span recording -------------------------------------------------------


def test_successful_call_records_one_span():
    client, rec = _client([_response(usage=_USAGE)])
    _run(_ask(client, _state()))
    spans = rec.ledger.spans
    assert len(spans) == 1
    span = spans[0]
    assert span.status == "ok"
    assert span.request_id == "traj-1#r0001"
    assert span.transport_attempt_index == 1
    assert span.transport_attempt_id == "traj-1#r0001:1"
    assert span.parent_task_id == "task_a"
    assert span.parent_episode_id == "traj-1"
    assert span.input_tokens == 10 and span.output_tokens == 5
    assert span.usage_status is QuantityStatus.MEASURED


def test_retry_produces_contiguous_attempt_spans(monkeypatch):
    _patch_retry(monkeypatch)
    client, rec = _client([RuntimeError("boom"), _response(usage=_USAGE)])
    _run(_ask(client, _state()))
    spans = rec.ledger.spans
    assert len(spans) == 2
    assert spans[0].status == "failed" and spans[0].transport_attempt_index == 1
    assert spans[0].usage_status is QuantityStatus.UNKNOWN
    assert spans[1].status == "ok" and spans[1].transport_attempt_index == 2
    assert spans[0].request_id == spans[1].request_id


def test_exhausted_retry_records_every_failed_attempt(monkeypatch):
    _patch_retry(monkeypatch, attempts=2)
    client, rec = _client([RuntimeError("x"), RuntimeError("y")])
    with pytest.raises(RuntimeError):
        _run(_ask(client, _state()))
    spans = rec.ledger.spans
    assert len(spans) == 2
    assert all(s.status == "failed" for s in spans)
    assert [s.transport_attempt_index for s in spans] == [1, 2]


def test_cancelled_attempt_recorded_and_propagates(monkeypatch):
    _patch_retry(monkeypatch)
    client, rec = _client([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        _run(_ask(client, _state()))
    assert len(rec.ledger.spans) == 1
    assert rec.ledger.spans[0].status == "cancelled"


def test_missing_usage_response_stays_unknown():
    client, rec = _client([_response(usage=None)])
    _run(_ask(client, _state()))
    span = rec.ledger.spans[0]
    assert span.status == "ok"
    assert span.usage_status is QuantityStatus.UNKNOWN
    assert span.input_tokens == 0


def test_authorization_and_session_headers_on_wire():
    api = _FakeAPI([_response(usage=_USAGE)])
    rec = _recorder()
    client = solo_client.SpecAgentChatClient(api, recorder=rec, authorization="raw-token-123")
    _run(_ask(client, _state()))
    headers = api.posts[0]["options"]["headers"]
    assert headers["Authorization"] == "raw-token-123"
    assert headers["x-specagent-session-id"] == "0"


def test_unbound_post_is_anomaly_not_crash():
    api = _FakeAPI([_response(usage=_USAGE)])
    rec = _recorder()
    solo_client.SpecAgentChatClient(api, recorder=rec)

    async def go():
        return await api.post("/chat/completions", body={}, cast_to=httpx.Response)

    _run(go())
    assert rec.anomalies and rec.anomalies[0]["kind"] == "unbound_post"
    assert len(rec.ledger.spans) == 0


# Concurrency: no cross-task contamination --------------------------------------------------


def test_concurrent_requests_keep_identity():
    class _InterleavingAPI(_FakeAPI):
        async def post(self, path, *, cast_to=None, body=None, options=None, **kw):
            self.posts.append({"path": path, "body": body, "options": options})
            await asyncio.sleep(0)  # force an interleave point
            return _response(usage=_USAGE)

    rec = _recorder()
    client = solo_client.SpecAgentChatClient(_InterleavingAPI([]), recorder=rec)

    async def go():
        await asyncio.gather(*[_ask(client, _state(f"t{i}", f"tr{i}")) for i in range(4)])

    _run(go())
    by_trial = {}
    for span in rec.ledger.spans:
        by_trial.setdefault(span.parent_episode_id, []).append(span)
    assert set(by_trial) == {"tr0", "tr1", "tr2", "tr3"}
    for trial, spans in by_trial.items():
        assert len(spans) == 1
        assert spans[0].parent_task_id == "t" + trial[2:]


# Environment lifecycle: reset, tool filtering, end-to-end scoring --------------------------


def _task_row(example_id, task_name, recipient_assert):
    # Mirrors the automationbench.domains loaders: the task name lives in info["task_name"].
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": f"Send an email to {recipient_assert}"}],
        "answer": "",
        "info": {
            "task_name": task_name,
            "initial_state": {},
            "assertions": [{"type": "gmail_message_sent", "to": recipient_assert}],
            "zapier_tools": ["gmail_send_email"],
        },
    }


def _env(rows, **kwargs):
    kwargs.setdefault("toolset", "limited_zapier")
    kwargs.setdefault("max_turns", 8)
    return AutomationBenchEnv(
        dataset=datasets.Dataset.from_list(rows), rubric=create_rubric(), **kwargs
    )


def _evaluate(env, client, num_examples, state_columns=("trajectory_id",)):
    return _run(
        env.evaluate(
            client=client,
            model="fake",
            sampling_args={},
            num_examples=num_examples,
            rollouts_per_example=1,
            max_concurrent=1,
            state_columns=list(state_columns),
        )
    )


def test_setup_state_builds_fresh_world_per_task():
    env = _env([_task_row(0, "t_a", "a@x.com"), _task_row(1, "t_b", "b@x.com")])
    st_a = {"info": dict(_task_row(0, "t_a", "a@x.com")["info"]), "prompt": [], "example_id": 0}
    st_b = {"info": dict(_task_row(1, "t_b", "b@x.com")["info"]), "prompt": [], "example_id": 1}

    async def go():
        return await env.setup_state(st_a), await env.setup_state(st_b)

    sa, sb = _run(go())
    assert sa["world"] is not sb["world"]
    sa["world"].gmail.messages.append(object())
    assert len(sb["world"].gmail.messages) == 0
    # Per-task tool filtering applies to rollout-local state.
    assert [t.name for t in sa["tool_defs"]] == ["gmail_send_email"]
    # Contract fingerprint captured before normalization.
    assert sa["_task_contract_sha256"]


def test_setup_state_rebuilds_world_on_repeat_call():
    env = _env([_task_row(0, "t_a", "a@x.com")])
    info = dict(_task_row(0, "t_a", "a@x.com")["info"])

    async def go():
        st1 = await env.setup_state({"info": info, "prompt": [], "example_id": 0})
        st1["world"].gmail.messages.append(object())
        st2 = await env.setup_state({"info": info, "prompt": [], "example_id": 0})
        return st1, st2

    st1, st2 = _run(go())
    assert st1["world"] is not st2["world"]
    assert len(st2["world"].gmail.messages) == 0


def test_env_evaluate_end_to_end_offline():
    """Full rollout: the tool call mutates the world, the assertion scores 1.0, spans join."""
    env = _env([_task_row(0, "t_pass", "bob@example.com")])
    rec = _recorder()
    script = [
        _response(content=None, tool_calls=_gmail_tool_call("bob@example.com"),
                  usage=_USAGE, finish="tool_calls"),
        _response(content="Sent the email.", usage=_USAGE),
    ]
    client = solo_client.SpecAgentChatClient(_FakeAPI(script), recorder=rec)
    results = _evaluate(
        env, client, 1, state_columns=("trajectory_id", "_assertion_results", "_perf")
    )
    outputs = results["outputs"]
    assert len(outputs) == 1
    out = outputs[0]
    assert out["reward"] == 1.0
    assert out["metrics"]["task_completed_correctly"] == 1.0
    trial = out["trajectory_id"]
    assert trial

    spans = [s for s in rec.ledger.spans if s.parent_episode_id == trial]
    assert len(spans) == 2
    assert all(s.parent_task_id == "t_pass" for s in spans)
    assert all(s.status == "ok" for s in spans)

    summary = adapter.build_run_summary(
        raw_outputs=outputs,
        export_tasks=[
            {"name": "t_pass", "passed": True, "steps": 2,
             "messages": [{"role": "assistant", "content": "Sent the email."}]}
        ],
        recorder=rec,
        model="fake",
        role=ModelRole.WEAK,
        benchmark_makespan_seconds=1.0,
        step_cap=8,
    )
    task = summary["tasks"][0]
    assert task["task_id"] == "t_pass" and task["trial_id"] == trial
    assert task["n_transport_attempts"] == 2
    assert task["usage"]["input_tokens"] == 20
    assert task["usage"]["output_status"] == "measured"


def test_world_state_not_shared_across_tasks():
    """If per-task worlds leaked, task B would score on task A's writes."""
    env = _env([
        _task_row(0, "t_send", "bob@example.com"),
        _task_row(1, "t_nosend", "bob@example.com"),
    ])
    rec = _recorder()
    script = [
        # Task A sends to bob and passes.
        _response(content=None, tool_calls=_gmail_tool_call("bob@example.com"),
                  usage=_USAGE, finish="tool_calls"),
        _response(content="done", usage=_USAGE),
        # Task B never calls the tool and must fail (fresh world, no leak).
        _response(content="done", usage=_USAGE),
    ]
    client = solo_client.SpecAgentChatClient(_FakeAPI(script), recorder=rec)
    results = _evaluate(env, client, 2)
    outputs = sorted(results["outputs"], key=lambda o: o["example_id"])
    assert outputs[0]["reward"] == 1.0  # t_send
    assert outputs[1]["reward"] == 0.0  # t_nosend: proves no leaked world
    assert len({o["trajectory_id"] for o in outputs}) == 2
    for out in outputs:
        spans = [s for s in rec.ledger.spans if s.parent_episode_id == out["trajectory_id"]]
        expected_task = "t_send" if out["example_id"] == 0 else "t_nosend"
        assert spans and all(s.parent_task_id == expected_task for s in spans)


def test_rollout_transport_failure_fails_closed(monkeypatch):
    """A rollout whose transport dies is aborted by the framework: no reward is fabricated and
    the failed spans stay in the ledger bound to the right task."""
    _patch_retry(monkeypatch, attempts=2)
    env = _env([_task_row(0, "t_fail", "x@x.com")])
    rec = _recorder()
    client = solo_client.SpecAgentChatClient(
        _FakeAPI([RuntimeError("down"), RuntimeError("down")]), recorder=rec
    )
    results = _evaluate(env, client, 1)
    assert len(results["outputs"]) == 1
    out = results["outputs"][0]
    assert out["error"] is not None
    assert not out["reward"]

    spans = rec.ledger.spans
    assert len(spans) == 2
    assert all(s.status == "failed" for s in spans)
    assert all(s.parent_task_id == "t_fail" for s in spans)
    assert all(s.usage_status is QuantityStatus.UNKNOWN for s in spans)
