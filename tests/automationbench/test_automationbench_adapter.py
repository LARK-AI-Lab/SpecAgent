"""Offline tests for the AutomationBench accounting adapter and the runner's pure helpers.

These run in the repository environment: ``specagent_adapter`` must import without
verifiers/openai/datasets. The verifiers-bound behaviour (environment reset, concurrency,
transport spans, auth header) lives in ``test_automationbench_env.py``.
"""

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INTEG = ROOT / "benchmarks" / "automationbench"
for _p in (str(ROOT / "src"), str(INTEG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import specagent_adapter as adapter  # noqa: E402

from specagent.protocol.accounting import (  # noqa: E402
    InferencePhase,
    ParsedUsage,
    QuantityStatus,
    missing_transport_attempts,
    transport_attempt_count,
)
from specagent.protocol.types import ModelRole  # noqa: E402

# run.py has a generic basename; load it under a unique module name.
_RUN_SPEC = spec_from_file_location("automationbench_run", INTEG / "run.py")
assert _RUN_SPEC and _RUN_SPEC.loader
runner = module_from_spec(_RUN_SPEC)
_RUN_SPEC.loader.exec_module(runner)


def _state(task_name, trajectory_id, example_id=0, info_as_string=False):
    info = {
        "task_name": task_name,
        "initial_state": {},
        "assertions": [],
        "zapier_tools": [],
    }
    return {
        "info": json.dumps(info) if info_as_string else info,
        "trajectory_id": trajectory_id,
        "example_id": example_id,
        "prompt": [{"role": "user", "content": "hi"}],
    }


def _recorder(role=ModelRole.WEAK, phase=InferencePhase.ACT, **kw):
    return adapter.AttemptRecorder(role=role, phase=phase, **kw)


def _usage(input_t=10, output_t=5):
    return ParsedUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        input_status=QuantityStatus.MEASURED,
        output_status=QuantityStatus.MEASURED,
        provenance="openai_usage",
    )


# Identity ------------------------------------------------------------------


def test_task_and_trial_identity_stable_and_distinct():
    rec = _recorder()
    ctx_a1 = rec.begin_request(_state("task_a", "traj-1"))
    ctx_a2 = rec.begin_request(_state("task_a", "traj-1"))
    ctx_b = rec.begin_request(_state("task_b", "traj-2"))
    ctx_a_retry = rec.begin_request(_state("task_a", "traj-3"))  # re-rolled task

    # task_id is the benchmark task name: stable across trials of one task.
    assert ctx_a1.task_id == ctx_a_retry.task_id == "task_a"
    assert ctx_b.task_id == "task_b" != ctx_a1.task_id
    # trial_id is the rollout's trajectory_id: unique per attempt.
    assert ctx_a1.trial_id == "traj-1" == ctx_a2.trial_id
    assert ctx_a_retry.trial_id == "traj-3" != ctx_a1.trial_id
    # Request ids are per-trial sequences and cannot collide across tasks.
    assert ctx_a1.request_id == "traj-1#r0001"
    assert ctx_a2.request_id == "traj-1#r0002"
    assert ctx_b.request_id == "traj-2#r0001"
    assert ctx_a_retry.request_id == "traj-3#r0001"
    assert len({ctx_a1.request_id, ctx_a2.request_id, ctx_b.request_id}) == 3


def test_task_id_from_json_string_info_and_fallback():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_json", "t1", info_as_string=True))
    assert ctx.task_id == "task_json"
    ctx2 = rec.begin_request({"info": {}, "example_id": 7, "trajectory_id": "t2"})
    assert ctx2.task_id == "example_7"
    ctx3 = rec.begin_request(None)
    assert ctx3.attribution == "run_unattributed"
    assert rec.anomalies and rec.anomalies[-1]["kind"] == "unbound_state"


def test_adapter_module_resolves_to_one_object_under_both_names():
    # CURRENT_REQUEST is process-global identity; a duplicate module would split request
    # binding from transport recording.
    assert sys.modules["specagent_adapter"] is adapter
    assert sys.modules["benchmarks.automationbench.specagent_adapter"] is adapter


# Failed / retried / cancelled attempts -------------------------------------


def test_failed_and_cancelled_attempts_produce_spans():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    s1 = rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.5,
                            parsed=None, status="failed")
    s2 = rec.record_attempt(ctx, attempt_index=2, duration_seconds=0.7,
                            parsed=_usage(), status="ok")
    ctx2 = rec.begin_request(_state("task_a", "traj-1"))
    s3 = rec.record_attempt(ctx2, attempt_index=1, duration_seconds=0.1,
                            status="cancelled")

    assert s1.status == "failed" and s1.usage_status is QuantityStatus.UNKNOWN
    assert s2.status == "ok" and s2.output_tokens == 5
    assert s2.committed_output_tokens == 5 and s1.committed_output_tokens == 0
    assert s3.status == "cancelled"
    # Contiguous indices per request: no inferred holes.
    assert missing_transport_attempts(rec.ledger.spans) == 0
    assert transport_attempt_count(rec.ledger.spans) == 3
    summary = rec.ledger.summarize()
    assert summary.n_failed_calls == 2
    assert summary.n_transport_attempts == 3
    # One measured attempt plus two with unknown usage: totals degrade to incomplete rather
    # than pretending a full measurement.
    assert summary.token_totals_status == QuantityStatus.INCOMPLETE.value


def test_non_contiguous_attempt_indices_count_as_holes():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.1, parsed=_usage())
    rec.record_attempt(ctx, attempt_index=3, duration_seconds=0.1, parsed=_usage())
    assert rec.transport_holes() == 1


# Missing usage stays unknown ------------------------------------------------


def test_missing_usage_never_zero_filled():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    span = rec.record_attempt(ctx, attempt_index=1, duration_seconds=1.0)
    assert span.usage_status is QuantityStatus.UNKNOWN
    for status in (
        span.resolved_input_status,
        span.resolved_output_status,
        span.resolved_cached_status,
        span.resolved_reasoning_status,
    ):
        assert status is QuantityStatus.UNKNOWN
    assert span.input_tokens == 0 and span.output_tokens == 0
    assert span.flops.status is QuantityStatus.UNKNOWN
    assert span.hbm_read_bytes.status is QuantityStatus.UNKNOWN
    assert rec.ledger.summarize().token_totals_status == QuantityStatus.UNKNOWN.value


def test_parse_response_body_variants():
    body = json.dumps(
        {
            "id": "chatcmpl-1",
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
    ).encode()
    parsed, provider_id = adapter.parse_response_body(body)
    assert provider_id == "chatcmpl-1"
    assert parsed.input_tokens == 11 and parsed.output_tokens == 7
    assert parsed.cached_input_tokens == 3 and parsed.reasoning_tokens == 2
    assert parsed.input_status is QuantityStatus.MEASURED

    missing, provider_id2 = adapter.parse_response_body(b'{"id": "x"}')
    assert missing.status is QuantityStatus.UNKNOWN
    assert missing.provenance == "missing_usage"
    assert provider_id2 == "x"

    garbage, _ = adapter.parse_response_body(b"not json")
    assert garbage.status is QuantityStatus.UNKNOWN
    none_parsed, _ = adapter.parse_response_body(None)
    assert none_parsed.status is QuantityStatus.UNKNOWN


def test_parsed_usage_from_response_prefers_usage_attribute():
    class _Usage:
        def model_dump(self):
            return {"prompt_tokens": 4, "completion_tokens": 2}

    class _Completion:
        usage = _Usage()
        id = "cmpl-attr"

    parsed, provider_id = adapter.parsed_usage_from_response(_Completion())
    assert (parsed.input_tokens, parsed.output_tokens) == (4, 2)
    assert provider_id == "cmpl-attr"

    class _HttpResponse:
        content = json.dumps(
            {"id": "cmpl-body", "usage": {"prompt_tokens": 9, "completion_tokens": 1}}
        ).encode()

    parsed, provider_id = adapter.parsed_usage_from_response(_HttpResponse())
    assert (parsed.input_tokens, parsed.output_tokens) == (9, 1)
    assert provider_id == "cmpl-body"


# Model-role attribution -------------------------------------------------------


@pytest.mark.parametrize(
    "role,phase,intent,expected",
    [
        (ModelRole.WEAK, InferencePhase.ACT, None, "weak_act"),
        (ModelRole.STRONG, InferencePhase.ACT, None, "strong_act"),
        (ModelRole.STRONG, InferencePhase.VERIFICATION, None, "strong_verify"),
        (ModelRole.STRONG, InferencePhase.CORRECTION, None, "strong_correct"),
        (ModelRole.WEAK, InferencePhase.ACT, "retry", "weak_retry"),
        (ModelRole.WEAK, InferencePhase.SUMMARY, None, "summary"),
    ],
)
def test_role_phase_intent_attribution(role, phase, intent, expected):
    rec = _recorder(role=role, phase=phase, intent=intent)
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    span = rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.1,
                              parsed=_usage())
    assert span.role is role
    assert span.canonical_phase.value == expected


def test_record_attempt_overrides_recorder_defaults():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    span = rec.record_attempt(
        ctx,
        attempt_index=1,
        duration_seconds=0.1,
        parsed=_usage(),
        role="strong",
        phase=InferencePhase.VERIFICATION,
        request_id=f"{ctx.request_id}:strong_verify:1",
    )
    assert span.role is ModelRole.STRONG
    assert span.phase is InferencePhase.VERIFICATION
    assert span.request_id == "traj-1#r0001:strong_verify:1"
    assert span.parent_task_id == "task_a"


# Benchmark infrastructure cost is isolated from agent generation -------------


def test_infra_cost_isolated_from_agent_generation():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.record_attempt(ctx, attempt_index=1, duration_seconds=1.0, parsed=_usage())

    payload = adapter.paper_payload(rec.ledger, model="unknown-model-xyz")
    assert payload is not None
    assert payload["cost_scope"] == "agent_generation_only"
    out = payload["out_of_scope"]
    assert out["judge"]["status"] == "excluded" and out["judge"]["value"] is None
    assert out["exclusive_gpu_seconds"]["status"] == "unknown"
    # The ledger contains only agent spans: scorer/env/tool work is pure Python inside the
    # rollout and is never recorded as inference.
    for span in rec.ledger.spans:
        assert span.role in (ModelRole.WEAK, ModelRole.STRONG)
        assert span.charge_inference


# Analysis-consumer compatibility ------------------------------------------------


def test_span_records_and_paper_payload_consumable():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.4, parsed=_usage())
    rec.record_attempt(ctx, attempt_index=2, duration_seconds=0.6, parsed=_usage())

    record = rec.ledger.spans[0].to_record()
    for key in (
        "schema_version",
        "canonical_phase",
        "request_id",
        "transport_attempt_id",
        "transport_attempt_index",
        "parent_task_id",
        "parent_episode_id",
        "flops",
        "hbm_read_bytes",
        "usage_status",
        "status",
        "attribution",
    ):
        assert key in record
    assert record["parent_task_id"] == "task_a"
    assert record["parent_episode_id"] == "traj-1"
    assert record["transport_attempt_id"] == "traj-1#r0001:1"

    payload = adapter.paper_payload(rec.ledger, model="unknown-model")
    hierarchy = payload["hierarchy"]
    assert set(hierarchy) == {"attempt", "request", "task", "benchmark"}
    task_rows = hierarchy["task"]
    assert task_rows and task_rows[0]["key"] == "task_a"
    request_rows = hierarchy["request"]
    assert request_rows[0]["key"] == "traj-1#r0001"
    # Unknown architecture: estimates stay unknown, never fabricated.
    bench = hierarchy["benchmark"][0]
    assert bench["flops"]["status"] in ("unknown", "incomplete")


def test_spec_lookup_for_known_and_unknown_models():
    from specagent.protocol.resource_estimate import (
        resolve_architecture_id,
        serving_spec_from_model_name,
    )

    # Unknown name: a name-only spec; enrichment marks it unsupported and never fabricates an
    # estimate.
    unknown_lookup = adapter.spec_lookup_for_model("definitely-not-a-model")
    assert unknown_lookup is not None
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.5, parsed=_usage())
    payload = adapter.paper_payload(rec.ledger, model="definitely-not-a-model")
    bench = payload["hierarchy"]["benchmark"][0]
    assert bench["flops"]["status"] in ("unknown", "incomplete")
    assert bench["flops"]["value"] is None

    # Registered aliases resolve to architecture-backed specs.
    candidates = ("Qwen3.5-4B", "Qwen/Qwen3.5-4B", "Qwen3.5-397B-A17B")
    resolved = [c for c in candidates if resolve_architecture_id(c)]
    assert resolved, "no Qwen3.5 architecture alias registered"
    spec = serving_spec_from_model_name(resolved[0])
    assert spec is not None and spec.model_id


# Fail-closed classification -------------------------------------------------------


def test_fail_closed_validation():
    cap = 8
    dangling = {
        "name": "t",
        "steps": 2,
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
        ],
    }
    res = adapter.validate_export_task(dangling, step_cap=cap)
    assert not res["ok"] and "unresolved_tool_call" in res["flags"]

    at_cap = dict(dangling, steps=cap)
    res = adapter.validate_export_task(at_cap, step_cap=cap)
    assert "tool_call_at_step_cap" in res["flags"]

    errored = {"name": "t", "steps": 1, "messages": [], "errors": [{"e": 1}]}
    assert "error" in adapter.validate_export_task(errored, step_cap=cap)["flags"]

    refused = {"name": "t", "steps": 1, "messages": [], "died_on_refusal": True}
    assert "refusal" in adapter.validate_export_task(refused, step_cap=cap)["flags"]

    clean = {
        "name": "t",
        "steps": 2,
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "done"},
        ],
    }
    assert adapter.validate_export_task(clean, step_cap=cap)["ok"]


def test_fail_closed_rollout_output():
    cap = 8
    base = {"info": {"task_name": "t"}, "trajectory_id": "tr", "is_completed": True,
            "is_truncated": False, "error": None, "stop_condition": None,
            "completion": [{"role": "assistant", "content": "done"}]}
    assert adapter.validate_rollout_output(base, step_cap=cap)["ok"]

    assert "error" in adapter.validate_rollout_output(
        dict(base, error={"message": "boom"}), step_cap=cap)["flags"]
    assert "timeout" in adapter.validate_rollout_output(
        dict(base, stop_condition="timeout_reached"), step_cap=cap)["flags"]
    assert "incomplete" in adapter.validate_rollout_output(
        dict(base, is_completed=False), step_cap=cap)["flags"]
    assert "truncated" in adapter.validate_rollout_output(
        dict(base, is_truncated=True), step_cap=cap)["flags"]
    flags = adapter.validate_rollout_output(
        dict(base, completion=[{"role": "assistant", "tool_calls": [{"id": "c"}]}],
             trajectory=[{"a": 1}]),
        step_cap=cap,
    )["flags"]
    assert "unresolved_tool_call" in flags


# Run-summary join --------------------------------------------------------------------


def _raw_output(task="task_a", trial="traj-1"):
    return {
        "info": {"task_name": task},
        "trajectory_id": trial,
        "example_id": 0,
        "reward": 1.0,
        "is_completed": True,
        "is_truncated": False,
        "metrics": {"partial_credit": 1.0, "task_completed_correctly": 1.0},
        "timing": {"total": 1.25},
        "completion": [{"role": "assistant", "content": "done"}],
    }


def test_build_run_summary_joins_by_identity():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.5, parsed=_usage())

    export_task = {"name": "task_a", "passed": True, "steps": 1,
                   "messages": [{"role": "assistant", "content": "done"}]}
    summary = adapter.build_run_summary(
        raw_outputs=[_raw_output()],
        export_tasks=[export_task],
        recorder=rec,
        model="m",
        role=ModelRole.WEAK,
        benchmark_makespan_seconds=3.0,
        step_cap=8,
    )
    assert summary["adapter_version"] == adapter.ADAPTER_VERSION == "automationbench_adapter"
    assert summary["cost_scope"] == "agent_generation_only"
    assert summary["role"] == "weak"
    task = summary["tasks"][0]
    assert task["task_id"] == "task_a" and task["trial_id"] == "traj-1"
    assert task["passed"] is True and task["score"] == 1.0
    assert task["n_transport_attempts"] == 1
    assert task["usage"]["input_tokens"] == 10
    assert task["usage"]["input_status"] == "measured"
    assert task["end_to_end_wall_seconds"] == 1.25
    assert summary["benchmark_makespan_seconds"] == 3.0
    assert summary["run_flags"] == []
    assert summary["ledger_totals"]["missing_transport_attempts"] == 0


def test_build_run_summary_flags_missing_export_row_and_unknown_usage():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.record_attempt(ctx, attempt_index=1, duration_seconds=0.5, parsed=_usage())
    rec.record_attempt(ctx, attempt_index=2, duration_seconds=0.5, status="failed")

    output = dict(_raw_output(), timing={"total": 0.0})
    summary = adapter.build_run_summary(
        raw_outputs=[output],
        export_tasks=[],
        recorder=rec,
        model="m",
        role="strong",
        benchmark_makespan_seconds=None,
        step_cap=8,
    )
    task = summary["tasks"][0]
    assert task["export_validation"] == {"ok": None, "flags": ["missing_export_row"]}
    assert "task_a:missing_export_row" in summary["run_flags"]
    # A scoring total of 0.0 means scoring never ran: unknown, not a real zero.
    assert task["end_to_end_wall_seconds"] is None
    assert task["n_failed_attempts"] == 1
    assert task["usage"]["input_status"] == "incomplete"
    assert summary["role"] == "strong"


# Dependency hygiene -----------------------------------------------------------------


@pytest.mark.parametrize(
    "module", ["specagent_adapter", "rp_backend", "decision_fields", "rp_fused_protocol"]
)
def test_pure_modules_import_without_benchmark_deps(module):
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "sys.path.insert(0, 'benchmarks/automationbench');"
        f"import {module};"
        "forbidden = {'verifiers', 'automationbench', 'openai', 'datasets', 'httpx'};"
        "leaked = forbidden & set(sys.modules);"
        "assert not leaked, leaked"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# Runner: arguments, output dir, endpoint env -------------------------------------------


def test_runner_defaults_to_the_paper_setting():
    args = runner._parse_args([])
    assert args.protocol == "rp_fused"
    assert args.mode == "full"
    assert args.domains == "finance,hr,marketing,operations,sales,support"
    assert args.max_turns == 50
    assert args.role == "weak"
    assert runner._parse_args(["--protocol", "solo", "--role", "strong"]).role == "strong"


@pytest.mark.parametrize(
    "argv", [["--protocol", "unknown"], ["--mode", "unknown"], ["--role", "judge"]]
)
def test_runner_rejects_removed_or_unknown_choices(argv, capsys):
    with pytest.raises(SystemExit):
        runner._parse_args(argv)


def test_runner_out_dir_rules(tmp_path):
    with pytest.raises(SystemExit):
        runner._out_dir(runner._parse_args(["--mode", "full"]))
    with pytest.raises(SystemExit):
        runner._out_dir(runner._parse_args(["--mode", "smoke"]))
    full = runner._out_dir(runner._parse_args(["--mode", "full", "--run-name", "r1"]))
    assert full == runner.DUMPS_ROOT / "r1"
    smoke = runner._out_dir(runner._parse_args(["--mode", "smoke", "--out-dir", str(tmp_path)]))
    assert smoke == tmp_path


def test_runner_role_endpoint_requires_every_variable(monkeypatch):
    for name in ("BASE_URL", "API_KEY", "MODEL"):
        monkeypatch.delenv(f"SPECAGENT_STRONG_{name}", raising=False)
    monkeypatch.setenv("SPECAGENT_STRONG_BASE_URL", "http://strong.test/v1")
    with pytest.raises(SystemExit) as excinfo:
        runner._role_endpoint("strong")
    message = str(excinfo.value)
    assert "SPECAGENT_STRONG_API_KEY" in message and "SPECAGENT_STRONG_MODEL" in message
    assert "SPECAGENT_STRONG_BASE_URL" not in message

    monkeypatch.setenv("SPECAGENT_STRONG_API_KEY", "k")
    monkeypatch.setenv("SPECAGENT_STRONG_MODEL", "strong-model")
    assert runner._role_endpoint("strong") == ("http://strong.test/v1", "k", "strong-model")


# Manifest freeze (baseline mode) ---------------------------------------------------------


def _fake_dataset_rows(n=5):
    return [
        {
            "example_id": i,
            "info": {"task_name": f"task_{i}", "domain": "sales"},
            "prompt": [],
        }
        for i in range(n)
    ]


def _freeze(path, rows, **overrides):
    kwargs = dict(
        run_name="t", model="m", base_url="u", domains=["sales"], dataset=rows,
        rollouts_per_example=3, shard_size=2, max_turns=8, toolset="limited_zapier",
    )
    kwargs.update(overrides)
    return runner._freeze_manifest(path, **kwargs)


def test_manifest_freeze_and_mutation_refusal(tmp_path):
    rows = _fake_dataset_rows(5)
    path = tmp_path / "manifest.json"
    m = _freeze(path, rows)
    assert m["planned_trajectories"] == 15
    assert m["protocol"] == "solo" and m["rp_fused_profile"] is None
    assert len(m["tasks"]) == 5 and len(m["shards"]) == 3
    assert m["shard_status"] == {"0": "pending", "1": "pending", "2": "pending"}
    # Task ids are joinable and the shard plan covers all indices.
    assert [t["task_id"] for t in m["tasks"]] == [f"task_{i}" for i in range(5)]
    assert sorted(i for s in m["shards"] for i in s) == [0, 1, 2, 3, 4]

    # The same plan re-freezes idempotently (resume path).
    m2 = _freeze(path, rows)
    assert m2["tasks"] == m["tasks"]

    # A different plan must not silently overwrite the frozen design.
    with pytest.raises(SystemExit):
        _freeze(path, rows, rollouts_per_example=1)


def test_manifest_freeze_pins_the_fused_profile(tmp_path):
    import rp_fused_protocol as rfp

    rows = _fake_dataset_rows(2)
    path = tmp_path / "manifest.json"
    profile = json.loads(json.dumps(rfp.MANIFEST_PROFILE))
    m = _freeze(
        path, rows, protocol="rp_fused", strong_model="s", strong_base_url="su",
        rp_fused_profile=profile,
    )
    assert m["rp_fused_profile"] == profile
    assert json.loads(path.read_text())["rp_fused_profile"] == profile

    # Resuming under a changed profile is refused.
    changed = dict(profile, horizon=profile["horizon"] + 1)
    with pytest.raises(SystemExit):
        _freeze(
            path, rows, protocol="rp_fused", strong_model="s", strong_base_url="su",
            rp_fused_profile=changed,
        )


def test_recorder_rank_sidecar():
    rec = _recorder()
    ctx = rec.begin_request(_state("task_a", "traj-1"))
    rec.note_rank(ctx.request_id, dp_rank=3, route_key="abc", route_source="session")
    assert rec.rank_rows == [
        {"request_id": ctx.request_id, "dp_rank": 3,
         "route_key": "abc", "route_source": "session"}
    ]
