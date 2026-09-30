"""AutomationBench runner under SpecAgent accounting.

Protocols:

* ``--protocol rp_fused`` (default): the SpecAgent profile of ``rp_fused_protocol.py``. Weak drafts,
  Strong reviews held state changes and replies, and a rejection is repaired by the verifier's
  replacement or one Strong turn. Uses both SPECAGENT_WEAK_* and SPECAGENT_STRONG_* endpoints.
* ``--protocol solo``: the single-model baseline of ``--role`` (weak or strong).

Modes:

* ``--mode full`` (default): the paper run over the six public domains, 50 turns per task.
  Requires ``--run-name``; writes to ``dumps/automationbench/<run_name>/``. Freezes a
  ``manifest.json`` (task set, k rollouts per task, shard plan) before any model call, then runs
  shard by shard with per-shard checkpoints so a crashed run resumes with ``--resume``.

Artifacts per run dir:

* ``manifest.json``            frozen task/shard design plus live status
* ``export.json``              the benchmark's own export (official scorer)
* ``specagent_spans.jsonl``    one InferenceSpan.to_record() per transport attempt
* ``specagent_resources.json`` paper_resources_from_ledger (agent generation only)
* ``run_summary.json``         task/trial/span join plus fail-closed flags
* ``rank_map.jsonl``           request_id -> dp_rank / route_key (sidecar)
* ``rp_decisions.jsonl``       one provenance row per gateway request (rp_fused only)
* ``shards/shard_NNNN.json``   per-shard outputs and flags (resume checkpoints)

``exclusive_gpu_seconds`` stays UNKNOWN: a shared data-parallel rank cannot attribute exclusive
compute. Per-request ``duration_seconds`` and ``rank_map.jsonl`` are the inputs for a labeled
``duration x 1 GPU`` proxy, which includes queueing.

Run inside the AutomationBench environment:

    SPECAGENT_WEAK_BASE_URL=... SPECAGENT_WEAK_API_KEY=... SPECAGENT_WEAK_MODEL=... \
    SPECAGENT_STRONG_BASE_URL=... SPECAGENT_STRONG_API_KEY=... SPECAGENT_STRONG_MODEL=... \
    python benchmarks/automationbench/run.py --run-name specagent

* ``--mode smoke``: at most ``MAX_SMOKE_TASKS`` tasks, written under ``--out-dir``, e.g.
  ``--mode smoke --domains simple --num-examples 1 --out-dir out/smoke``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (str(REPO_ROOT / "src"), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_SMOKE_TASKS = 3
STATE_COLUMNS = [
    "_usage",
    "_debug",
    "_assertion_results",
    "_end_state",
    "_perf",
    "trajectory_id",
]
DUMPS_ROOT = REPO_ROOT / "dumps" / "automationbench"


def _j(value: Any) -> Any:
    """Normalize verifiers/pydantic values to plain JSON structures.

    ``json.dumps(default=str)`` would stringify Message objects, breaking
    resume-time consumers (e.g. export_results) that expect dicts.
    """
    return json.loads(
        json.dumps(
            value,
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
        )
    )


def _bench_root() -> Path:
    root = Path(
        os.environ.get("AUTOMATIONBENCH_ROOT")
        or (REPO_ROOT / "third_party" / "AutomationBench")
    )
    if not (root / "automationbench" / "runner.py").is_file():
        raise SystemExit(
            f"AutomationBench checkout unavailable at {root}; "
            "set AUTOMATIONBENCH_ROOT or vendor third_party/AutomationBench"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--run-name", default=None,
                        help="full: dumps/automationbench/<run-name>/")
    parser.add_argument("--protocol", choices=["rp_fused", "solo"], default="rp_fused",
                        help="rp_fused: route turns through the SpecAgent gateway with "
                        "the AutomationBench profile. solo: the single-model baseline "
                        "of --role.")
    parser.add_argument("--role", choices=["weak", "strong"], default="weak",
                        help="solo: which model runs alone. rp_fused: recorded as the "
                        "nominal actor role; both endpoints are used.")
    parser.add_argument(
        "--domains",
        default="finance,hr,marketing,operations,sales,support",
        help="comma-separated; the default is the six public domains (600 tasks)",
    )
    parser.add_argument("--tasks", default=None, help="comma-separated task names")
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=8,
                        help="full: tasks per checkpoint shard")
    parser.add_argument("--resume", action="store_true",
                        help="full: skip shards already checkpointed")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument(
        "--toolset",
        default="limited_zapier",
        choices=["zapier", "limited_zapier", "api"],
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def _role_endpoint(role: str) -> tuple[str, str, str]:
    prefix = f"SPECAGENT_{role.upper()}_"
    base_url = os.environ.get(f"{prefix}BASE_URL")
    api_key = os.environ.get(f"{prefix}API_KEY")
    model = os.environ.get(f"{prefix}MODEL")
    missing = [
        name
        for name, value in (
            ("BASE_URL", base_url),
            ("API_KEY", api_key),
            ("MODEL", model),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing endpoint env for role {role}: "
            + ", ".join(f"{prefix}{name}" for name in missing)
        )
    return str(base_url), str(api_key), str(model)


def _out_dir(args: argparse.Namespace) -> Path:
    if args.mode == "full":
        if not args.run_name:
            raise SystemExit("--mode full requires --run-name")
        return DUMPS_ROOT / args.run_name
    if not args.out_dir:
        raise SystemExit("--mode smoke requires --out-dir")
    return Path(args.out_dir)


def _freeze_manifest(
    path: Path,
    *,
    run_name: str | None,
    model: str,
    base_url: str,
    domains: list[str],
    dataset: Any,
    rollouts_per_example: int,
    shard_size: int,
    max_turns: int,
    toolset: str,
    protocol: str = "solo",
    strong_model: str | None = None,
    strong_base_url: str | None = None,
    rp_fused_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the pre-run manifest; refuse to overwrite a differing one."""
    import specagent_adapter as adapter  # noqa: E402

    tasks = []
    for i, row in enumerate(dataset):
        info = row.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        tasks.append(
            {
                "index": i,
                "example_id": row.get("example_id", i),
                "task_id": adapter.task_id_from_state(row),
                "domain": info.get("domain"),
                "n_rollouts": rollouts_per_example,
            }
        )
    shards = [
        [t["index"] for t in tasks[i : i + shard_size]]
        for i in range(0, len(tasks), shard_size)
    ]
    manifest = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "base_url": base_url,
        "domains": domains,
        "dataset_size": len(tasks),
        "rollouts_per_example": rollouts_per_example,
        "planned_trajectories": len(tasks) * rollouts_per_example,
        "step_cap": max_turns,
        "toolset": toolset,
        "protocol": protocol,
        "strong_model": strong_model,
        "strong_base_url": strong_base_url,
        "rp_fused_profile": rp_fused_profile,
        "shard_size": shard_size,
        "tasks": tasks,
        "shards": shards,
        "shard_status": {str(i): "pending" for i in range(len(shards))},
        # trajectory_ids are assigned by verifiers at runtime (uuid4); the
        # *design* (task set, k, order) is what the manifest freezes. Actual
        # ids are recorded into shard checkpoints and run_summary.
    }
    if path.exists():
        existing = json.loads(path.read_text())
        same = all(
            existing.get(k) == manifest[k]
            for k in (
                "tasks",
                "rollouts_per_example",
                "domains",
                "model",
                "protocol",
                "strong_model",
                "rp_fused_profile",
            )
        )
        if not same:
            raise SystemExit(
                f"Existing manifest at {path} differs from this plan; "
                "refusing to silently mutate a frozen design. Use a new "
                "--run-name or delete the run dir intentionally."
            )
        manifest = existing
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


async def _run(args: argparse.Namespace) -> int:
    import specagent_adapter as adapter
    from solo_client import SpecAgentChatClient

    from specagent.endpoint_config import resolve_role_runtime
    from specagent.http_auth import authorization_header
    from specagent.protocol.accounting import InferencePhase, span_from_record
    from specagent.protocol.types import ModelRole

    bench_root = _bench_root()
    from automationbench.domains import get_combined_dataset  # noqa: E402
    from automationbench.export import export_results  # noqa: E402
    from automationbench.pricing import PricingDatabase  # noqa: E402
    from automationbench.rubric import create_rubric  # noqa: E402
    from automationbench.runner import AutomationBenchEnv  # noqa: E402
    from automationbench.usage import calculate_run_usage  # noqa: E402
    from verifiers.types import ClientConfig  # noqa: E402

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    dataset = get_combined_dataset(domains)

    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        indices = [
            i
            for i, row in enumerate(dataset)
            if adapter.task_id_from_state(row) in wanted
        ]
        missing = wanted - {adapter.task_id_from_state(dataset[i]) for i in indices}
        if missing:
            raise SystemExit(f"Tasks not found in {domains}: {sorted(missing)}")
        dataset = dataset.select(indices)
    if args.num_examples > 0:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))
    if len(dataset) == 0:
        raise SystemExit("Empty task selection")
    if args.mode == "smoke" and len(dataset) > MAX_SMOKE_TASKS:
        raise SystemExit(
            f"smoke mode allows at most {MAX_SMOKE_TASKS} tasks; selection has {len(dataset)}. "
            "Use --mode full --run-name <name> for full runs."
        )

    if args.protocol == "rp_fused":
        # Both roles run; weak is the actor and nominal endpoint.
        base_url, api_key, model = _role_endpoint("weak")
        runtime = resolve_role_runtime("weak")
    else:
        base_url, api_key, model = _role_endpoint(args.role)
        runtime = resolve_role_runtime(args.role)
    authorization = (
        authorization_header(api_key, scheme=runtime.auth_scheme)
        if runtime.auth_scheme != "bearer"
        else None
    )
    strong_model: str | None = None
    strong_base_url: str | None = None
    rp_backend = None
    rp_gateway_config = None
    rp_fused_profile = None
    if args.protocol == "rp_fused":
        import rp_backend as rpb
        import rp_fused_protocol as rfp

        strong_base_url, strong_api_key, strong_model = _role_endpoint("strong")
        strong_runtime = resolve_role_runtime("strong")

        def _role_ep(
            ep_base: str, ep_key: str, ep_runtime: Any
        ) -> rpb.RoleEndpoint:
            return rpb.RoleEndpoint(
                base_url=ep_base,
                api_key=ep_key,
                authorization=(
                    authorization_header(ep_key, scheme=ep_runtime.auth_scheme)
                    if ep_runtime.auth_scheme != "bearer"
                    else None
                ),
                extra_body=(
                    dict(ep_runtime.extra_body) if ep_runtime.extra_body else None
                ),
            )

        rp_backend = rpb.RecordingRoutedBackend(
            weak_model=model,
            weak=_role_ep(base_url, api_key, runtime),
            strong_model=strong_model,
            strong=_role_ep(strong_base_url, strong_api_key, strong_runtime),
        )
        rp_fused_profile = dict(rfp.MANIFEST_PROFILE)
        rp_gateway_config = rfp.rp_fused_gateway_config(
            weak_model=model,
            strong_model=strong_model,
            weak_extra_body=(
                dict(runtime.extra_body) if runtime.extra_body else None
            ),
            strong_extra_body=(
                dict(strong_runtime.extra_body)
                if strong_runtime.extra_body
                else None
            ),
            verifier_extra_body=(
                dict(strong_runtime.verifier_extra_body)
                if strong_runtime.verifier_extra_body
                else None
            ),
        )

    out_dir = _out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shards").mkdir(exist_ok=True)

    manifest = _freeze_manifest(
        out_dir / "manifest.json",
        run_name=args.run_name,
        model=model,
        base_url=base_url,
        domains=domains,
        dataset=dataset,
        rollouts_per_example=args.rollouts_per_example,
        shard_size=args.shard_size,
        max_turns=args.max_turns,
        toolset=args.toolset,
        protocol=args.protocol,
        strong_model=strong_model,
        strong_base_url=strong_base_url,
        rp_fused_profile=rp_fused_profile,
    )

    recorder = adapter.AttemptRecorder(
        role=ModelRole(args.role), phase=InferencePhase.ACT, model=model
    )

    os.environ["SPECAGENT_AB_API_KEY"] = api_key
    config = ClientConfig(
        api_key_var="SPECAGENT_AB_API_KEY",
        api_base_url=base_url,
        # SDK-level retries are off: every transport attempt must be visible
        # to the recorder. Retrying client's own backoff still applies.
        max_retries=0,
    )
    if args.protocol == "rp_fused":
        from rp_client import SpecAgentGatewayClient

        client = SpecAgentGatewayClient(
            config,
            recorder=recorder,
            backend=rp_backend,
            gateway_config=rp_gateway_config,
        )
    else:
        client = SpecAgentChatClient(
            config, recorder=recorder, authorization=authorization
        )

    sampling_args: dict[str, Any] = {}
    if args.protocol == "solo" and runtime.extra_body:
        sampling_args["extra_body"] = dict(runtime.extra_body)

    started = time.time()
    all_outputs: list[dict[str, Any]] = []
    all_flags: list[str] = []
    crashed_shards: list[str] = []

    for shard_i, shard_task_indices in enumerate(manifest["shards"]):
        shard_file = out_dir / "shards" / f"shard_{shard_i:04d}.json"
        status = manifest["shard_status"].get(str(shard_i), "pending")
        if args.resume and status == "done" and shard_file.exists():
            prior = json.loads(shard_file.read_text())
            all_outputs.extend(prior.get("outputs", []))
            all_flags.extend(prior.get("run_flags", []))
            continue
        if not args.resume and shard_file.exists() and status == "done":
            raise SystemExit(
                f"Shard {shard_i} already done; pass --resume to continue "
                "or use a new --run-name."
            )

        shard_ds = dataset.select(shard_task_indices)
        env = AutomationBenchEnv(
            dataset=shard_ds,
            rubric=create_rubric(),
            max_turns=args.max_turns,
            toolset=args.toolset,
        )
        shard_started = time.time()
        span_start = len(recorder.ledger.spans)
        rank_start = len(recorder.rank_rows)
        anom_start = len(recorder.anomalies)
        dec_start = (
            len(client.decisions) if hasattr(client, "decisions") else 0
        )
        crash: str | None = None
        try:
            results = await env.evaluate(
                client=client,
                model=model,
                sampling_args=sampling_args,
                num_examples=len(shard_ds),
                rollouts_per_example=args.rollouts_per_example,
                max_concurrent=args.max_concurrent,
                state_columns=STATE_COLUMNS,
            )
        except Exception as exc:  # noqa: BLE001
            # verifiers only absorbs vf.Error inside a rollout; other
            # exceptions crash evaluate. The ledger still holds every attempt
            # span — flag the shard, dump what exists, and continue to the
            # next shard instead of silently dropping the run.
            crash = f"{type(exc).__name__}: {exc}"
            results = {"outputs": [], "metadata": {}}
            crashed_shards.append(f"shard_{shard_i:04d}:{crash.split(':', 1)[0]}")
        shard_wall = time.time() - shard_started

        raw_outputs = results["outputs"]
        outputs = [
            _j(
                {
                    "task": adapter.task_id_from_state(ro),
                    "reward": ro.get("reward", 0.0),
                    "info": ro.get("info", {}),
                    "prompt": ro.get("prompt"),
                    "completion": ro.get("completion"),
                    "trajectory_id": ro.get("trajectory_id"),
                    "error": ro.get("error"),
                    "stop_condition": ro.get("stop_condition"),
                    "is_completed": ro.get("is_completed"),
                    "is_truncated": ro.get("is_truncated"),
                    "steps": len(ro.get("trajectory") or ()),
                    "metrics": ro.get("metrics"),
                    "timing": ro.get("timing"),
                    "_usage": ro.get("_usage"),
                    "_assertion_results": ro.get("_assertion_results"),
                    "_perf": ro.get("_perf"),
                }
            )
            for ro in raw_outputs
        ]
        shard_summary = adapter.build_run_summary(
            raw_outputs=raw_outputs,
            export_tasks=[],
            recorder=recorder,
            model=model,
            role=args.role,
            benchmark_makespan_seconds=shard_wall,
            step_cap=args.max_turns,
        )
        payload = {
            "shard_index": shard_i,
            "task_indices": shard_task_indices,
            "wall_seconds": shard_wall,
            "outputs": outputs,
            "run_flags": shard_summary["run_flags"],
            # This shard's accounting rows are embedded so a --resume run
            # can rebuild run-level artifacts without re-serving anything.
            "spans": [
                s.to_record() for s in recorder.ledger.spans[span_start:]
            ],
            "rank_rows": recorder.rank_rows[rank_start:],
            "anomalies": recorder.anomalies[anom_start:],
            "rp_decisions": (
                client.decisions[dec_start:]
                if hasattr(client, "decisions") and args.protocol == "rp_fused"
                else []
            ),
        }
        if crash:
            payload["evaluate_crashed"] = crash
        shard_file.write_text(json.dumps(payload, indent=2, default=str))
        manifest["shard_status"][str(shard_i)] = "crashed" if crash else "done"
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        all_outputs.extend(outputs)
        all_flags.extend(shard_summary["run_flags"])
        print(
            f"shard {shard_i}/{len(manifest['shards'])} done "
            f"({len(raw_outputs)} rollouts, {shard_wall:.1f}s)"
            + (f" CRASH {crash}" if crash else ""),
            flush=True,
        )

    makespan = time.time() - started

    export = {"tasks": []}
    raw_for_summary = all_outputs
    if all_outputs:
        usage = calculate_run_usage(
            {"outputs": all_outputs, "metadata": {}}, model, PricingDatabase()
        )
        export = export_results(
            all_outputs,
            usage,
            model,
            domains,
            out_dir / "export.json",
            duration_seconds=makespan,
            toolset=args.toolset,
            step_cap=args.max_turns,
        )

    # Rebuild run-level accounting from the shard checkpoints, not the live
    # recorder: under --resume the recorder only holds this session's spans,
    # while shard files carry every completed shard's rows. A single-session
    # run and a resumed run must produce identical artifacts.
    replay = adapter.AttemptRecorder(
        role=ModelRole(args.role), phase=InferencePhase.ACT, model=model
    )
    for shard_i in range(len(manifest["shards"])):
        sf = out_dir / "shards" / f"shard_{shard_i:04d}.json"
        if not sf.exists():
            continue
        prior = json.loads(sf.read_text())
        for record in prior.get("spans", []):
            replay.ledger.add(span_from_record(record))
        replay.rank_rows.extend(prior.get("rank_rows", []))
        replay.anomalies.extend(prior.get("anomalies", []))

    with (out_dir / "specagent_spans.jsonl").open("w") as fh:
        for span in replay.ledger.spans:
            fh.write(json.dumps(span.to_record(), default=str) + "\n")

    with (out_dir / "rank_map.jsonl").open("w") as fh:
        for row in replay.rank_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    if args.protocol == "rp_fused":
        with (out_dir / "rp_decisions.jsonl").open("w") as fh:
            for shard_i in range(len(manifest["shards"])):
                sf = out_dir / "shards" / f"shard_{shard_i:04d}.json"
                if not sf.exists():
                    continue
                prior = json.loads(sf.read_text())
                for row in prior.get("rp_decisions", []):
                    fh.write(json.dumps(row, default=str) + "\n")

    resources = adapter.paper_payload(replay.ledger, model=model)
    (out_dir / "specagent_resources.json").write_text(
        json.dumps(resources, indent=2, default=str)
    )

    summary = adapter.build_run_summary(
        raw_outputs=raw_for_summary,
        export_tasks=export.get("tasks", []),
        recorder=replay,
        model=model,
        role=args.role,
        benchmark_makespan_seconds=makespan,
        step_cap=args.max_turns,
    )
    summary["benchmark_root"] = str(bench_root)
    summary["mode"] = args.mode
    summary["run_name"] = args.run_name
    summary["protocol"] = args.protocol
    if args.protocol == "rp_fused":
        summary["strong_model"] = strong_model
        summary["strong_base_url"] = strong_base_url
        summary["rp_fused_profile"] = rp_fused_profile
    summary["rollouts_per_example"] = args.rollouts_per_example
    summary["crashed_shards"] = crashed_shards
    summary["gpu_seconds_note"] = (
        "exclusive_gpu_seconds stays UNKNOWN: a shared data-parallel rank cannot "
        "attribute exclusive compute. duration_seconds + rank_map.jsonl are "
        "the inputs for a labeled duration x 1 GPU proxy, which includes queueing."
    )
    if crashed_shards:
        summary["run_flags"].append(f"crashed_shards:{len(crashed_shards)}")
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(json.dumps({k: summary[k] for k in ("n_tasks", "run_flags")}, indent=2))
    for task in summary["tasks"]:
        print(
            f"  {task['task_id']}: passed={task['passed']} "
            f"attempts={task['n_transport_attempts']} "
            f"flags={task['validation']['flags'] + task['export_validation']['flags']}"
        )
    print(f"Artifacts in {out_dir}")
    return 1 if summary["run_flags"] or summary["anomalies"] else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
