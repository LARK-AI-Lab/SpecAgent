#!/usr/bin/env python3
"""Run VitaBench with SpecAgent, or with the Weak or Strong model alone.

VitaBench's agent posts plain OpenAI chat completions to the ``base_url`` of its models.yaml
entry, so SpecAgent runs as the gateway server (``python -m specagent.server --setting
vitabench``) and only the agent entry points at it; the user simulator and the judge get direct
endpoints. ``apply_session_patch.py`` stamps an ``x-specagent-session-id`` header per task
attempt so concurrent tasks keep separate gateway sessions. The gateway classifies tools by
VitaBench's own ``@is_tool`` labels, written to ``tool_classes.json`` in the run directory.

Endpoints come from flags or the environment (flag > SPECAGENT_{ROLE}_*):

  SPECAGENT_WEAK_BASE_URL, _MODEL, _API_KEY, _AUTH_SCHEME    modes weak and specagent
  SPECAGENT_STRONG_BASE_URL, ...                             modes strong and specagent
  SPECAGENT_USER_* / SPECAGENT_EVALUATOR_*                   user simulator / judge; both fall
                                                             back to SPECAGENT_JUDGE_*

Defaults are the paper setting: the Delivery domain (all 100 tasks), one trial per task, at most
100 steps, eight tasks in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from apply_session_patch import apply as apply_session_patch  # noqa: E402
from configure_models import build_models_yaml  # noqa: E402
from tool_classes import mutates_map, report, scan  # noqa: E402

# Request-body fields of the direct baselines, matching the paper's Weak-only and Strong-only
# runs: thinking on for both; Weak with reasoning_effort high and the 8,192-token output cap that
# SpecAgent's Weak drafts also have.
BASELINE_AGENT_EXTRA: dict[str, dict] = {
    "weak": {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "high",
        "max_tokens": 8192,
    },
    "strong": {"chat_template_kwargs": {"enable_thinking": True}, "separate_reasoning": True},
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _completions_url(base_url: str) -> str:
    """vita posts to base_url verbatim: normalize to a full chat-completions URL."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _api_base(base_url: str) -> str:
    """The /v1 base the gateway server appends /chat/completions to."""
    return _completions_url(base_url)[: -len("/chat/completions")]


def _role(role: str, args: argparse.Namespace, *, fallbacks: tuple[str, ...] = ()) -> dict:
    """Endpoint config for one role: flag > SPECAGENT_{ROLE}_* > fallback roles."""

    def pick(kind: str) -> str | None:
        flag = getattr(args, f"{role}_{kind}", None)
        if flag:
            return flag
        for name in (role, *fallbacks):
            value = _env(f"SPECAGENT_{name.upper()}_{kind.upper()}")
            if value:
                return value
        return None

    base_url, model = pick("base_url"), pick("model")
    if not base_url or not model:
        raise SystemExit(
            f"missing {role} endpoint: set --{role}-base-url/--{role}-model or "
            f"SPECAGENT_{role.upper()}_BASE_URL/SPECAGENT_{role.upper()}_MODEL"
        )
    return {
        "base_url": base_url,
        "model": model,
        "api_key": pick("api_key") or "none",
        "auth_scheme": pick("auth_scheme") or "bearer",
    }


def _extra_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit("--*-extra-json must be a JSON object")
    return value


def gateway_command(
    args: argparse.Namespace, run_dir: Path, weak: dict, strong: dict, class_file: Path
) -> list[str]:
    """The release gateway server in the paper's VitaBench setting."""
    return [
        sys.executable, "-m", "specagent.server",
        "--setting", "vitabench",
        "--weak-base-url", _api_base(weak["base_url"]),
        "--weak-model", weak["model"],
        "--weak-auth-scheme", weak["auth_scheme"],
        "--strong-base-url", _api_base(strong["base_url"]),
        "--strong-model", strong["model"],
        "--strong-auth-scheme", strong["auth_scheme"],
        "--tool-class-file", str(class_file),
        "--host", "127.0.0.1",
        "--port", str(args.gateway_port),
        "--telemetry", str(run_dir / "gateway.jsonl"),
        "--log-file", str(run_dir / "gateway.log"),
        *args.gateway_arg,
    ]


def _launch_gateway(args: argparse.Namespace, run_dir: Path, bench: Path):
    """Start the gateway server; returns (base URL, process)."""
    weak, strong = _role("weak", args), _role("strong", args)
    class_file = run_dir / "tool_classes.json"
    class_file.write_text(
        json.dumps(mutates_map(bench), indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"tool classification: {report(scan(bench))} -> {class_file}")
    env = dict(os.environ)
    # Keys travel in the environment, not on the command line.
    env["SPECAGENT_WEAK_API_KEY"] = weak["api_key"]
    env["SPECAGENT_STRONG_API_KEY"] = strong["api_key"]
    log = open(run_dir / "gateway.out", "w", encoding="utf-8")
    proc = subprocess.Popen(
        gateway_command(args, run_dir, weak, strong, class_file),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{args.gateway_port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"gateway exited early (rc={proc.returncode}); see {log.name}")
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=2) as response:
                if response.status == 200:
                    return url, proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit(f"gateway did not become healthy; see {log.name}")


def _summarize(results_path: Path, output_dir: Path) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    sims = data.get("simulations") or []
    rows = []
    rewards = []
    terminations: dict[str, int] = {}
    for sim in sims:
        reward = (sim.get("reward_info") or {}).get("reward")
        term = sim.get("termination_reason")
        terminations[term] = terminations.get(term, 0) + 1
        if isinstance(reward, (int, float)) and reward >= 0:
            rewards.append(reward)
        rows.append(
            {
                "task_id": sim.get("task_id"),
                "trial": sim.get("trial"),
                "reward": reward,
                "termination": term,
                "steps": len(sim.get("messages") or []),
                "duration_s": round(sim.get("duration") or 0.0, 1),
                "agent_cost": sim.get("agent_cost"),
                "user_cost": sim.get("user_cost"),
            }
        )
    summary = {
        "n": len(sims),
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        "terminations": terminations,
        "tasks": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bench_default = _env("VITABENCH_DIR")
    parser.add_argument(
        "--benchmark-root", type=Path, default=bench_default, required=bench_default is None,
        help="VitaBench checkout ($VITABENCH_DIR when set)",
    )
    parser.add_argument("--mode", required=True, choices=("specagent", "weak", "strong"))
    parser.add_argument("--domain", default="delivery",
                        help="vita --domain value; a comma-joined list runs cross-domain tasks")
    parser.add_argument("--task-set-name", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="Task ids to run (default: every task of the domain)")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--language", default="english", choices=("english", "chinese"))
    parser.add_argument("--evaluation-type", default="trajectory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--gateway-port", type=int, default=int(_env("GATEWAY_PORT", "18084")))
    parser.add_argument(
        "--gateway-arg", action="append", default=[], metavar="ARG",
        help="Extra argument for the gateway server, e.g. "
        "--gateway-arg=--verifier-extra-body='{\"enable_thinking\":false}' (repeatable)",
    )
    parser.add_argument("--enable-think", action="store_true", help="vita's --enable-think")
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep an existing save file and let vita resume it (answers vita's prompts)",
    )
    parser.add_argument(
        "--async-eval", action=argparse.BooleanOptionalAction, default=True,
        help="Judge finished tasks in a background pool (VITA_ASYNC_EVAL) so judge latency "
        "stays out of the simulation's wall time; rewards still land in the results file",
    )
    parser.add_argument(
        "--defer-eval", action="store_true",
        help="Skip judging (VITA_DEFER_EVAL); fill the -1 placeholders later with reeval.py",
    )
    for role in ("weak", "strong", "user", "evaluator", "judge"):
        parser.add_argument(f"--{role}-base-url")
        parser.add_argument(f"--{role}-model")
        parser.add_argument(f"--{role}-api-key")
        parser.add_argument(f"--{role}-auth-scheme", choices=("bearer", "raw"))
    for role in ("agent", "user", "evaluator"):
        parser.add_argument(
            f"--{role}-extra-json", default=None,
            help=f"JSON object merged into the {role} models.yaml entry, so into its requests"
            + (" (default for weak/strong: BASELINE_AGENT_EXTRA)" if role == "agent" else ""),
        )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    bench = args.benchmark_root.expanduser().resolve()
    if not (bench / "src" / "vita" / "run.py").is_file():
        raise SystemExit(f"not a VitaBench checkout: {bench}")

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / f"{args.mode}_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # One agent instance == one task attempt == one gateway session.
    apply_session_patch(bench)

    gateway = None
    if args.mode == "specagent":
        url, gateway = _launch_gateway(args, output_dir, bench)
        agent = {"base_url": url, "model": "specagent", "api_key": "none",
                 "auth_scheme": "bearer"}
        agent_extra = _extra_json(args.agent_extra_json)
    else:
        agent = _role(args.mode, args)
        agent_extra = (
            _extra_json(args.agent_extra_json)
            if args.agent_extra_json
            else BASELINE_AGENT_EXTRA[args.mode]
        )
    user = _role("user", args, fallbacks=("judge",))
    evaluator = _role("evaluator", args, fallbacks=("judge",))

    try:
        models_cfg, entry_names = build_models_yaml(
            agent_name=agent["model"],
            agent_base_url=_completions_url(agent["base_url"]),
            agent_api_key=agent["api_key"],
            agent_auth_scheme=agent["auth_scheme"],
            user_name=user["model"],
            user_base_url=_completions_url(user["base_url"]),
            user_api_key=user["api_key"],
            user_auth_scheme=user["auth_scheme"],
            evaluator_name=evaluator["model"],
            evaluator_base_url=_completions_url(evaluator["base_url"]),
            evaluator_api_key=evaluator["api_key"],
            evaluator_auth_scheme=evaluator["auth_scheme"],
            agent_extra=agent_extra,
            user_extra=_extra_json(args.user_extra_json),
            evaluator_extra=_extra_json(args.evaluator_extra_json),
        )
        import yaml

        models_yaml = output_dir / "models.yaml"
        models_yaml.write_text(
            yaml.safe_dump(models_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        models_yaml.chmod(0o600)
        return _run_vita(args, bench, output_dir, run_tag, models_yaml, entry_names)
    finally:
        if gateway is not None:
            gateway.terminate()


def vita_command(args: argparse.Namespace, entry_names: dict, save_name: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "vita.cli", "run",
        "--domain", "delivery,instore,ota" if args.domain == "cross_domain" else args.domain,
        "--agent", "llm_agent",
        "--agent-llm", entry_names["agent"],
        "--user", "user_simulator",
        "--user-llm", entry_names["user"],
        "--evaluator-llm", entry_names["evaluator"],
        "--language", args.language,
        "--evaluation-type", args.evaluation_type,
        "--num-trials", str(args.num_trials),
        "--max-steps", str(args.max_steps),
        "--max-errors", str(args.max_errors),
        "--max-concurrency", str(args.max_concurrency),
        "--seed", str(args.seed),
        "--save-to", save_name,
    ]
    if args.task_set_name:
        cmd += ["--task-set-name", args.task_set_name]
    if args.task_ids:
        cmd += ["--task-ids", *args.task_ids]
    if args.num_tasks:
        cmd += ["--num-tasks", str(args.num_tasks)]
    if args.enable_think:
        cmd += ["--enable-think"]
    return cmd


def _run_vita(args, bench: Path, output_dir: Path, run_tag: str, models_yaml: Path,
              entry_names: dict) -> int:
    save_name = f"specagent_{args.mode}_{run_tag}.json"
    # vita prompts interactively when save_to exists; remove this driver's own stale file unless
    # --resume asks vita to continue it.
    stale = bench / "data" / "simulations" / save_name
    if stale.is_file() and not args.resume:
        print(f"removing stale results file {stale}")
        stale.unlink()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(bench / "src"), env.get("PYTHONPATH", "")) if part
    )
    env["VITA_MODEL_CONFIG_PATH"] = str(models_yaml)
    if args.defer_eval:
        env["VITA_DEFER_EVAL"] = "1"
    elif args.async_eval:
        env["VITA_ASYNC_EVAL"] = "1"

    with open(output_dir / "vita_run.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            vita_command(args, entry_names, save_name), cwd=bench, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            stdin=subprocess.PIPE if args.resume else None,
        )
        if args.resume:
            # Answer vita's resume and config-change prompts.
            try:
                proc.stdin.write("y\n" * 2000)
                proc.stdin.close()
            except OSError:
                pass
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()

    results_path = bench / "data" / "simulations" / save_name
    if not results_path.is_file():
        print(f"WARNING: no results file at {results_path} (vita rc={rc})")
        return rc or 1
    summary = _summarize(results_path, output_dir)
    (output_dir / "simulations.json").write_text(
        results_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(
        f"\n=== {args.mode} {args.domain} n={summary['n']} "
        f"reward_mean={summary['reward_mean']} terminations={summary['terminations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
