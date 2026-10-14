#!/usr/bin/env python3
"""Re-evaluate deferred VitaBench result files in parallel across judge endpoints.

Sim runs launched with ``run_eval.py --defer-eval`` store ``reward=-1.0``
placeholders (see apply_session_patch.py: VITA_DEFER_EVAL). This driver fills
them in via vita's own ``--re-evaluate-file`` path:

  - one re-eval process per results file (each spawns its own
    ThreadPoolExecutor of --concurrency judge calls)
  - files are assigned round-robin across --judge-endpoints so judge
    replicas share the load
  - output lands next to the input as ``<stem>.eval.json`` under
    data/simulations (vita's own save path), plus a copied summary row

Usage:
  python reeval.py --results out/specagent_RUN/simulations.json \
      --judge-endpoints http://judge-a/v1,http://judge-b/v1
  python reeval.py --results specagent_specagent_ --from-sim-dir --judge-endpoints ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_BENCH_ROOT = os.environ.get("VITABENCH_DIR")
DOMAINS = ("delivery", "instore", "ota", "cross_domain")


def _domain_of(path: Path) -> str:
    m = re.search(r"_(delivery|instore|ota|cross_domain)_s\d+\.json$", path.name)
    if m:
        return m.group(1)
    try:
        info = json.loads(path.read_text(encoding="utf-8")).get("info") or {}
        dom = (info.get("environment_info") or {}).get("domain_name")
        if dom in DOMAINS:
            return dom
    except Exception:
        pass
    raise SystemExit(f"cannot infer domain from {path.name}")


def _pending(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for sim in data.get("simulations") or []:
        r = ((sim.get("reward_info") or {}).get("reward"))
        if isinstance(r, (int, float)) and r < 0:
            return True
    return False


def _judge_yaml(path: Path, base_url: str, model: str, api_key: str) -> Path:
    entry = {
        "name": "judge",
        "base_url": base_url.rstrip("/") + "/chat/completions",
        "model": model,
        "temperature": 0.0,
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    }
    path.write_text(
        yaml.safe_dump({"default": {"temperature": 0.0}, "models": [entry]},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True,
                    help="result json paths, or name-prefixes with --from-sim-dir")
    ap.add_argument("--from-sim-dir", action="store_true",
                    help="resolve --results entries as name prefixes inside "
                         "the bench's data/simulations directory")
    ap.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCH_ROOT,
                    required=DEFAULT_BENCH_ROOT is None,
                    help="VitaBench checkout ($VITABENCH_DIR when set)")
    ap.add_argument("--judge-endpoints", required=True,
                    help="comma-separated /v1 base URLs, rotated across files")
    ap.add_argument("--judge-model", default=os.environ.get("SPECAGENT_JUDGE_MODEL",
                                                          "deepseek-v4-flash-0731"))
    ap.add_argument("--judge-api-key", default=os.environ.get("SPECAGENT_JUDGE_API_KEY", "none"))
    ap.add_argument("--concurrency", type=int, default=4,
                    help="judge threads inside each re-eval process")
    ap.add_argument("--language", default="english")
    args = ap.parse_args()

    bench = args.benchmark_root
    sim_dir = bench / "data" / "simulations"
    files: list[Path] = []
    for item in args.results:
        p = Path(item)
        if args.from_sim_dir:
            files.extend(sorted(sim_dir.glob(f"{item}*.json")))
        elif p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)
    files = [f for f in dict.fromkeys(files) if f.is_file() and _pending(f)]
    if not files:
        print("nothing pending re-evaluation")
        return 0

    endpoints = [e.strip() for e in args.judge_endpoints.split(",") if e.strip()]
    if not endpoints:
        raise SystemExit("no judge endpoints")
    procs = []
    for i, f in enumerate(files):
        url = endpoints[i % len(endpoints)]
        job_yaml = _judge_yaml(sim_dir / f".{f.stem}.models.yaml", url,
                               args.judge_model, args.judge_api_key)
        out_name = f"{f.stem}.eval.json"
        cmd = [
            sys.executable, "-m", "vita.cli", "run",
            "--domain", _domain_of(f),
            "--agent", "llm_agent", "--agent-llm", "judge",
            "--user", "user_simulator", "--user-llm", "judge",
            "--evaluator-llm", "judge",
            "--language", args.language,
            "--max-concurrency", str(args.concurrency),
            "--re-evaluate-file", str(f),
            "--save-to", out_name,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{bench / 'src'}:{env.get('PYTHONPATH', '')}"
        env["VITA_MODEL_CONFIG_PATH"] = str(job_yaml)
        log = open(f.with_suffix(".reeval.log"), "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=bench, env=env,
                                       stdout=log, stderr=subprocess.STDOUT),
                      f, out_name, url, log))
        print(f"[{i}] {f.name} -> {url} ({out_name})")

    rc = 0
    for proc, f, out_name, _url, log in procs:
        proc.wait()
        log.close()
        out = sim_dir / out_name
        ok = proc.returncode == 0 and out.is_file()
        print(f"{'OK ' if ok else 'FAIL'} {f.name} (rc={proc.returncode})")
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
