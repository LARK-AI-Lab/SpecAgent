#!/usr/bin/env python3
"""Apply the SpecAgent patch to a VitaBench checkout (upstream commit 742e240).

Five edits, all anchored and idempotent; a checkout that drifted from the anchors fails closed:

1. ``vita/utils/llm_utils.py`` -- add a ``SPECAGENT_SESSION_ID`` ContextVar and,
   inside ``generate()``, stamp it as the ``x-specagent-session-id`` request
   header. The SpecAgent gateway keys per-task speculative state on that header;
   without it the gateway falls back to hashing the system prompt, which is the
   domain policy + env time -- not guaranteed unique across tasks (retries and
   same-time tasks would share one session and contaminate each other).

2. ``vita/agent/llm_agent.py`` -- ``LLMAgent.__init__``/``LLMSoloAgent.__init__``
   set the ContextVar to a fresh ``vita-<uuid4>``. Each task attempt builds a new
   agent in its worker thread (``_run_task_internal``), and ContextVar state is
   per-thread, so ``--max-concurrency`` tasks never share a session and a retried
   task gets a fresh one alongside its fresh environment.

3. ``vita/cli.py`` -- the three ``models.get(args.*_llm, {})`` lookups feed the
   whole yaml entry into ``generate(**llm_args)``. configure_models.py emits a
   ``model:`` field inside deduped same-name entries (e.g. a shared user/judge
   model) so ``data.update(models[model])`` rewrites the request's model field
   back to the served name -- but as a **kwarg** it collides with generate()'s
   ``model`` parameter. Strip the pin from the kwargs copies.

4. ``vita/evaluator/evaluator.py`` -- ``VITA_DEFER_EVAL`` env gate: when set,
   ``evaluate_simulation()`` returns a placeholder RewardInfo immediately so
   the judge stays out of the simulation worker's critical path. Rewards are
   filled in afterwards via ``vita.cli run --re-evaluate-file``, which runs
   judge calls in a ThreadPoolExecutor decoupled from sim wall-time.

5. ``vita/run.py`` -- ``VITA_ASYNC_EVAL`` env gate: when set, ``run_task``
   attaches a ``reward=-1`` placeholder and submits the real
   ``evaluate_simulation`` to a background ThreadPoolExecutor
   (``VITA_EVAL_WORKERS``, default 8). Judge latency overlaps the next task's
   simulation instead of serialising into shard wall-time; a drain at the end
   of ``run_tasks`` waits for stragglers and rewrites their ``reward_info``
   in the checkpoint file. If the process dies before the drain, placeholders
   stay -1 and ``reeval.py`` fills them the same way as defer mode.

The header is also sent (harmlessly) by the user simulator and evaluator when
they share the worker thread -- their requests go to their own endpoints, which
ignore it. Only the agent model's ``base_url`` should point at the gateway.

Usage: python apply_session_patch.py [VITA_ROOT]
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "SPECAGENT_SESSION_ID"

LLM_UTILS_ANCHOR = "from vita.config import ("
LLM_UTILS_BLOCK = '''
# SpecAgent integration: per-agent-instance session identity. LLMAgent sets one
# uuid per construction (one per task attempt, including retries); generate()
# stamps it as the x-specagent-session-id header so the SpecAgent gateway keys
# speculative state per task instead of hashing the shared system prompt.
# ContextVar state is per-thread, so --max-concurrency tasks never share one.
SPECAGENT_SESSION_ID: contextvars.ContextVar = contextvars.ContextVar(
    "vita_specagent_session_id", default=None
)

'''

GENERATE_ANCHOR = '            headers = models[model]["headers"]'
GENERATE_BLOCK = '''
            _specagent_session = SPECAGENT_SESSION_ID.get()
            if _specagent_session is not None:
                headers = {**headers, "x-specagent-session-id": _specagent_session}
'''

TIMEOUT_ANCHOR = "timeout=(10, 600)"
TIMEOUT_NEW = "timeout=(10, float(os.environ.get(\"VITA_LLM_TIMEOUT_S\", \"600\")))"

AGENT_ANCHOR = '        self.enable_think = enable_think'
AGENT_BLOCK = '''
        # SpecAgent: one agent instance == one task attempt == one gateway
        # session. A retry rebuilds the agent, hence a fresh id.
        SPECAGENT_SESSION_ID.set(f"vita-{uuid.uuid4().hex}")
'''

IMPORTS_LLM_UTILS = "import contextvars\n"
IMPORTS_LLM_UTILS_ANCHOR = "import json\nimport re\nimport time\n"
AGENT_IMPORT_ANCHOR = "from vita.utils.llm_utils import generate"
AGENT_IMPORT_NEW = "from vita.utils.llm_utils import SPECAGENT_SESSION_ID, generate"
AGENT_UUID_ANCHOR = "from copy import deepcopy"


def _patch_llm_utils(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    changed = False
    if MARKER not in src:
        if IMPORTS_LLM_UTILS_ANCHOR not in src:
            raise SystemExit(f"{path}: import anchor not found -- file drifted?")
        if GENERATE_ANCHOR not in src:
            raise SystemExit(f"{path}: generate() headers anchor not found -- file drifted?")
        src = src.replace(IMPORTS_LLM_UTILS_ANCHOR, IMPORTS_LLM_UTILS + IMPORTS_LLM_UTILS_ANCHOR, 1)
        src = src.replace(LLM_UTILS_ANCHOR, LLM_UTILS_BLOCK + LLM_UTILS_ANCHOR, 1)
        src = src.replace(GENERATE_ANCHOR, GENERATE_ANCHOR + GENERATE_BLOCK, 1)
        changed = True
    else:
        print(f"{path}: session-id patch already applied")
    if TIMEOUT_ANCHOR in src:
        src = src.replace(TIMEOUT_ANCHOR, TIMEOUT_NEW, 1)
        if "\nimport os\n" not in src:
            src = src.replace("import contextvars\n", "import contextvars\nimport os\n", 1)
        changed = True
        print(f"{path}: applied VITA_LLM_TIMEOUT_S env gate")
    elif TIMEOUT_NEW not in src:
        raise SystemExit(f"{path}: generate() timeout anchor not found -- file drifted?")
    if changed:
        path.write_text(src, encoding="utf-8")
    return changed


def _patch_llm_agent(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{path}: session-id patch already applied, no-op")
        return False
    if AGENT_UUID_ANCHOR not in src or AGENT_IMPORT_ANCHOR not in src:
        raise SystemExit(f"{path}: import anchors not found -- file drifted?")
    count = src.count(AGENT_ANCHOR)
    if count != 2:
        raise SystemExit(
            f"{path}: expected 2 agent __init__ anchors (LLMAgent + LLMSoloAgent), found {count}"
        )
    src = src.replace(AGENT_UUID_ANCHOR, "import uuid\n" + AGENT_UUID_ANCHOR, 1)
    src = src.replace(AGENT_IMPORT_ANCHOR, AGENT_IMPORT_NEW, 1)
    src = src.replace(AGENT_ANCHOR, AGENT_ANCHOR + AGENT_BLOCK)
    path.write_text(src, encoding="utf-8")
    print(f"{path}: applied session-id patch (2 agent classes)")
    return True


CLI_MARKER = 'if k != "model"'
CLI_ANCHOR = "models.get(args.{role}_llm, {{}})"
CLI_ROLES = ("agent", "user", "evaluator")


def _patch_cli(path: Path) -> bool:
    """Keep the ``model:`` pin inside deduped yaml entries out of **llm_args.

    cli.py passes ``models.get(args.*_llm, {})`` verbatim as generate() kwargs;
    an entry carrying ``model:`` (same served model under two role names) would
    raise ``generate() got multiple values for keyword argument 'model'``.
    """
    src = path.read_text(encoding="utf-8")
    if CLI_MARKER in src:
        print(f"{path}: llm-args model-pin strip already applied, no-op")
        return False
    for role in CLI_ROLES:
        anchor = CLI_ANCHOR.format(role=role)
        if src.count(anchor) != 1:
            raise SystemExit(
                f"{path}: expected 1 cli.py llm_args anchor for {role}, "
                f"found {src.count(anchor)} -- file drifted?"
            )
    for role in CLI_ROLES:
        anchor = CLI_ANCHOR.format(role=role)
        stripped = (
            f"{{k: v for k, v in models.get(args.{role}_llm, {{}}).items() "
            'if k != "model"}'
        )
        src = src.replace(anchor, stripped, 1)
    path.write_text(src, encoding="utf-8")
    print(f"{path}: applied llm-args model-pin strip (3 roles)")
    return True


EVAL_MARKER = "VITA_DEFER_EVAL"
EVAL_IMPORT_ANCHOR = "from vita.evaluator.evaluator_traj import TrajectoryEvaluator"
EVAL_ANCHOR = """    \"\"\"
    Evaluate the simulation based on the evaluation type.
    \"\"\""""
EVAL_BLOCK = '''
    if os.environ.get("VITA_DEFER_EVAL"):
        return RewardInfo(
            reward=-1.0,
            info={"note": "evaluation deferred (VITA_DEFER_EVAL); fill via --re-evaluate-file"},
        )
'''


def _patch_evaluator(path: Path) -> bool:
    """Keep judge calls out of the sim worker when VITA_DEFER_EVAL is set."""
    src = path.read_text(encoding="utf-8")
    if EVAL_MARKER in src:
        print(f"{path}: defer-eval patch already applied, no-op")
        return False
    if EVAL_IMPORT_ANCHOR not in src:
        raise SystemExit(f"{path}: import anchor not found -- file drifted?")
    if src.count(EVAL_ANCHOR) != 1:
        raise SystemExit(
            f"{path}: expected 1 evaluate_simulation docstring anchor, "
            f"found {src.count(EVAL_ANCHOR)} -- file drifted?"
        )
    if "import os\n" not in src:
        src = "import os\n" + src
    src = src.replace(EVAL_ANCHOR, EVAL_ANCHOR + EVAL_BLOCK, 1)
    path.write_text(src, encoding="utf-8")
    print(f"{path}: applied VITA_DEFER_EVAL gate")
    return True


ASYNC_MARKER = "VITA_ASYNC_EVAL"
RUN_IMPORT_ANCHOR = "from vita.evaluator.evaluator import evaluate_simulation"
RUN_MODULE_BLOCK = '''

# SpecAgent: optional background evaluation. With VITA_ASYNC_EVAL=1 the sim
# worker attaches a reward=-1 placeholder and submits the real
# evaluate_simulation to a pool, so judge latency overlaps the next task
# instead of serialising into shard wall-time. _spec_drain_evals waits for
# stragglers at the end of run_tasks and rewrites reward_info in the save file.
import os as _spec_os
import threading as _spec_threading

_SPEC_ASYNC_EVAL = _spec_os.environ.get("VITA_ASYNC_EVAL", "") not in ("", "0")
_SPEC_EVAL_POOL = None
_SPEC_EVAL_PENDING = []
_SPEC_EVAL_LOCK = _spec_threading.Lock()


def _spec_eval_pool():
    global _SPEC_EVAL_POOL
    if _SPEC_EVAL_POOL is None:
        _SPEC_EVAL_POOL = ThreadPoolExecutor(
            max_workers=int(_spec_os.environ.get("VITA_EVAL_WORKERS", "8")),
            thread_name_prefix="vita-eval",
        )
    return _SPEC_EVAL_POOL


def _spec_submit_eval(**kwargs):
    from vita.data_model.simulation import RewardInfo

    placeholder = RewardInfo(
        reward=-1.0, info={"note": "eval pending (VITA_ASYNC_EVAL)"}
    )
    fut = _spec_eval_pool().submit(evaluate_simulation, **kwargs)
    with _SPEC_EVAL_LOCK:
        _SPEC_EVAL_PENDING.append((kwargs["simulation"], fut))
    return placeholder


def _spec_drain_evals(save_to):
    from vita.data_model.simulation import RewardInfo

    with _SPEC_EVAL_LOCK:
        pending = list(_SPEC_EVAL_PENDING)
        _SPEC_EVAL_PENDING.clear()
    if not pending:
        return
    by_key = {}
    for sim, fut in pending:
        try:
            sim.reward_info = fut.result()
        except Exception as exc:
            sim.reward_info = RewardInfo(
                reward=-1.0, info={"note": f"async eval failed: {exc}"}
            )
        by_key[(sim.task_id, sim.trial)] = sim.reward_info
    if save_to is None:
        return
    try:
        path = Path(save_to)
        ckpt = json.loads(path.read_text(encoding="utf-8"))
        for entry in ckpt.get("simulations", []):
            key = (entry.get("task_id"), entry.get("trial"))
            if key in by_key:
                entry["reward_info"] = by_key[key].model_dump(mode="json")
        path.write_text(
            json.dumps(ckpt, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"async eval drained: {len(by_key)} rewards written to {path}")
    except Exception as exc:
        logger.warning(
            f"async eval drain failed ({exc}); run reeval.py to fill placeholders"
        )
'''

RUN_EVAL_ANCHOR = """    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        llm_evaluator=llm_evaluator,
        llm_args_evaluator=llm_args_evaluator,
        language=language,
    )
    simulation.reward_info = reward_info"""

RUN_EVAL_NEW = """    if _SPEC_ASYNC_EVAL:
        simulation.reward_info = _spec_submit_eval(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=evaluation_type,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
        reward_info = simulation.reward_info
    else:
        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=evaluation_type,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
        simulation.reward_info = reward_info"""

RUN_DRAIN_ANCHOR = "        res = list(executor.map(_run, *zip(*args)))"
RUN_DRAIN_NEW = RUN_DRAIN_ANCHOR + "\n        _spec_drain_evals(save_to)"


def _patch_run(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if ASYNC_MARKER in src:
        print(f"{path}: async-eval patch already applied, no-op")
        return False
    for anchor, name in (
        (RUN_IMPORT_ANCHOR, "evaluator import"),
        (RUN_EVAL_ANCHOR, "evaluate_simulation call site"),
        (RUN_DRAIN_ANCHOR, "executor.map drain point"),
    ):
        if src.count(anchor) != 1:
            raise SystemExit(
                f"{path}: expected 1 {name} anchor, found {src.count(anchor)} "
                "-- file drifted?"
            )
    src = src.replace(RUN_IMPORT_ANCHOR, RUN_IMPORT_ANCHOR + RUN_MODULE_BLOCK, 1)
    src = src.replace(RUN_EVAL_ANCHOR, RUN_EVAL_NEW, 1)
    src = src.replace(RUN_DRAIN_ANCHOR, RUN_DRAIN_NEW, 1)
    path.write_text(src, encoding="utf-8")
    print(f"{path}: applied VITA_ASYNC_EVAL background-judge patch")
    return True


def apply(vita_root: Path) -> bool:
    llm_utils = vita_root / "src" / "vita" / "utils" / "llm_utils.py"
    llm_agent = vita_root / "src" / "vita" / "agent" / "llm_agent.py"
    cli = vita_root / "src" / "vita" / "cli.py"
    evaluator = vita_root / "src" / "vita" / "evaluator" / "evaluator.py"
    run_py = vita_root / "src" / "vita" / "run.py"
    for path in (llm_utils, llm_agent, cli, evaluator, run_py):
        if not path.is_file():
            raise SystemExit(f"not a VitaBench checkout: {path} missing")
    changed = _patch_llm_utils(llm_utils)
    changed |= _patch_llm_agent(llm_agent)
    changed |= _patch_cli(cli)
    changed |= _patch_evaluator(evaluator)
    changed |= _patch_run(run_py)
    return changed


def main() -> int:
    root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()
    apply(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
