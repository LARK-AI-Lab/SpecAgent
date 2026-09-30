"""CLI entrypoint for tau2 baseline / SpecAgent evaluation runs.

Thin wrapper around tau2.run.run_domain: registers our custom agent
factories (baseline_agent / agent in this directory) and builds a
TextRunConfig, then lets tau2's own runner handle task iteration,
concurrency, and results.json writing -- see tau2.runner.batch.run_domain.

Modes:
  reflective_prefix  SpecAgent (default): Weak drafts, Strong verifies chunks
                     that end in a state-changing call and repairs rejections.
  baseline_weak      Weak model alone.
  baseline_strong    Strong model alone.

MODEL SERVING IS OUT OF SCOPE FOR THIS SCRIPT. Weak/Strong actor base_url,
model name, and API key are read from SPECAGENT_WEAK_*/SPECAGENT_STRONG_*
env vars (or the matching --weak-*/--strong-* CLI flags), which whoever
stands up the model servers must set. This script will refuse to start with
a clear error if they are missing for the mode requested, rather than
guessing a port.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
# baseline_agent.py / agent.py live alongside this script and import each
# other and `openai_client` as bare module names rather than dotted package
# paths, since benchmarks/ has no __init__.py anywhere.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from specagent.http_auth import parse_auth_scheme  # noqa: E402


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _judge_config() -> dict | None:
    """External model for the two roles that must NOT be the model under test:
    the user simulator and the nl-assertion evaluator. Using an agent-family
    model for either risks inflating scores, so both are pointed at a
    third-party endpoint via SPECAGENT_JUDGE_*.
    """
    base_url = os.environ.get("SPECAGENT_JUDGE_BASE_URL")
    model = os.environ.get("SPECAGENT_JUDGE_MODEL")
    if not base_url or not model:
        return None
    cfg = {
        "model": model if "/" in model else f"openai/{model}",
        "api_base": base_url,
        "api_key": os.environ.get("SPECAGENT_JUDGE_API_KEY", "EMPTY"),
    }
    # A self-hosted reasoning model may think by default and spend the whole
    # completion budget on it, which as a user simulator is a blank turn every
    # turn. Hosted endpoints have no such kwarg, so this is opt-in.
    if os.environ.get("SPECAGENT_JUDGE_DISABLE_THINKING", "").lower() in ("1", "true", "yes", "on"):
        cfg["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return cfg


def _user_config(judge: dict | None) -> dict | None:
    """The user simulator's own endpoint. Falls back to the judge's in full, so
    setting only SPECAGENT_JUDGE_* works; when both roles are self-hosted they
    can live on separate replicas via SPECAGENT_USER_*.
    """
    if judge is None:
        return None
    model = os.environ.get("SPECAGENT_USER_MODEL") or judge["model"]
    cfg = dict(judge)
    cfg["model"] = model if "/" in model else f"openai/{model}"
    cfg["api_base"] = os.environ.get("SPECAGENT_USER_BASE_URL") or judge["api_base"]
    cfg["api_key"] = os.environ.get("SPECAGENT_USER_API_KEY") or judge["api_key"]
    return cfg


# Greedy decoding can make a user simulator lock onto one restatement of its
# goal and repeat it verbatim until the step cap, and every simulation that ends
# in max_steps scores 0, so the simulator is sampled. The judge stays greedy:
# grading should not vary.
_JUDGE_TEMPERATURE = _env_float("SPECAGENT_JUDGE_TEMPERATURE", 0.0)
_USER_TEMPERATURE = _env_float("SPECAGENT_USER_TEMPERATURE", 0.7)


def _llm_args(cfg: dict, *, temperature: float) -> dict:
    """litellm kwargs for one of the two non-agent roles."""
    args = {
        "temperature": temperature,
        "api_base": cfg["api_base"],
        "api_key": cfg["api_key"],
    }
    if cfg.get("extra_body"):
        args["extra_body"] = cfg["extra_body"]
    return args


# Must run BEFORE tau2.run is imported: the nl-assertion evaluator does a
# `from tau2.config import DEFAULT_LLM_NL_ASSERTIONS` (binding the value at
# its own import time), so tau2.config has to be corrected first. main() also
# re-patches the evaluator module directly, which covers CLI overrides and any
# import ordering we don't control.
_JUDGE = _judge_config()
if _JUDGE is not None:
    import tau2.config as _tau2_config

    _tau2_config.DEFAULT_LLM_NL_ASSERTIONS = _JUDGE["model"]
    _tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = _llm_args(_JUDGE, temperature=_JUDGE_TEMPERATURE)

from agent import create_specagent_tau2_agent  # noqa: E402
from baseline_agent import create_baseline_tau2_agent  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.run import run_domain  # noqa: E402

WEAK_AGENT_NAME = "specagent_weak"
STRONG_AGENT_NAME = "specagent_strong"
SPECAGENT_AGENT_NAME = "specagent"


def _register_agents() -> None:
    for name in (WEAK_AGENT_NAME, STRONG_AGENT_NAME):
        if registry.get_agent_factory(name) is None:
            registry.register_agent_factory(create_baseline_tau2_agent, name)
    if registry.get_agent_factory(SPECAGENT_AGENT_NAME) is None:
        registry.register_agent_factory(create_specagent_tau2_agent, SPECAGENT_AGENT_NAME)


def _role_config(role: str, args: argparse.Namespace) -> dict:
    prefix = role.upper()
    base_url = getattr(args, f"{role}_base_url") or os.environ.get(f"SPECAGENT_{prefix}_BASE_URL")
    model = getattr(args, f"{role}_model") or os.environ.get(f"SPECAGENT_{prefix}_MODEL")
    if not base_url or not model:
        raise SystemExit(
            f"Missing {role} serving config: set --{role}-base-url/--{role}-model or "
            f"SPECAGENT_{prefix}_BASE_URL/SPECAGENT_{prefix}_MODEL env vars. "
            f"(Model serving is handled outside this script.)"
        )
    api_key = (
        getattr(args, f"{role}_api_key") or os.environ.get(f"SPECAGENT_{prefix}_API_KEY", "EMPTY")
    )
    thinking_kwarg = (
        getattr(args, f"{role}_thinking_kwarg")
        or os.environ.get(f"SPECAGENT_{prefix}_THINKING_KWARG")
    )
    auth_scheme = parse_auth_scheme(
        getattr(args, f"{role}_auth_scheme", None)
        or os.environ.get(f"SPECAGENT_{prefix}_AUTH_SCHEME"),
        default="bearer",
    )
    thinking = getattr(args, f"{role}_thinking")
    if thinking is None:
        env_thinking = os.environ.get(f"SPECAGENT_{prefix}_THINKING")
        thinking = (
            env_thinking.lower() in ("1", "true", "yes") if env_thinking is not None else None
        )
    max_output_tokens = (
        getattr(args, f"{role}_max_output_tokens")
        or _int_env(f"SPECAGENT_{prefix}_MAX_OUTPUT_TOKENS")
    )
    return {
        "role": role,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "thinking_kwarg": thinking_kwarg,
        "thinking": thinking,
        "max_output_tokens": max_output_tokens,
        "auth_scheme": auth_scheme,
    }


def _int_env(name: str) -> int | None:
    val = os.environ.get(name)
    return int(val) if val else None


def _add_role_args(parser: argparse.ArgumentParser, role: str) -> None:
    parser.add_argument(f"--{role}-base-url")
    parser.add_argument(f"--{role}-model")
    parser.add_argument(f"--{role}-api-key")
    parser.add_argument(f"--{role}-thinking-kwarg")
    parser.add_argument(f"--{role}-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(f"--{role}-max-output-tokens", type=int)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["airline", "retail", "telecom"])
    parser.add_argument(
        "--mode",
        default="reflective_prefix",
        choices=["reflective_prefix", "baseline_weak", "baseline_strong"],
        help="SpecAgent (reflective_prefix), or one model alone",
    )
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--user", default="user_simulator")
    parser.add_argument(
        "--llm-user", default=None,
        help="tau2 user-simulator model (litellm-routed; unrelated to weak/strong serving)",
    )
    parser.add_argument(
        "--horizon", type=int, default=6,
        help="reflective_prefix: maximum Weak draft steps per chunk (K)",
    )
    parser.add_argument(
        "--verifier-max-output-tokens", type=int, default=None,
        help="reflective_prefix: output cap for the verifier call "
             "(defaults to --strong-max-output-tokens)",
    )
    parser.add_argument(
        "--baseline-stats", default=None,
        help="path for baseline token/time records (else $SPECAGENT_BASELINE_STATS)",
    )
    parser.add_argument(
        "--specagent-stats", default=None,
        help="path for reflective_prefix token/time records (else $SPECAGENT_STATS)",
    )
    parser.add_argument(
        "--allow-default-judge", action="store_true",
        help="permit tau2's built-in judge/user-sim instead of SPECAGENT_JUDGE_* "
             "(not for reported results)",
    )
    _add_role_args(parser, "weak")
    _add_role_args(parser, "strong")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.baseline_stats:
        os.environ["SPECAGENT_BASELINE_STATS"] = args.baseline_stats
        from specagent.evals.trial_shard_io import configure_from_stats_env
        configure_from_stats_env(args.baseline_stats)
    if args.specagent_stats:
        os.environ["SPECAGENT_STATS"] = args.specagent_stats
        from specagent.evals.trial_shard_io import configure_from_stats_env
        configure_from_stats_env(args.specagent_stats)

    _register_agents()

    llm_args_agent: dict
    if args.mode == "baseline_weak":
        agent_name = WEAK_AGENT_NAME
        llm_args_agent = _role_config("weak", args)
    elif args.mode == "baseline_strong":
        agent_name = STRONG_AGENT_NAME
        llm_args_agent = _role_config("strong", args)
    else:
        agent_name = SPECAGENT_AGENT_NAME
        strong_config = _role_config("strong", args)
        if args.verifier_max_output_tokens:
            strong_config["verifier_max_output_tokens"] = args.verifier_max_output_tokens
        llm_args_agent = {
            "domain": args.domain,
            "horizon": args.horizon,
            "weak": _role_config("weak", args),
            "strong": strong_config,
        }

    judge = _judge_config()
    if judge is not None:
        # Re-patch at the reference site: evaluate_nl_assertions() resolves
        # these from its own module globals at call time, so this takes effect
        # regardless of import ordering.
        import tau2.evaluator.evaluator_nl_assertions as nl_eval

        nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
        nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = _llm_args(judge, temperature=_JUDGE_TEMPERATURE)
        user = _user_config(judge)
        print(
            f"[judge] nl_assertions -> {judge['model']} @ {judge['api_base']} "
            f"T={_JUDGE_TEMPERATURE}; "
            f"user simulator -> {user['model']} @ {user['api_base']} T={_USER_TEMPERATURE}"
        )
    elif not args.allow_default_judge:
        raise SystemExit(
            "No external judge configured. The user simulator and nl-assertion "
            "evaluator would fall back to tau2's default model via litellm, "
            "which needs OPENAI_API_KEY and risks same-family judge bias. Set "
            "SPECAGENT_JUDGE_BASE_URL/_MODEL/_API_KEY, or pass --allow-default-judge."
        )

    config_kwargs = dict(
        domain=args.domain,
        agent=agent_name,
        llm_agent=llm_args_agent.get("model", "specagent"),
        llm_args_agent=llm_args_agent,
        user=args.user,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        save_to=args.save_to,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
    )
    if judge is not None:
        user = _user_config(judge)
        config_kwargs["llm_user"] = args.llm_user or user["model"]
        config_kwargs["llm_args_user"] = _llm_args(user, temperature=_USER_TEMPERATURE)
    elif args.llm_user:
        config_kwargs["llm_user"] = args.llm_user

    config = TextRunConfig(**config_kwargs)
    results = run_domain(config)
    print(f"Done: {len(results.simulations)} simulation(s) -> {args.save_to}")


if __name__ == "__main__":
    main()
