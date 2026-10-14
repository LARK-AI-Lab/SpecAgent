#!/usr/bin/env python3
"""Write a VitaBench models.yaml that routes only the agent through SpecAgent.

VitaBench resolves endpoints per model name from ``VITA_MODEL_CONFIG_PATH`` and
``generate()`` POSTs straight to each entry's ``base_url`` (full
/v1/chat/completions path, not just /v1). Three roles exist:

  agent     -> the SpecAgent gateway (mode=specagent) or a direct model endpoint
               (mode=weak / mode=strong baselines, no protocol in the loop)
  user      -> direct endpoint; the user simulator must NOT go through the
               gateway (it would share the agent's session)
  evaluator -> direct endpoint; same reasoning

The generated file carries no secrets: API keys come from the environment at
runtime via SPECAGENT_*_API_KEY -- write them into this file only through
``--*-api-key`` flags on run_eval.py, never hardcoded.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _entry(
    base_url: str,
    api_key: str,
    temperature: float,
    auth_scheme: str = "bearer",
    extra: dict | None = None,
) -> dict:
    authorization = api_key if auth_scheme == "raw" else f"Bearer {api_key}"
    entry = {
        "base_url": base_url.rstrip("/"),
        "temperature": temperature,
        "headers": {
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
    }
    # Extra fields (e.g. chat_template_kwargs for thinking control) ride the
    # entry into the request body via data.update(models[model]).
    entry.update(extra or {})
    return entry


def build_models_yaml(
    *,
    agent_name: str,
    agent_base_url: str,
    agent_api_key: str = "none",
    agent_temperature: float = 0.0,
    agent_auth_scheme: str = "bearer",
    user_name: str,
    user_base_url: str,
    user_api_key: str = "none",
    user_temperature: float = 0.7,
    user_auth_scheme: str = "bearer",
    evaluator_name: str,
    evaluator_base_url: str,
    evaluator_api_key: str = "none",
    evaluator_temperature: float = 0.0,
    evaluator_auth_scheme: str = "bearer",
    agent_extra: dict | None = None,
    user_extra: dict | None = None,
    evaluator_extra: dict | None = None,
) -> dict:
    """Build the models.yaml dict. ``base_url`` values must already include the
    full /v1/chat/completions path -- vita's generate() posts to them verbatim.

    Returns (config, entry_names): entry_names maps role -> the ``name`` vita
    must resolve for that role. Vita indexes entries by name, so when two roles
    share a served model (e.g. user simulator and judge on one model) their entries
    would collide -- colliding names are suffixed ``::<role>`` and the entry
    pins an inner ``model:`` field back to the served name, which
    ``data.update(models[model])`` then copies over the request's model field."""
    roles = [
        ("agent", agent_name, _entry(
            agent_base_url, agent_api_key, agent_temperature, agent_auth_scheme, agent_extra)),
        ("user", user_name, _entry(
            user_base_url, user_api_key, user_temperature, user_auth_scheme, user_extra)),
        ("evaluator", evaluator_name, _entry(
            evaluator_base_url, evaluator_api_key, evaluator_temperature, evaluator_auth_scheme,
            evaluator_extra)),
    ]
    seen: dict[str, int] = {}
    entry_names: dict[str, str] = {}
    models = []
    for role, served_name, entry in roles:
        if served_name in seen:
            entry_name = f"{served_name}::{role}"
            entry = {**entry, "model": served_name}
        else:
            entry_name = served_name
        seen[served_name] = seen.get(served_name, 0) + 1
        entry_names[role] = entry_name
        models.append({"name": entry_name, **entry})
    config = {
        "default": {
            "temperature": 0.0,
            "headers": {"Content-Type": "application/json"},
        },
        "models": models,
    }
    return config, entry_names


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-name", default="specagent")
    parser.add_argument("--agent-base-url", required=True)
    parser.add_argument("--agent-api-key", default="none")
    parser.add_argument("--agent-auth-scheme", default="bearer", choices=("bearer", "raw"))
    parser.add_argument("--user-name", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--user-api-key", default="none")
    parser.add_argument("--user-auth-scheme", default="bearer", choices=("bearer", "raw"))
    parser.add_argument("--evaluator-name", required=True)
    parser.add_argument("--evaluator-base-url", required=True)
    parser.add_argument("--evaluator-api-key", default="none")
    parser.add_argument("--evaluator-auth-scheme", default="bearer", choices=("bearer", "raw"))
    args = parser.parse_args()

    config, _entry_names = build_models_yaml(
        agent_name=args.agent_name,
        agent_base_url=args.agent_base_url,
        agent_api_key=args.agent_api_key,
        agent_auth_scheme=args.agent_auth_scheme,
        user_name=args.user_name,
        user_base_url=args.user_base_url,
        user_api_key=args.user_api_key,
        user_auth_scheme=args.user_auth_scheme,
        evaluator_name=args.evaluator_name,
        evaluator_base_url=args.evaluator_base_url,
        evaluator_api_key=args.evaluator_api_key,
        evaluator_auth_scheme=args.evaluator_auth_scheme,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    args.output.chmod(0o600)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
