"""Resolve per-role Strong/Weak runtime options from env without leaking across roles.

A Strong endpoint's settings must not flip Weak auth, and one model family's thinking switch
(e.g. `chat_template_kwargs.thinking`) must not reach an endpoint that expects another.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from specagent.http_auth import parse_auth_scheme

QWEN_THINKING_KWARG = "enable_thinking"
DEEPSEEK_THINKING_KWARG = "thinking"

QWEN_STRONG_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": True},
    "separate_reasoning": True,
}
QWEN_VERIFIER_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "separate_reasoning": True,
}
DEEPSEEK_STRONG_EXTRA_BODY = {
    "chat_template_kwargs": {"thinking": True},
    "separate_reasoning": True,
}
DEEPSEEK_VERIFIER_EXTRA_BODY = {
    "chat_template_kwargs": {"thinking": False},
    "separate_reasoning": True,
    "enable_thinking": False,
}
WEAK_DEFAULT_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": True}}


def _json_env(environ: Mapping[str, str], key: str) -> dict[str, Any] | None:
    raw = environ.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _thinking_kwarg(environ: Mapping[str, str], prefix: str) -> str | None:
    raw = environ.get(f"{prefix}THINKING_KWARG")
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


@dataclass(frozen=True)
class RoleRuntimeConfig:
    role: str
    auth_scheme: str
    thinking_kwarg: str | None
    extra_body: dict[str, Any] | None
    verifier_extra_body: dict[str, Any] | None = None

    def thinking_keys(self) -> set[str]:
        body = self.extra_body or {}
        keys = set((body.get("chat_template_kwargs") or {}).keys())
        if self.thinking_kwarg:
            keys.add(self.thinking_kwarg)
        return keys


def resolve_role_runtime(
    role: str, environ: Mapping[str, str] | None = None
) -> RoleRuntimeConfig:
    import os

    if role not in {"weak", "strong"}:
        raise ValueError(f"unknown role {role!r}")
    env = os.environ if environ is None else environ
    prefix = f"SPECAGENT_{role.upper()}_"
    auth_scheme = parse_auth_scheme(env.get(f"{prefix}AUTH_SCHEME"), default="bearer")
    thinking_kwarg = _thinking_kwarg(env, prefix)
    extra_key = "STRONG_EXTRA_BODY" if role == "strong" else "WEAK_EXTRA_BODY"
    extra = _json_env(env, f"{prefix}EXTRA_BODY") or _json_env(env, extra_key)
    verifier = None
    if role == "strong":
        verifier = _json_env(env, "SPECAGENT_STRONG_VERIFIER_EXTRA_BODY") or _json_env(
            env, "VERIFIER_EXTRA_BODY"
        )
        if extra is None:
            if thinking_kwarg == QWEN_THINKING_KWARG:
                extra = dict(QWEN_STRONG_EXTRA_BODY)
            elif thinking_kwarg == DEEPSEEK_THINKING_KWARG:
                extra = dict(DEEPSEEK_STRONG_EXTRA_BODY)
        if verifier is None:
            if thinking_kwarg == QWEN_THINKING_KWARG:
                verifier = dict(QWEN_VERIFIER_EXTRA_BODY)
            elif thinking_kwarg == DEEPSEEK_THINKING_KWARG:
                verifier = dict(DEEPSEEK_VERIFIER_EXTRA_BODY)
    elif extra is None:
        extra = dict(WEAK_DEFAULT_EXTRA_BODY)
    return RoleRuntimeConfig(
        role=role,
        auth_scheme=auth_scheme,
        thinking_kwarg=thinking_kwarg,
        extra_body=extra,
        verifier_extra_body=verifier,
    )
