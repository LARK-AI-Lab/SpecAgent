"""Per-call Authorization scheme for OpenAI-compatible backends.

Scheme is chosen by the caller (Weak vs Strong), never by a process-global env
read inside these helpers. Unknown values raise ValueError.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VALID_AUTH_SCHEMES = ("bearer", "raw")


def parse_auth_scheme(value: str | None = None, *, default: str = "bearer") -> str:
    if default not in VALID_AUTH_SCHEMES:
        raise ValueError(
            f"unknown auth scheme default {default!r}; expected 'bearer' or 'raw'"
        )
    if value is None or str(value).strip() == "":
        return default
    scheme = str(value).strip().lower()
    if scheme not in VALID_AUTH_SCHEMES:
        raise ValueError(
            f"unknown auth scheme {value!r}; expected 'bearer' or 'raw'"
        )
    return scheme


def authorization_header(api_key: str, *, scheme: str = "bearer") -> str:
    parsed = parse_auth_scheme(scheme)
    key = api_key or ""
    if parsed == "raw":
        if key.lower().startswith("bearer "):
            return key.split(" ", 1)[1].strip()
        return key
    if key.lower().startswith("bearer "):
        return key
    return f"Bearer {key}"


def openai_init_kwargs(
    base_url: str,
    api_key: str,
    timeout: float,
    *,
    scheme: str = "bearer",
) -> dict[str, Any]:
    """kwargs for openai.AsyncOpenAI / OpenAI. Does not read process env."""
    parsed = parse_auth_scheme(scheme)
    if parsed == "raw":
        token = authorization_header(api_key, scheme="raw")
        return {
            "base_url": base_url,
            "api_key": "unused",
            "timeout": timeout,
            "default_headers": {"Authorization": token},
        }
    return {"base_url": base_url, "api_key": api_key, "timeout": timeout}


def strong_auth_scheme_from_env(environ: Mapping[str, str] | None = None) -> str:
    import os

    env = os.environ if environ is None else environ
    return parse_auth_scheme(env.get("SPECAGENT_STRONG_AUTH_SCHEME"), default="bearer")


def weak_auth_scheme_from_env(environ: Mapping[str, str] | None = None) -> str:
    import os

    env = os.environ if environ is None else environ
    return parse_auth_scheme(env.get("SPECAGENT_WEAK_AUTH_SCHEME"), default="bearer")
