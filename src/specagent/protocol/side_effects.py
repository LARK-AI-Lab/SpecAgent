"""Classify whether an agent action mutates state and must be gated.

The protocol treats three kinds of action as irrevocable: a tool call that changes durable
state, a message sent to the user, and a claim that the task is done. Each ends a speculative
rollout as a pending step that only runs after Strong accepts it.

tau2 tags its tools with ``mutates_state``; adapters for it should classify from that marker.
Most MCP tools carry no such marker, so this module uses explicit allow/deny lists plus
conservative name heuristics. Unknown tools are treated as mutating: better
to wait for Strong than to pollute an environment we cannot reset. Read heuristics must never
match a mutation; a false "read" executes without verification, while a false "write" only costs
one extra verification.

When the caller knows which tools the agent actually has (``known_tools``), a name outside that
set is never gated: the call cannot execute, the runtime returns an error, and paying a Strong
verification for a hallucinated tool name buys nothing. Real read tools whose names miss the
``verb_`` marker form (``web_fetch``, ``memory_get``, ``*__resources_list``) are covered by the
exact list and the suffix markers below.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

# Action dicts without a tool name that still mutate state (a reply to the user is irrevocable).
MUTATING_ACTION_TYPES = frozenset({"message", "user_message", "reply", "claim_done"})

# Local / meta tools that end the task or change durable workspace/MCP state, plus MCP tools
# whose names would otherwise look like reads.
MUTATING_EXACT = frozenset(
    {
        "claim_done",
        "python_execute",
        "write_file",
        "edit_file",
        "create_file",
        "create_directory",
        "move_file",
        "search_and_replace",
        # Checks a passenger in. Not a lookup.
        "check_in",
        # AutomationBench Zapier meta-tool: always writes into WorldState.
        "execute_tool",
    }
)

# Confirmed read-only lookups. Prefer this list over heuristics for names that contain both a
# read verb and a write-looking substring. Entries are matched against the full tool name and
# against the normalized suffix after the MCP ``server__`` prefix.
READONLY_EXACT = frozenset(
    {
        "list_allowed_directories",
        "list_directory",
        "list_events",
        "get_workbook_metadata",
        "get_document_text",
        "read_data_from_excel",
        "read_file",
        "read",
        "web_search",
        "memory_search",
        "session_status",
        "maps__directions",
        "maps__geocode",
        "maps__distance_matrix",
        "directions",
        "geocode",
        "reverse_geocode",
        "distance_matrix",
        "notion__API-post-search",
        "post_search",
        "post_database_query",
        "check_connection",
        "check_entry_requirements",
        "canvas_health_check",
        "canvas_get_user_profile",
        "canvas_get_course_grades",
        "canvas_get_submission",
        "canvas_get_assignment",
        "canvas_get_course",
        "canvas_get_account_reports",
        "canvas_list_announcements",
        "canvas_list_assignments",
        "canvas_list_courses",
        "canvas_list_quizzes",
        "canvas_list_assignment_groups",
        "canvas_list_account_users",
        "woo_coupons_list",
        "woo_products_list",
        "woo_products_reviews_list",
        "woo_products_categories_list",
        "manage_context",
        "sleep",
        # Document retrieval tools.
        "search",
        "get_document",
        # AutomationBench API toolset: schema search and a pure encoding helper.
        "api_search",
        "search_tools",
        "base64_encode",
        # Local tools whose names carry no read verb prefix.
        "web_fetch",
        "memory_get",
        "dir_list",
        # MCP protocol-level listings (``<server>__resources_list`` / ``prompts_list``).
        "resources_list",
        "prompts_list",
        # MCP reads named without the ``verb_`` separator.
        "listaccounts",
        "weather__check",
        # MCP-Atlas keyless servers: lookups whose names carry no read verb. Chosen from each
        # server's documented behaviour, not from run outcomes; the pinned servers name tools
        # ``<server>_<tool>`` with a single underscore, so the full name is listed.
        "calculator_calculate",
        "cli-mcp-server_show_security_rules",
        "clinicaltrialsgov-mcp-server_clinicaltrials_analyze_trends",
        "context7_resolve-library-id",
        "filesystem_directory_tree",
        "git_git_diff",
        "git_git_diff_staged",
        "git_git_diff_unstaged",
        "git_git_log",
        "git_git_show",
        "mcp-code-executor_check_installed_packages",
        "memory_open_nodes",
        "osm-mcp-server_analyze_commute",
        "osm-mcp-server_analyze_neighborhood",
        "osm-mcp-server_explore_area",
        "osm-mcp-server_geocode_address",
        "osm-mcp-server_reverse_geocode",
        "osm-mcp-server_suggest_meeting_point",
        "whois_whois_as",
        "whois_whois_domain",
        "whois_whois_ip",
        "whois_whois_tld",
        "wikipedia_extract_key_facts",
        "wikipedia_summarize_article_for_query",
        "wikipedia_summarize_article_section",
    }
)

# HTTP methods that do not change AutomationBench WorldState when used with api_fetch.
_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD"})

_WRITE_MARKERS = (
    "create_",
    "write_",
    "update_",
    "send_",
    "add_",
    "delete_",
    "remove_",
    "insert_",
    "upload_",
    "rename_",
    "move_",
    "format_",
    "append_",
    "post_",
    "schedule_",
    "share_",
    "edit_",
    "replace",
    # ``mark_read`` / ``mark_emails`` change notification and mailbox state.
    "mark_",
)

# Only verbs that are read-only in every API we have seen. ``check_`` is deliberately absent:
# ``check_in`` is a mutation.
_READ_MARKERS = (
    "list_",
    "get_",
    "search_",
    "read_",
    "find_",
    "lookup_",
    "fetch_",
    "describe_",
    "retrieve",
    "track_",
    "estimate_",
    "health",
    "metadata",
)

# Read verbs in suffix position (``memory_get``, ``dir_list``, ``prompts_list``). Checked after
# ``_WRITE_MARKERS`` so ``send_list``-style names still gate. ``_read`` is deliberately absent:
# ``mark_read`` is a mutation.
_READ_SUFFIXES = (
    "_list",
    "_get",
    "_search",
    "_status",
    "_info",
)


def action_name(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("tool") or action.get("name") or "")
    if isinstance(action, str):
        return action
    return str(action or "")


def _normalized_suffix(name: str) -> str:
    """Strip an MCP ``server__`` prefix and an ``API-`` prefix; unify separators."""
    suffix = name.rsplit("__", 1)[-1]
    if suffix.startswith("API-"):
        suffix = suffix[4:]
    return suffix.replace("-", "_")


def _http_method(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("method") or "").upper()


def action_mutates_state(
    action: Any,
    *,
    unknown_mutates: bool = True,
    known_tools: Collection[str] | None = None,
    readonly_tools: Collection[str] = (),
) -> bool:
    """Return True if the action should be gated (not executed until Strong accepts).

    ``known_tools`` is the set of tool names the agent can actually call (e.g. the ``tools`` of
    the current chat request). When given, a tool name outside it is not gated: it cannot
    execute, so releasing it only yields a runtime error the drafting model can recover from.
    """
    if isinstance(action, dict) and not action.get("tool") and not action.get("name"):
        if str(action.get("type") or "") in MUTATING_ACTION_TYPES:
            return True
    name = action_name(action)
    if not name:
        return unknown_mutates
    if known_tools is not None and name not in known_tools:
        return False
    if name in readonly_tools:
        return False
    candidates = (name, _normalized_suffix(name))
    if any(candidate in MUTATING_EXACT for candidate in candidates):
        return True
    if any(candidate in READONLY_EXACT for candidate in candidates):
        return False
    # api_fetch looks like a read ("fetch_") but POST/PUT/PATCH/DELETE mutate WorldState.
    if _normalized_suffix(name).lower() == "api_fetch":
        method = _http_method(action)
        if not method:
            return True
        return method not in _SAFE_HTTP_METHODS
    lowered = _normalized_suffix(name).lower()
    if any(marker in lowered for marker in _WRITE_MARKERS):
        return True
    if any(marker in lowered for marker in _READ_MARKERS):
        return False
    if lowered.endswith(_READ_SUFFIXES):
        return False
    return unknown_mutates

