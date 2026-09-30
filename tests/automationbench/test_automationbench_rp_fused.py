"""Offline tests for the AutomationBench SpecAgent profile.

Pure: they run in the repository environment without verifiers/openai/httpx. The backend's
transport is an injected ``post_fn``, the gateway runs in-process, and verdicts are canned.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTEG = ROOT / "benchmarks" / "automationbench"
for _p in (str(ROOT / "src"), str(INTEG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decision_fields as df  # noqa: E402
import rp_backend as rpb  # noqa: E402
import rp_fused_protocol as rfp  # noqa: E402

from specagent.runtime.gateway import GatewayConfig, SpeculativeToolGateway  # noqa: E402

PAPER_CONFIGS = ROOT / "tests" / "golden" / "paper_configs.json"

WEAK_URL = "http://weak.test/v1"
STRONG_URL = "http://strong.test/v1"

READ_TOOL = "salesforce_find_records"
WRITE_TOOL = "gmail_send_email"

TOOLS = [
    {"type": "function", "function": {"name": READ_TOOL, "parameters": {}}},
    {"type": "function", "function": {"name": WRITE_TOOL, "parameters": {}}},
    {
        "type": "function",
        "function": {"name": "quickbooks_create_invoice", "parameters": {}},
    },
]

# AutomationBench pydantic schemas emit every Optional parameter as a required+nullable property
# (additionalProperties: false).
AB_SCHEMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": READ_TOOL,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": WRITE_TOOL,
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "body": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
]


def _completion(content, usage=None, rid="resp-1", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": rid,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage
        or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _verdict(accepted, reason=None, replacement=None):
    return json.dumps(
        {"reason": reason, "accepted_steps": accepted, "replacement_action": replacement}
    )


def _backend(post_fn):
    return rpb.RecordingRoutedBackend(
        weak_model="weak-model",
        weak=rpb.RoleEndpoint(base_url=WEAK_URL, api_key="k-weak"),
        strong_model="strong-model",
        strong=rpb.RoleEndpoint(base_url=STRONG_URL, api_key="k-strong"),
        post_fn=post_fn,
        retry_base_delay=0.0,
    )


def _gateway(backend):
    gw = SpeculativeToolGateway(
        backend, rfp.rp_fused_gateway_config("weak-model", "strong-model")
    )
    rpb.tag_gateway_phases(gw)
    return gw


def _request(tools=None):
    messages = [
        {"role": "system", "content": "You are a workflow automation agent."},
        {"role": "user", "content": "Create an invoice for Meridian Corp orders."},
    ]
    return {"model": "weak-model", "messages": messages, "tools": tools or TOOLS}


def _ids(response):
    return [c["id"] for c in (response["choices"][0]["message"].get("tool_calls") or [])]


def _weak_wrong_recipient():
    return rpb.PostResult(
        200,
        _completion(
            None,
            rid="w1",
            tool_calls=[_call("bad1", WRITE_TOOL, {"to": "wrong@x.com"})],
        ),
        {},
    )


# Profile shape ---------------------------------------------------------------------


def test_profile_matches_the_recorded_paper_setting():
    fields = json.loads(PAPER_CONFIGS.read_text())["automationbench"]
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()}
    assert rfp.rp_fused_gateway_config("weak", "strong") == GatewayConfig(
        weak_model="weak", strong_model="strong", **fields
    )


def test_profile_config_fields():
    cfg = rfp.rp_fused_gateway_config("weak-model", "strong-model")
    assert cfg.strict_action_validation is False
    assert cfg.fuse_correction_boundaries == ("mutation",)
    assert cfg.verifier_max_tokens == 4096
    assert cfg.horizon == 8
    assert cfg.verifier_history_reasoning is False
    assert cfg.correction_history_reasoning is False
    assert "replacement_action" in cfg.verifier_prompt_suffix
    assert "materially identical arguments" in cfg.verifier_prompt_suffix
    assert cfg.weak_review_hint is True
    assert cfg.repeat_call_guard is True
    # Role overlays default to the incoming request body.
    assert cfg.weak_extra_body is None
    assert cfg.strong_extra_body is None
    assert cfg.verifier_extra_body is None


def test_profile_passes_role_overlays_through():
    cfg = rfp.rp_fused_gateway_config(
        "w",
        "s",
        weak_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        strong_extra_body={"thinking": {"type": "enabled"}},
        verifier_extra_body={"thinking": {"type": "disabled"}},
    )
    assert cfg.weak_extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert cfg.strong_extra_body == {"thinking": {"type": "enabled"}}
    assert cfg.verifier_extra_body == {"thinking": {"type": "disabled"}}


def test_manifest_profile_json_roundtrip():
    payload = json.loads(json.dumps(rfp.MANIFEST_PROFILE))
    assert payload == {
        "output_keys": ["reason", "accepted_steps", "replacement_action"],
        "correction_policy": "rp_fused",
        "review_scope": "irrevocable",
        "horizon_role": "draft_batching_cap",
        "multi_call_execution": "validate_all_then_execute_native",
        "horizon": 8,
        "fuse_correction_boundaries": ["mutation"],
        "strict_action_validation": False,
        "verifier_max_tokens": 4096,
        "verifier_history_reasoning": False,
        "correction_history_reasoning": False,
        "verifier_prompt_suffix": rfp.PROFILE_CONFIG["verifier_prompt_suffix"],
        "weak_review_hint": True,
        "repeat_call_guard": True,
    }
    # Every configuration field the manifest pins is a real GatewayConfig field.
    assert set(rfp.PROFILE_CONFIG) <= set(GatewayConfig.__dataclass_fields__)


# Repair at a mutation boundary ----------------------------------------------------------


def test_replacement_commits_at_mutation_boundary():
    """A valid replacement commits directly: one Weak draft plus one Strong review, no separate
    repair turn."""
    strong_calls = []

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        strong_calls.append(body)
        return rpb.PostResult(
            200,
            _completion(
                _verdict(
                    0,
                    reason="Step 1: recipient wrong@x.com is not the account email.",
                    replacement={
                        "tool": WRITE_TOOL,
                        "arguments": {"to": "billing@meridian.com"},
                    },
                ),
                rid="v1",
            ),
            {},
        )

    backend = _backend(post)
    decision = _gateway(backend).handle(_request())

    assert [c["phase"] for c in backend.calls] == ["weak_act", "strong_verify"]
    assert decision.source == "strong_fused_correction"
    assert "bad1" not in _ids(decision.response)
    delivered = decision.response["choices"][0]["message"]["tool_calls"]
    assert delivered[0]["function"]["name"] == WRITE_TOOL
    assert json.loads(delivered[0]["function"]["arguments"]) == {
        "to": "billing@meridian.com"
    }
    # The profile's policy rode the review prompt; JSON output was requested.
    system_prompt = strong_calls[0]["messages"][0]["content"]
    assert "replacement_action" in system_prompt
    assert rfp.PROFILE_CONFIG["verifier_prompt_suffix"] in system_prompt
    assert strong_calls[0].get("response_format") == {"type": "json_object"}
    assert strong_calls[0]["max_tokens"] == 4096

    fields = df.decision_fields(decision)
    assert fields["source"] == "strong_fused_correction"
    assert fields["boundary"] == "mutation"
    assert fields["mutating"] is True
    assert fields["accept_class"] == "zero"
    assert fields["kept_steps"] == 0
    assert fields["strong_route"] is True
    assert fields["fused_correction"] is True
    assert fields["withheld_mutations"] == 1
    assert fields["delivered_tool_calls"] == 1
    assert fields["corrector_action_type"] == WRITE_TOOL
    assert fields["corrector_generation_count"] == 0
    assert fields["verdict"]["accepted_steps"] == 0
    assert fields["verdict"]["fused_proposed"] is True
    assert fields["verdict"]["reason"].startswith("Step 1: recipient")


def test_null_replacement_goes_to_one_strong_repair_turn():
    """A null replacement: exactly one Strong repair turn, never re-verified."""

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        # backend.calls already holds the in-flight call: dispatch on its phase.
        if backend.calls[-1]["phase"] == "strong_verify":
            return rpb.PostResult(
                200,
                _completion(_verdict(0, reason="Step 1: wrong recipient."), rid="v1"),
                {},
            )
        return rpb.PostResult(
            200,
            _completion(
                None,
                rid="c1",
                tool_calls=[_call("fix1", WRITE_TOOL, {"to": "billing@meridian.com"})],
            ),
            {},
        )

    backend = _backend(post)
    decision = _gateway(backend).handle(_request())

    assert [c["phase"] for c in backend.calls] == ["weak_act", "strong_verify", "strong_correct"]
    assert decision.source == "strong_correction"
    assert _ids(decision.response) == ["fix1"]
    fields = df.decision_fields(decision)
    assert fields["fused_correction"] is False
    assert fields["strong_route"] is True
    assert fields["corrector_generation_count"] == 1
    assert fields["correction_tool_call_count"] == 1
    assert fields["verdict"]["fused_proposed"] is False


def test_unknown_tool_replacement_commits_under_relaxed_validation():
    """With strict_action_validation off, a replacement naming a tool outside the contracts
    still commits: the environment answers it as an error observation instead of the gateway
    forcing a repair turn."""

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        if backend.calls[-1]["phase"] == "strong_verify":
            return rpb.PostResult(
                200,
                _completion(
                    _verdict(
                        0,
                        reason="Step 1: wrong recipient.",
                        replacement={
                            "tool": "salesforce_delete_account",
                            "arguments": {"id": "x"},
                        },
                    ),
                    rid="v1",
                ),
                {},
            )
        raise AssertionError("no repair turn expected")

    backend = _backend(post)
    decision = _gateway(backend).handle(_request())

    assert decision.source == "strong_fused_correction"
    delivered = decision.response["choices"][0]["message"]["tool_calls"]
    assert delivered[0]["function"]["name"] == "salesforce_delete_account"


def test_nullable_required_schema_replacement_commits():
    """With strict_action_validation off, a replacement omitting required-but-nullable keys
    commits: the environment is the argument validator."""

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        return rpb.PostResult(
            200,
            _completion(
                _verdict(
                    0,
                    reason="Step 1: wrong recipient.",
                    replacement={"tool": WRITE_TOOL, "arguments": {"to": "a@b.com"}},
                ),
                rid="v1",
            ),
            {},
        )

    backend = _backend(post)
    decision = _gateway(backend).handle(_request(tools=AB_SCHEMA_TOOLS))

    assert decision.source == "strong_fused_correction"
    delivered = decision.response["choices"][0]["message"]["tool_calls"]
    assert json.loads(delivered[0]["function"]["arguments"]) == {"to": "a@b.com"}


def test_replacement_that_replays_the_rejected_step_falls_back_to_repair():
    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        if backend.calls[-1]["phase"] == "strong_verify":
            return rpb.PostResult(
                200,
                _completion(
                    _verdict(
                        0,
                        reason="Step 1: wrong recipient.",
                        replacement={"tool": WRITE_TOOL, "arguments": {"to": "wrong@x.com"}},
                    ),
                    rid="v1",
                ),
                {},
            )
        return rpb.PostResult(
            200,
            _completion(
                None,
                rid="c1",
                tool_calls=[_call("fix1", WRITE_TOOL, {"to": "billing@meridian.com"})],
            ),
            {},
        )

    backend = _backend(post)
    decision = _gateway(backend).handle(_request())

    assert decision.source == "strong_correction"
    assert _ids(decision.response) == ["fix1"]
    brief = df.decision_fields(decision)["verdict"]
    assert brief["replacement_validation_failure_kind"] == "replays_rejected"
    assert brief["fused_proposed"] is False
    assert df.decision_fields(decision)["replacement_validation_failure_count"] == 1


def test_schema_invalid_repair_call_is_delivered_instead_of_aborting():
    """With strict_action_validation off, a repair call with bad argument types is delivered:
    the environment soft-fails it as an error observation and the rollout is not aborted."""
    strong_calls = []

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        strong_calls.append(body)
        if len(strong_calls) == 1:
            return rpb.PostResult(
                200,
                _completion(_verdict(0, reason="Step 1: wrong recipient."), rid="v1"),
                {},
            )
        return rpb.PostResult(
            200,
            _completion(
                None,
                rid="c1",
                tool_calls=[
                    _call(
                        "fix1",
                        WRITE_TOOL,
                        {"to": "a@b.com", "subject": 123, "body": ["x"]},
                    )
                ],
            ),
            {},
        )

    backend = _backend(post)
    decision = _gateway(backend).handle(_request(tools=AB_SCHEMA_TOOLS))

    assert decision.source == "strong_correction"
    delivered = decision.response["choices"][0]["message"]["tool_calls"]
    assert json.loads(delivered[0]["function"]["arguments"])["to"] == "a@b.com"


def test_stray_verdict_keys_are_discarded_and_the_replacement_commits():
    """Keys outside the three-key verdict (a null rejection_code, a corrector) are discarded and
    recorded; they neither fail the verdict nor cost a format retry."""

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return _weak_wrong_recipient()
        stray = json.dumps(
            {
                "reason": "Step 1: recipient wrong@x.com is not the account email.",
                "accepted_steps": 0,
                "rejection_code": None,
                "corrector": "strong",
                "replacement_action": {
                    "tool": WRITE_TOOL,
                    "arguments": {
                        "to": "billing@meridian.com",
                        "subject": None,
                        "body": None,
                    },
                },
            }
        )
        return rpb.PostResult(200, _completion(stray, rid="v1"), {})

    backend = _backend(post)
    decision = _gateway(backend).handle(_request(tools=AB_SCHEMA_TOOLS))

    assert [c["phase"] for c in backend.calls] == ["weak_act", "strong_verify"]
    assert decision.source == "strong_fused_correction"
    delivered = decision.response["choices"][0]["message"]["tool_calls"]
    assert json.loads(delivered[0]["function"]["arguments"]) == {
        "to": "billing@meridian.com",
        "subject": None,
        "body": None,
    }
    brief = df.decision_fields(decision)["verdict"]
    assert brief["format_retries"] == 0
    assert sorted(brief["discarded_verdict_fields"]) == ["corrector", "rejection_code"]


# Irrevocable scope: reads commit unreviewed, replies stay gated ------------------------


def test_read_only_turn_delivers_without_a_strong_call():
    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return rpb.PostResult(
                200,
                _completion(
                    None,
                    rid="w1",
                    tool_calls=[_call("r1", READ_TOOL, {"query": "Meridian"})],
                ),
                {},
            )
        raise AssertionError("strong must not be called for a read-only turn")

    backend = _backend(post)
    decision = _gateway(backend).handle(_request())

    assert decision.source == "weak_read"
    assert _ids(decision.response) == ["r1"]
    assert [c["phase"] for c in backend.calls] == ["weak_act"]
    fields = df.decision_fields(decision)
    assert fields["verdict"] is None
    assert fields["accept_class"] is None
    assert fields["strong_route"] is False
    assert fields["delivered_tool_calls"] == 1


def test_message_boundary_is_verified_but_never_fused():
    """Replies stay gated. The verdict contract offers replacement_action at every reviewed
    boundary, but the profile fuses only at mutations: a replacement at a message boundary is
    not executed and a Strong repair turn repairs the reply instead."""
    strong_calls = []

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return rpb.PostResult(
                200, _completion("All done, invoices sent.", rid="w1"), {}
            )
        strong_calls.append(body)
        if backend.calls[-1]["phase"] == "strong_verify":
            return rpb.PostResult(
                200,
                _completion(
                    _verdict(
                        0,
                        reason="Step 1: reply claims work not yet done.",
                        replacement={
                            "tool": READ_TOOL,
                            "arguments": {"query": "Meridian"},
                        },
                    ),
                    rid="v1",
                ),
                {},
            )
        return rpb.PostResult(
            200,
            _completion(
                None,
                rid="c1",
                tool_calls=[_call("fix1", READ_TOOL, {"query": "Meridian"})],
            ),
            {},
        )

    backend = _backend(post)
    decision = _gateway(backend).handle(_request())

    assert "replacement_action" in strong_calls[0]["messages"][0]["content"]
    assert decision.source == "strong_correction"
    assert decision.boundary == "message"
    assert _ids(decision.response) == ["fix1"]
    fields = df.decision_fields(decision)
    assert fields["fused_correction"] is False
    assert fields["corrector_action_type"] == READ_TOOL


def test_repeated_read_batch_is_routed_to_strong_repair():
    """repeat_call_guard: a drafted batch that only repeats executed calls is withheld and
    repaired by Strong without a review."""
    read = _call("r1", READ_TOOL, {"query": "Meridian"})
    repeat = _call("r2", READ_TOOL, {"query": "Meridian"})
    weak_turns = [read, repeat]

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return rpb.PostResult(
                200, _completion(None, rid="w", tool_calls=[weak_turns.pop(0)]), {}
            )
        assert backend.calls[-1]["phase"] == "strong_correct"
        return rpb.PostResult(200, _completion("Meridian has two open orders.", rid="c1"), {})

    backend = _backend(post)
    gateway = _gateway(backend)
    first = gateway.handle(_request())
    assert first.source == "weak_read"

    follow_up = _request()
    follow_up["messages"] = [
        *follow_up["messages"],
        {"role": "assistant", "content": None, "tool_calls": [read]},
        {"role": "tool", "tool_call_id": "r1", "content": "[{'id': 'o1'}, {'id': 'o2'}]"},
    ]
    decision = gateway.handle(follow_up)

    assert decision.source == "strong_correction"
    assert decision.boundary == "repeat"
    assert decision.repeat_rejected == 1
    assert [c["phase"] for c in backend.calls] == ["weak_act", "weak_act", "strong_correct"]
    assert decision.response["choices"][0]["message"]["content"] == (
        "Meridian has two open orders."
    )
    fields = df.decision_fields(decision)
    assert fields["corrector_action_type"] == "reply"
    assert fields["delivered_tool_calls"] == 0


# decision_fields on its own --------------------------------------------------------------


def test_decision_fields_partial_accept_and_format_failure_tail():
    from specagent.protocol.types import PrefixVerdict
    from specagent.runtime.gateway import GatewayDecision

    steps = (
        {"index": 1, "action": {"type": "tool", "tool": READ_TOOL, "arguments": {}}},
        {"index": 2, "action": {"type": "tool", "tool": WRITE_TOOL, "arguments": {}}},
        {"index": 3, "action": {"type": "message", "content": "done"}},
    )
    verdict = PrefixVerdict(
        accepted_steps=1,
        reason="",
        raw="x" * 500,
        diagnostic_status="format_failure",
        format_failure_kind="unparseable",
        format_retries=1,
    )
    decision = GatewayDecision(
        response={"choices": [{"message": {"tool_calls": [{"id": "a"}, {"id": "b"}]}}]},
        source="weak_partial_then_correction",
        verdict=verdict,
        candidate_length=3,
        candidate_steps=steps,
    )
    fields = df.decision_fields(decision)
    assert fields["accept_class"] == "partial"
    assert fields["kept_steps"] == 1
    # The rejected suffix holds a state-changing call and a reply; both are irrevocable, the
    # accepted read is not counted.
    assert fields["withheld_mutations"] == 2
    assert fields["format_retries"] == 1
    assert fields["delivered_tool_calls"] == 2
    assert fields["strong_route"] is False
    brief = fields["verdict"]
    assert brief["reason"] is None
    assert brief["format_failure_kind"] == "unparseable"
    assert brief["raw_tail"] == "x" * 400
