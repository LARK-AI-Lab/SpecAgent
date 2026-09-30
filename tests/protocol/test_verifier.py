import json
import unittest

from specagent.protocol import parse_prefix_verdict
from specagent.protocol.verifier import (
    ANSWER_REVIEW_CHECKS,
    RP_FUSED_VERIFIER_SYSTEM_PROMPT,
    STRICT_ANSWER_EXAMPLE,
    STRICT_ANSWER_RULE,
    VERIFIER_CONTRACTS,
    VerdictFormatError,
    strict_answer_review,
    verifier_messages,
)


class PrefixVerdictParserTest(unittest.TestCase):
    def test_parses_structured_verdict(self) -> None:
        verdict = parse_prefix_verdict(
            '{"reason":"step 3 writes the wrong file","accepted_steps":2,'
            '"replacement_action":null,"confidence":0.8}',
            candidate_length=4,
        )
        self.assertEqual(verdict.accepted_steps, 2)
        self.assertEqual(verdict.confidence, 0.8)
        self.assertEqual(verdict.rejection_code.value, "other")

    def test_rejects_out_of_bounds_or_unstructured_output(self) -> None:
        # accepted_steps beyond the drafted length is never clamped: committing steps that were
        # never produced must fail closed so the caller retries or falls back.
        with self.assertRaises(VerdictFormatError) as raised:
            parse_prefix_verdict(
                {"reason": None, "accepted_steps": 5, "replacement_action": None},
                candidate_length=4,
            )
        self.assertEqual(raised.exception.kind, "invalid_accepted_steps")
        for bad in (-1, True, "2"):
            with self.subTest(accepted=bad), self.assertRaises(VerdictFormatError):
                parse_prefix_verdict(
                    {"reason": "r", "accepted_steps": bad, "replacement_action": None},
                    candidate_length=4,
                )
        with self.assertRaises(ValueError):
            parse_prefix_verdict("accept the first two", candidate_length=4)
        with self.assertRaises(ValueError):
            parse_prefix_verdict(
                {"reason": None, "accepted_steps": 0, "replacement_action": None},
                candidate_length=0,
            )

    def test_full_accept_discards_stray_rejection_code(self) -> None:
        verdict = parse_prefix_verdict(
            {"reason": None, "accepted_steps": 4, "replacement_action": None},
            candidate_length=4,
        )
        self.assertIsNone(verdict.rejection_code)
        self.assertEqual(verdict.discarded_verdict_fields, ())

        # Safe salvage: a stray code on a full accept cannot affect the prefix, so it is
        # discarded and recorded rather than retried.
        verdict = parse_prefix_verdict(
            {
                "reason": None,
                "accepted_steps": 4,
                "rejection_code": "other",
                "replacement_action": None,
            },
            candidate_length=4,
        )
        self.assertIsNone(verdict.rejection_code)
        self.assertIn("rejection_code", verdict.discarded_verdict_fields)

    def test_parses_reason_without_confidence(self) -> None:
        verdict = parse_prefix_verdict(
            {
                "reason": "  step 2 writes  'not published' although the input says it was\n",
                "accepted_steps": 1,
                "replacement_action": None,
            },
            candidate_length=3,
        )
        self.assertEqual(verdict.accepted_steps, 1)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertEqual(
            verdict.reason, "step 2 writes 'not published' although the input says it was"
        )
        long = parse_prefix_verdict(
            {"reason": "x" * 1000, "accepted_steps": 1, "replacement_action": None},
            candidate_length=3,
        )
        self.assertEqual(len(long.reason), 800)
        # A full accept canonicalizes a stray reason to null and flags it.
        full = parse_prefix_verdict(
            {"reason": "x" * 1000, "accepted_steps": 3, "replacement_action": None},
            candidate_length=3,
        )
        self.assertEqual(full.reason, "")
        self.assertTrue(full.reason_on_full_accept)
        with self.assertRaises(ValueError):
            parse_prefix_verdict(
                {"reason": ["not", "a", "string"], "accepted_steps": 0},
                candidate_length=1,
            )

    def test_parses_markdown_fenced_json(self) -> None:
        verdict = parse_prefix_verdict(
            '```json\n{"reason": "wrong step", "accepted_steps": 0,'
            ' "replacement_action": null}\n```',
            candidate_length=4,
        )
        self.assertEqual(verdict.accepted_steps, 0)
        self.assertEqual(verdict.reason, "wrong step")
        self.assertEqual(verdict.rejection_code.value, "other")

    def test_parses_optional_replacement_action_on_rejection(self) -> None:
        verdict = parse_prefix_verdict(
            {
                "reason": "step 1 targets the wrong file",
                "accepted_steps": 0,
                "replacement_action": {
                    "tool": "write_file",
                    "arguments": {"path": "right.txt", "content": "ok"},
                },
            },
            candidate_length=1,
        )

        self.assertEqual(
            verdict.replacement_action,
            {
                "tool": "write_file",
                "arguments": {"path": "right.txt", "content": "ok"},
            },
        )
        self.assertEqual(verdict.replacement_validation_failure_kind, "")

    def test_replacement_action_is_shape_checked_and_ignored_on_acceptance(self) -> None:
        accepted = parse_prefix_verdict(
            {
                "reason": "safe",
                "accepted_steps": 1,
                "replacement_action": {"type": "message", "content": "not used"},
            },
            candidate_length=1,
        )
        self.assertIsNone(accepted.replacement_action)
        self.assertTrue(accepted.replacement_on_full_accept)

        malformed = (
            {"type": "message", "content": ""},
            {"tool": "", "arguments": {}},
            {"tool": "write_file", "arguments": "{}"},
            {"type": "message", "content": "ok", "tool": "write_file"},
            ["write_file", {}],
        )
        for action in malformed:
            with self.subTest(action=action):
                verdict = parse_prefix_verdict(
                    {"reason": "bad", "accepted_steps": 0, "replacement_action": action},
                    candidate_length=1,
                )
                self.assertIsNone(verdict.replacement_action)
                self.assertEqual(
                    verdict.replacement_validation_failure_kind, "malformed_replacement"
                )

    def test_malformed_replacement_suffix_keeps_core_verdict_but_discards_action(self) -> None:
        verdict = parse_prefix_verdict(
            '{"reason":"bad path","accepted_steps":0,'
            '"replacement_action":{"tool":"write","arguments":',
            candidate_length=1,
        )

        self.assertEqual(verdict.accepted_steps, 0)
        self.assertEqual(verdict.reason, "bad path")
        self.assertEqual(verdict.rejection_code.value, "other")
        self.assertIsNone(verdict.replacement_action)
        self.assertEqual(verdict.replacement_validation_failure_kind, "malformed_replacement")

    def test_only_known_contracts_are_accepted(self) -> None:
        self.assertEqual(VERIFIER_CONTRACTS, ("rp_fused", "reflective_prefix"))
        raw = {"reason": None, "accepted_steps": 1, "replacement_action": None}
        for contract in VERIFIER_CONTRACTS:
            with self.subTest(contract=contract):
                self.assertEqual(
                    parse_prefix_verdict(raw, 1, contract=contract).accepted_steps, 1
                )
        for contract in ("unknown", ""):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                parse_prefix_verdict(raw, 1, contract=contract)


class OptionalRejectionCodeTests(unittest.TestCase):
    """The contract has no rejection taxonomy; the runtime supplies the sentinel.

    A malformed or explicit-null code cannot affect the accepted prefix or the reason, so it is
    discarded and recorded on the verdict rather than retried.
    """

    def test_absent_code_on_rejection_becomes_other(self) -> None:
        verdict = parse_prefix_verdict('{"reason": "r", "accepted_steps": 0}', 2)
        self.assertEqual(verdict.rejection_code.value, "other")
        self.assertEqual(verdict.accepted_steps, 0)

    def test_absent_code_on_full_acceptance_stays_none(self) -> None:
        verdict = parse_prefix_verdict('{"reason": "ok", "accepted_steps": 2}', 2)
        self.assertIsNone(verdict.rejection_code)

    def test_explicit_null_on_a_rejection_is_discarded_as_other(self) -> None:
        verdict = parse_prefix_verdict(
            '{"reason": "r", "accepted_steps": 0, "rejection_code": null}', 2
        )
        self.assertEqual(verdict.rejection_code.value, "other")
        self.assertIn("rejection_code", verdict.discarded_verdict_fields)

    def test_corrector_hint_is_discarded(self) -> None:
        verdict = parse_prefix_verdict(
            '{"reason":"r","accepted_steps":0,"replacement_action":null,"corrector":"weak"}', 1
        )
        self.assertIsNone(verdict.corrector)
        self.assertIn("corrector", verdict.discarded_verdict_fields)


class ReflectivePrefixContractTest(unittest.TestCase):
    @staticmethod
    def parse(raw, k=3):
        return parse_prefix_verdict(raw, candidate_length=k, contract="reflective_prefix")

    def test_full_accept_normalizes_strays_and_records_the_violation(self) -> None:
        clean = self.parse({"reason": None, "accepted_steps": 3, "replacement_action": None})
        self.assertEqual(clean.diagnostic_status, "model_verdict")
        self.assertIsNone(clean.rejection_code)

        raw = json.dumps(
            {
                "reason": "step 3 is wrong",
                "accepted_steps": 3,
                "replacement_action": {"tool": "read_file", "arguments": {}},
            }
        )
        stray = self.parse(raw)
        self.assertEqual(stray.accepted_steps, 3)
        self.assertEqual(stray.reason, "")
        self.assertIsNone(stray.replacement_action)
        self.assertEqual(stray.diagnostic_status, "full_accept_contract_violation")
        self.assertEqual(stray.raw, raw)

    def test_partial_accept_keeps_reason_and_tool_replacement(self) -> None:
        verdict = self.parse(
            {
                "reason": "  step 2  reads the wrong file ",
                "accepted_steps": 1,
                "replacement_action": {"tool": "read_file", "arguments": {"path": "y"}},
            }
        )
        self.assertEqual(verdict.accepted_steps, 1)
        self.assertEqual(verdict.reason, "step 2 reads the wrong file")
        self.assertEqual(verdict.rejection_code.value, "other")
        self.assertEqual(
            verdict.replacement_action, {"tool": "read_file", "arguments": {"path": "y"}}
        )
        null_action = self.parse(
            {"reason": "bad", "accepted_steps": 0, "replacement_action": None}
        )
        self.assertIsNone(null_action.replacement_action)

    def test_partial_accept_stays_strict(self) -> None:
        invalid = (
            {"reason": "", "accepted_steps": 1, "replacement_action": None},
            {"reason": None, "accepted_steps": 1, "replacement_action": None},
            {
                "reason": "bad",
                "accepted_steps": 1,
                "replacement_action": {"type": "message", "content": "hi"},
            },
            {"reason": "bad", "accepted_steps": 1, "replacement_action": {"tool": "x"}},
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.parse(raw)

    def test_key_set_is_exact(self) -> None:
        with self.assertRaises(ValueError):
            self.parse(
                {
                    "reason": None,
                    "accepted_steps": 3,
                    "replacement_action": None,
                    "confidence": 1.0,
                }
            )
        with self.assertRaises(ValueError):
            self.parse({"reason": None, "accepted_steps": 3})


class VerifierPromptTest(unittest.TestCase):
    def test_prompt_commits_only_correct_mutations(self) -> None:
        prompt = " ".join(RP_FUSED_VERIFIER_SYSTEM_PROMPT.split())
        self.assertIn("Recoverable does not mean acceptable", prompt)
        self.assertIn("anchors.workspace_dir", prompt)
        self.assertIn("accepted_steps<K", prompt)
        self.assertIn("no markdown fences", prompt)

    def test_prompt_declares_three_keys_and_treats_replies_as_turn_ending(self) -> None:
        prompt = RP_FUSED_VERIFIER_SYSTEM_PROMPT
        self.assertIn(
            '{"reason": <string or null>, "accepted_steps": <int>, '
            '"replacement_action": <tool action object or null>}',
            prompt,
        )
        self.assertNotIn('"confidence"', prompt)
        self.assertNotIn('"rejection_code"', prompt)
        self.assertNotIn('"corrector"', prompt)
        self.assertIn("ends the agent's turn", prompt)
        self.assertIn("Read-only question answering is the exception", prompt)
        self.assertIn("An executed read cannot be undone", " ".join(prompt.split()))

    def test_verifier_messages_carry_prompt_and_candidate_length(self) -> None:
        payload = {"task": {"instruction": "t"}, "proposed_trajectory": [{}, {}]}
        messages = verifier_messages(payload)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], RP_FUSED_VERIFIER_SYSTEM_PROMPT)
        self.assertEqual(json.loads(messages[1]["content"])["candidate_length"], 2)
        self.assertNotIn("candidate_length", payload)

    def test_strict_answer_review_replaces_the_permissive_qa_rule(self) -> None:
        strict = strict_answer_review(RP_FUSED_VERIFIER_SYSTEM_PROMPT)
        self.assertIn(STRICT_ANSWER_RULE, strict)
        self.assertIn(STRICT_ANSWER_EXAMPLE, strict)
        self.assertNotIn("Read-only question answering is the exception", strict)
        self.assertNotIn("although confidence is low: accept", strict)
        # The permissive text is gone, so a second application cannot find it.
        with self.assertRaises(AssertionError):
            strict_answer_review(strict)

    def test_answer_review_checks_are_a_separate_suffix(self) -> None:
        self.assertNotIn(ANSWER_REVIEW_CHECKS, RP_FUSED_VERIFIER_SYSTEM_PROMPT)
        self.assertTrue(ANSWER_REVIEW_CHECKS.startswith("Answer review checks"))


class UnverifiedIdentifierRuleTest(unittest.TestCase):
    """A guessed lookup key returning [] must not be read as proof the entity does not exist."""

    def test_the_prompt_carries_the_rule(self) -> None:
        prompt = RP_FUSED_VERIFIER_SYSTEM_PROMPT
        self.assertIn("evidence about the id, not about the world", prompt)
        self.assertIn("name the\n  unverified identifier as the defect", prompt)

    def test_the_rule_forbids_promoting_one_empty_result_to_a_settled_fact(self) -> None:
        prompt = RP_FUSED_VERIFIER_SYSTEM_PROMPT
        self.assertIn("Never treat one empty result as", prompt)
        self.assertIn("your own later verdicts may build on", prompt)


class UserDecisionBoundaryTest(unittest.TestCase):
    """A reply that puts a user-owned decision back to the user is not premature.

    The carve-out must sit before the premature-reply rule fires, and it must be decidable from
    the reply itself.
    """

    def test_the_carve_out_is_a_precondition_of_the_premature_rule(self) -> None:
        prompt = RP_FUSED_VERIFIER_SYSTEM_PROMPT
        self.assertIn("Before\n  you call a reply premature", prompt)
        self.assertIn("putting a choice back to the user IS the correct terminal", prompt)
        self.assertIn(
            "recommends one option and asks them to confirm\n  has done the work", prompt
        )

    def test_the_carve_out_is_stated_once(self) -> None:
        prompt = RP_FUSED_VERIFIER_SYSTEM_PROMPT
        self.assertNotIn("a decision the task reserves for the user", prompt)
        self.assertEqual(prompt.count("putting a choice back to the user"), 1)


if __name__ == "__main__":
    unittest.main()
