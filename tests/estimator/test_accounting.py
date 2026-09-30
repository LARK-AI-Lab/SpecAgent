"""Resource ledger: token/status accounting and non-upgrading quantity merges."""

import json
import math
import unittest
from pathlib import Path

from specagent.protocol import (
    InferencePhase,
    InferenceSpan,
    ModelRole,
    ResourceLedger,
)
from specagent.protocol.accounting import (
    LEGACY_CALL_KEYS,
    CanonicalPhase,
    LedgerConflictError,
    QuantityStatus,
    ResourceQuantity,
    canonical_phase,
    merge_quantities,
    span_from_legacy_call,
    token_fields_from_usage,
    unknown_quantity,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _span(**kwargs) -> InferenceSpan:
    payload = {
        "role": ModelRole.WEAK,
        "phase": InferencePhase.DRAFT,
        "input_tokens": 10,
        "output_tokens": 4,
        "duration_seconds": 0.2,
    }
    payload.update(kwargs)
    return InferenceSpan(**payload)


class ResourceLedgerTest(unittest.TestCase):
    def test_summarizes_role_phase_tokens_and_gpu_seconds(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.DRAFT,
                input_tokens=100,
                output_tokens=40,
                committed_output_tokens=30,
                discarded_output_tokens=10,
                duration_seconds=2.0,
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.VERIFICATION,
                input_tokens=300,
                output_tokens=5,
                cached_input_tokens=80,
                duration_seconds=0.5,
                gpu_count=4,
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.CORRECTION,
                input_tokens=200,
                output_tokens=20,
                duration_seconds=1.0,
                gpu_count=4,
            )
        )

        summary = ledger.summarize()

        self.assertEqual(summary.total_gpu_seconds, 8.0)
        self.assertEqual(summary.strong_gpu_seconds, 6.0)
        self.assertEqual(summary.weak_committed_output_tokens, 30)
        self.assertEqual(summary.weak_discarded_output_tokens, 10)
        self.assertEqual(summary.strong_verifier_input_tokens, 300)
        self.assertEqual(summary.strong_correction_output_tokens, 20)
        self.assertEqual(summary.strong_act_output_tokens, 0)

    def test_solo_act_output_tokens_are_committed(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.ACT,
                input_tokens=50,
                output_tokens=400,
                duration_seconds=1.0,
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.ACT,
                input_tokens=80,
                output_tokens=64000,
                duration_seconds=10.0,
                gpu_count=4,
            )
        )

        summary = ledger.summarize()

        self.assertEqual(summary.weak_committed_output_tokens, 400)
        self.assertEqual(summary.weak_discarded_output_tokens, 0)
        self.assertEqual(summary.strong_act_input_tokens, 80)
        self.assertEqual(summary.strong_act_output_tokens, 64000)
        self.assertEqual(summary.strong_generation_output_tokens, 64000)
        self.assertEqual(summary.strong_gpu_seconds, 40.0)
        self.assertEqual(summary.strong_verifier_output_tokens, 0)
        self.assertEqual(summary.strong_correction_output_tokens, 0)

    def test_solo_draft_without_disposition_counts_as_committed(self) -> None:
        """Legacy solo dumps tagged DRAFT and left committed_output_tokens=0."""
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.DRAFT,
                input_tokens=10,
                output_tokens=123,
                duration_seconds=0.5,
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.DRAFT,
                input_tokens=20,
                output_tokens=456,
                duration_seconds=1.0,
            )
        )

        summary = ledger.summarize()

        self.assertEqual(summary.weak_committed_output_tokens, 123)
        self.assertEqual(summary.strong_act_output_tokens, 456)


class ResourceLedgerV2Test(unittest.TestCase):
    def test_legacy_call_projection_keeps_seven_keys(self) -> None:
        span = _span()
        self.assertEqual(tuple(span.to_dict()), LEGACY_CALL_KEYS)

    def test_phase_map_covers_weak_verify_correct_opening_summary(self) -> None:
        self.assertEqual(
            canonical_phase(ModelRole.WEAK, InferencePhase.DRAFT),
            CanonicalPhase.WEAK_DRAFT,
        )
        self.assertEqual(
            canonical_phase(ModelRole.WEAK, InferencePhase.DRAFT, intent="retry"),
            CanonicalPhase.WEAK_RETRY,
        )
        self.assertEqual(
            canonical_phase(ModelRole.STRONG, InferencePhase.VERIFICATION),
            CanonicalPhase.STRONG_VERIFY,
        )
        self.assertEqual(
            canonical_phase(ModelRole.STRONG, InferencePhase.CORRECTION),
            CanonicalPhase.STRONG_CORRECT,
        )
        self.assertEqual(
            canonical_phase(ModelRole.STRONG, InferencePhase.OPENING),
            CanonicalPhase.STRONG_OPENING,
        )
        self.assertEqual(
            canonical_phase(ModelRole.WEAK, InferencePhase.SUMMARY),
            CanonicalPhase.SUMMARY,
        )
        self.assertEqual(
            canonical_phase("other", InferencePhase.DRAFT),
            CanonicalPhase.UNKNOWN,
        )

    def test_duplicate_telemetry_events_charge_once(self) -> None:
        ledger = ResourceLedger()
        first = _span(request_id="r1", transport_attempt_id="a1", input_tokens=11)
        duplicate = _span(request_id="r1", transport_attempt_id="a1", input_tokens=11)
        self.assertTrue(ledger.add(first))
        self.assertFalse(ledger.add(duplicate))
        summary = ledger.summarize()
        self.assertEqual(summary.n_calls, 1)
        self.assertEqual(summary.weak_input_tokens, 11)

    def test_failed_then_success_are_two_attempts_and_unknown_is_not_zero(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            _span(
                request_id="r2",
                transport_attempt_index=1,
                status="failed",
                usage_status=QuantityStatus.UNKNOWN,
                input_tokens=0,
                output_tokens=0,
                duration_seconds=0.4,
            )
        )
        ledger.add(
            _span(
                request_id="r2",
                transport_attempt_index=2,
                status="ok",
                input_tokens=15,
                output_tokens=3,
                duration_seconds=0.5,
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.n_transport_attempts, 2)
        self.assertEqual(summary.n_failed_calls, 1)
        self.assertEqual(summary.n_unknown_usage, 1)
        self.assertEqual(summary.weak_input_tokens, 15)
        self.assertEqual(summary.token_totals_status, QuantityStatus.INCOMPLETE.value)
        self.assertEqual(summary.flops.status, QuantityStatus.UNKNOWN)
        self.assertIsNone(summary.flops.value)

    def test_cached_replay_is_not_charged(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            _span(
                request_id="orig",
                transport_attempt_id="live",
                input_tokens=40,
                output_tokens=8,
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.DRAFT,
                input_tokens=0,
                output_tokens=0,
                duration_seconds=0.01,
                request_id="replay",
                source="cached_replay",
                charge_inference=False,
                usage_status=QuantityStatus.NOT_APPLICABLE,
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.n_calls, 1)
        self.assertEqual(summary.n_cached_replays, 1)
        self.assertEqual(summary.weak_input_tokens, 40)
        self.assertEqual(summary.n_transport_attempts, 1)

    def test_opening_summary_retry_and_unattributed_parent(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.OPENING,
                input_tokens=21,
                output_tokens=7,
                duration_seconds=1.2,
                parent_task_id="t1",
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.DRAFT,
                input_tokens=12,
                output_tokens=4,
                duration_seconds=0.3,
                intent="retry",
                parent_task_id="t1",
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.SUMMARY,
                input_tokens=90,
                output_tokens=20,
                duration_seconds=0.8,
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.n_strong_opening_calls, 1)
        self.assertEqual(summary.strong_opening_output_tokens, 7)
        self.assertEqual(summary.n_weak_retry_calls, 1)
        self.assertEqual(summary.n_summary_calls, 1)
        self.assertEqual(summary.n_unattributed, 1)
        self.assertEqual(summary.unattributed_input_tokens, 90)
        self.assertEqual(summary.phase_counts[CanonicalPhase.SUMMARY.value], 1)
        # Opening is Strong work but not verifier/correct/act.
        self.assertEqual(summary.n_strong_verify_calls, 0)
        self.assertEqual(summary.strong_act_output_tokens, 0)
        self.assertEqual(summary.n_strong_calls, 1)
        self.assertEqual(summary.strong_generation_output_tokens, 7)
        self.assertEqual(summary.strong_generation_output_tokens_act_correction, 0)

    def test_flops_hbm_stay_unknown_and_are_not_duration_or_tp_fillins(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.CORRECTION,
                input_tokens=10,
                output_tokens=10,
                duration_seconds=5.0,
                gpu_count=8,
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.strong_gpu_seconds, 40.0)
        self.assertIsNone(summary.flops.value)
        self.assertEqual(summary.flops.status, QuantityStatus.UNKNOWN)
        self.assertEqual(summary.flops.provenance, "no_server_counter")
        self.assertIsNone(summary.hbm_read_bytes.value)
        with self.assertRaises(ValueError):
            ResourceQuantity(value=5.0 * 8 * 397e9, status=QuantityStatus.UNKNOWN)

    def test_moe_batch_and_tp_are_not_multiplied_into_unknown_totals(self) -> None:
        measured = ResourceQuantity(
            value=12.0,
            status=QuantityStatus.MEASURED,
            provenance="measured_dram_bytes",
            coverage=0.4,
        )
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.VERIFICATION,
                input_tokens=8,
                output_tokens=1,
                duration_seconds=0.2,
                gpu_count=16,
                hbm_read_bytes=measured,
                parent_task_id="t1",
            )
        )
        ledger.add(
            InferenceSpan(
                role=ModelRole.STRONG,
                phase=InferencePhase.CORRECTION,
                input_tokens=9,
                output_tokens=2,
                duration_seconds=0.3,
                gpu_count=16,
                parent_task_id="t1",
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.hbm_read_bytes.status, QuantityStatus.INCOMPLETE)
        self.assertIsNone(summary.hbm_read_bytes.value)
        self.assertEqual(summary.hbm_read_bytes.observed_subtotal, 12.0)
        self.assertEqual(summary.hbm_read_bytes.observed_measured, 12.0)
        self.assertIsNone(summary.hbm_read_bytes.coverage)
        self.assertNotEqual(summary.hbm_read_bytes.observed_subtotal, 12.0 * 16)
        self.assertIsNone(summary.flops.value)

    def test_legacy_v1_calls_load_without_inventing_flops(self) -> None:
        payload = json.loads(
            (FIXTURES / "legacy_calls_v1.json").read_text(encoding="utf-8")
        )
        ledger = ResourceLedger()
        for row in payload:
            ledger.add(span_from_legacy_call(row))
        summary = ledger.summarize()
        self.assertEqual(summary.n_calls, 2)
        self.assertEqual(summary.weak_input_tokens, 100)
        self.assertEqual(summary.strong_correction_output_tokens, 20)
        self.assertEqual(summary.token_totals_status, QuantityStatus.LEGACY_UNSPECIFIED.value)
        self.assertIsNone(summary.flops.value)
        self.assertEqual(summary.calls[0].keys(), set(LEGACY_CALL_KEYS))
        rebuilt = span_from_legacy_call(summary.calls[0])
        self.assertEqual(rebuilt.input_tokens, 100)
        self.assertEqual(rebuilt.cache_provenance, "legacy_unspecified")

    def test_per_role_token_status_survives_strong_legacy(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            _span(
                role=ModelRole.WEAK,
                usage_status=QuantityStatus.MEASURED,
                input_tokens=40,
                output_tokens=6,
                request_id="weak-measured",
            )
        )
        ledger.add(
            _span(
                role=ModelRole.STRONG,
                phase=InferencePhase.VERIFICATION,
                usage_status=QuantityStatus.LEGACY_UNSPECIFIED,
                input_tokens=90,
                output_tokens=2,
                request_id="strong-legacy",
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.weak_input_tokens, 40)
        self.assertEqual(summary.weak_token_totals_status, QuantityStatus.MEASURED.value)
        self.assertEqual(
            summary.strong_token_totals_status,
            QuantityStatus.LEGACY_UNSPECIFIED.value,
        )
        self.assertEqual(summary.token_totals_status, QuantityStatus.INCOMPLETE.value)
        self.assertIn("mixed_role_token_status", str(summary.token_totals_status_reason))

    def test_unknown_quantity_helper_has_no_fill_in(self) -> None:
        quantity = unknown_quantity()
        self.assertIsNone(quantity.value)
        self.assertEqual(quantity.status, QuantityStatus.UNKNOWN)

    def test_merge_does_not_upgrade_coverage_status_or_fill_zeros(self) -> None:
        measured = ResourceQuantity(
            value=12,
            status=QuantityStatus.MEASURED,
            provenance="measured",
            coverage=0.4,
        )
        merged = merge_quantities([measured], observed_name="hbm_read")
        self.assertEqual(merged.value, 12)
        self.assertEqual(merged.status, QuantityStatus.MEASURED)
        self.assertEqual(merged.coverage, 0.4)
        with self.assertRaises(ValueError):
            ResourceQuantity(value=None, status=QuantityStatus.MEASURED, provenance="x")
        with self.assertRaises(ValueError):
            ResourceQuantity(value=-1, status=QuantityStatus.MEASURED, provenance="x")
        with self.assertRaises(ValueError):
            ResourceQuantity(value=math.nan, status=QuantityStatus.ESTIMATED, provenance="x")
        with self.assertRaises(ValueError):
            ResourceQuantity(value=math.inf, status=QuantityStatus.MEASURED, provenance="x")
        legacy = ResourceQuantity(
            value=12, status=QuantityStatus.LEGACY_UNSPECIFIED, provenance="legacy"
        )
        legacy_merged = merge_quantities([legacy], observed_name="hbm_read")
        self.assertEqual(legacy_merged.status, QuantityStatus.LEGACY_UNSPECIFIED)
        self.assertEqual(legacy_merged.value, 12)
        n_a = ResourceQuantity(
            value=None, status=QuantityStatus.NOT_APPLICABLE, provenance="replay"
        )
        mixed = merge_quantities([n_a, measured], observed_name="hbm_read")
        self.assertEqual(mixed.status, QuantityStatus.MEASURED)
        self.assertEqual(mixed.value, 12)
        estimated = ResourceQuantity(
            value=3, status=QuantityStatus.ESTIMATED, provenance="est", coverage=0.4
        )
        unknown = unknown_quantity()
        partial = merge_quantities(
            [measured, estimated, unknown], observed_name="hbm_read"
        )
        self.assertEqual(partial.status, QuantityStatus.INCOMPLETE)
        self.assertIsNone(partial.value)
        self.assertEqual(partial.observed_subtotal, 15)
        self.assertEqual(partial.observed_measured, 12)
        self.assertEqual(partial.observed_estimated, 3)
        self.assertIsNone(partial.coverage)

    def test_missing_attempt_propagates_to_token_and_resource_completeness(self) -> None:
        ledger = ResourceLedger()
        ledger.add(
            InferenceSpan(
                role=ModelRole.WEAK,
                phase=InferencePhase.DRAFT,
                input_tokens=10,
                output_tokens=1,
                duration_seconds=0.1,
                request_id="r",
                transport_attempt_index=2,
                flops=ResourceQuantity(
                    value=5, status=QuantityStatus.MEASURED, provenance="est"
                ),
            )
        )
        summary = ledger.summarize()
        self.assertEqual(summary.n_transport_attempts, 2)
        self.assertEqual(summary.n_calls, 1)
        self.assertEqual(summary.weak_input_tokens, 10)
        self.assertGreaterEqual(summary.n_unknown_usage, 1)
        self.assertEqual(summary.token_totals_status, QuantityStatus.INCOMPLETE.value)
        self.assertEqual(summary.flops.status, QuantityStatus.INCOMPLETE)
        self.assertIsNone(summary.flops.value)
        self.assertEqual(summary.flops.observed_subtotal, 5)
        self.assertEqual(summary.hbm_read_bytes.status, QuantityStatus.UNKNOWN)

    def test_replay_and_live_do_not_share_dedup_key(self) -> None:
        replay = InferenceSpan(
            role=ModelRole.WEAK,
            phase=InferencePhase.DRAFT,
            input_tokens=0,
            output_tokens=0,
            duration_seconds=0.01,
            request_id="r",
            transport_attempt_id="a",
            source="cached_replay",
            charge_inference=False,
            usage_status=QuantityStatus.NOT_APPLICABLE,
        )
        live = _span(
            request_id="r",
            transport_attempt_id="a",
            input_tokens=40,
            output_tokens=8,
        )
        first = ResourceLedger()
        self.assertTrue(first.add(replay))
        self.assertTrue(first.add(live))
        self.assertEqual(first.summarize().n_calls, 1)
        self.assertEqual(first.summarize().n_cached_replays, 1)
        self.assertEqual(first.summarize().weak_input_tokens, 40)
        second = ResourceLedger()
        self.assertTrue(second.add(live))
        self.assertTrue(second.add(replay))
        self.assertEqual(second.summarize().n_calls, 1)
        self.assertEqual(second.summarize().weak_input_tokens, 40)

    def test_conflicting_identity_raises(self) -> None:
        ledger = ResourceLedger()
        ledger.add(_span(request_id="r", transport_attempt_id="a", input_tokens=100))
        with self.assertRaises(LedgerConflictError):
            ledger.add(_span(request_id="r", transport_attempt_id="a", input_tokens=999))

    def test_parent_and_provenance_conflicts_fail_closed(self) -> None:
        fields = (
            ("parent_task_id", "task-1", "task-2"),
            ("parent_request_id", "req-1", "req-2"),
            ("parent_episode_id", "ep-1", "ep-2"),
            ("provider_trace_id", "trace-1", "trace-2"),
            ("cache_provenance", "openai_usage", "legacy_unspecified"),
        )
        for name, first_value, second_value in fields:
            with self.subTest(field=name):
                ledger = ResourceLedger()
                first = _span(
                    request_id="r",
                    transport_attempt_id="a",
                    **{name: first_value},
                )
                second = _span(
                    request_id="r",
                    transport_attempt_id="a",
                    **{name: second_value},
                )
                self.assertTrue(ledger.add(first))
                with self.assertRaises(LedgerConflictError):
                    ledger.add(second)
                reversed_ledger = ResourceLedger()
                self.assertTrue(reversed_ledger.add(second))
                with self.assertRaises(LedgerConflictError):
                    reversed_ledger.add(first)
        same = ResourceLedger()
        span = _span(
            request_id="r",
            transport_attempt_id="a",
            parent_task_id="task-1",
            parent_request_id="req-1",
            parent_episode_id="ep-1",
            provider_trace_id="trace-1",
            cache_provenance="openai_usage",
        )
        self.assertTrue(same.add(span))
        self.assertFalse(
            same.add(
                _span(
                    request_id="r",
                    transport_attempt_id="a",
                    parent_task_id="task-1",
                    parent_request_id="req-1",
                    parent_episode_id="ep-1",
                    provider_trace_id="trace-1",
                    cache_provenance="openai_usage",
                )
            )
        )
        self.assertEqual(same.summarize().n_calls, 1)

    def test_attempt_index_conflict_on_attempt_id_path(self) -> None:
        ledger = ResourceLedger()
        first = _span(
            request_id="r",
            transport_attempt_id="a",
            transport_attempt_index=1,
        )
        second = _span(
            request_id="r",
            transport_attempt_id="a",
            transport_attempt_index=2,
        )
        self.assertTrue(ledger.add(first))
        with self.assertRaises(LedgerConflictError):
            ledger.add(second)

    def test_parsed_usage_adapter_keeps_known_tokens_when_incomplete(self) -> None:
        from specagent.protocol.accounting import ParsedUsage

        parsed = ParsedUsage(
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=0,
            reasoning_tokens=0,
            input_status=QuantityStatus.MEASURED,
            output_status=QuantityStatus.MEASURED,
            cached_status=QuantityStatus.UNKNOWN,
            reasoning_status=QuantityStatus.UNKNOWN,
            provenance="openai_usage",
        )
        self.assertEqual(parsed.status, QuantityStatus.INCOMPLETE)
        self.assertEqual(parsed.to_dict()["status"], "incomplete")
        span = InferenceSpan(
            role=ModelRole.WEAK,
            phase=InferencePhase.DRAFT,
            duration_seconds=0.1,
            **token_fields_from_usage(parsed),
        )
        self.assertEqual(span.usage_status, QuantityStatus.INCOMPLETE)
        self.assertEqual(span.input_tokens, 100)
        ledger = ResourceLedger()
        ledger.add(span)
        summary = ledger.summarize()
        self.assertEqual(summary.weak_input_tokens, 100)
        self.assertEqual(summary.token_totals_status, QuantityStatus.INCOMPLETE.value)
        self.assertEqual(summary.weak_cached_input_tokens, 0)
        v2 = span_from_legacy_call(
            {
                "role": "weak",
                "phase": "draft",
                "duration_seconds": 0.1,
                "input_tokens": 4,
                "output_tokens": 1,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "schema_version": "resource_ledger_v2",
            }
        )
        self.assertEqual(v2.usage_status, QuantityStatus.LEGACY_UNSPECIFIED)

    def test_summary_without_parent_and_unknown_phase_and_strong_retry(self) -> None:
        span = InferenceSpan(
            role=ModelRole.WEAK,
            phase=InferencePhase.SUMMARY,
            input_tokens=90,
            output_tokens=20,
            duration_seconds=0.8,
        )
        self.assertEqual(span.attribution, "run_unattributed")
        ledger = ResourceLedger()
        ledger.add(span)
        self.assertEqual(ledger.summarize().n_unattributed, 1)
        loaded = span_from_legacy_call(
            {
                "role": "weak",
                "phase": "weird",
                "duration_seconds": 0,
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }
        )
        self.assertEqual(loaded.phase, InferencePhase.UNKNOWN)
        self.assertEqual(loaded.canonical_phase, CanonicalPhase.UNKNOWN)
        self.assertEqual(
            canonical_phase(
                ModelRole.STRONG, InferencePhase.DRAFT, intent="retry"
            ),
            CanonicalPhase.STRONG_ACT,
        )



    def test_incomplete_interval_bounds_are_aggregatable(self) -> None:
        first = ResourceQuantity(
            value=None,
            status=QuantityStatus.INCOMPLETE,
            provenance="range-a",
            observed_subtotal=10,
            observed_lower=10,
            observed_upper=40,
        )
        second = ResourceQuantity(
            value=None,
            status=QuantityStatus.INCOMPLETE,
            provenance="range-b",
            observed_subtotal=5,
            observed_lower=5,
            observed_upper=9,
        )
        point = ResourceQuantity(
            value=3, status=QuantityStatus.ESTIMATED, provenance="point"
        )
        merged = merge_quantities([first, second, point], observed_name="flops")
        self.assertEqual(merged.status, QuantityStatus.INCOMPLETE)
        self.assertIsNone(merged.value)
        self.assertEqual(merged.observed_lower, 18)
        self.assertEqual(merged.observed_upper, 52)
        self.assertEqual(merged.observed_subtotal, 18)
        self.assertEqual(merged.observed_estimated, 3)
        rebuilt = ResourceQuantity.from_dict(merged.to_dict())
        self.assertEqual(rebuilt.observed_upper, 52)
        self.assertNotEqual(rebuilt.observed_estimated, rebuilt.observed_upper)


if __name__ == "__main__":
    unittest.main()
