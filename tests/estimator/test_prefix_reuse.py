"""Prefix-reuse model: the paper's three rules, and agreement with the script behind the tables."""

from __future__ import annotations

import math
import random
from collections import defaultdict

from specagent.protocol.prefix_reuse import (
    CANDIDATE_TOKENS_FALLBACK,
    ReuseRequest,
    cached_input_tokens,
    default_candidate_tokens,
)


def test_nested_requests_reuse_the_running_maximum() -> None:
    requests = [
        ReuseRequest("s", (0,), "nested", 1000),
        ReuseRequest("s", (1,), "nested", 1500),
        ReuseRequest("s", (2,), "nested", 1200),  # shorter: fully cached
        ReuseRequest("other", (0,), "nested", 800),  # no reuse across sessions
    ]
    assert cached_input_tokens(requests) == [0, 1000, 1200, 0]


def test_verify_treats_the_candidate_as_new_text() -> None:
    requests = [
        ReuseRequest("s", (0,), "verify", 5000, candidate_tokens=400),
        ReuseRequest("s", (1,), "verify", 6000, candidate_tokens=500),
    ]
    # First: nothing cached. Second: backbone 5500 vs 4600 seen, so 900 + 500 new.
    assert cached_input_tokens(requests) == [0, 6000 - 1400]


def test_repair_assumes_a_300_token_note_and_its_own_backbone() -> None:
    requests = [
        ReuseRequest("s", (0,), "verify", 5000, candidate_tokens=400),
        ReuseRequest("s", (1,), "repair", 5300),
        ReuseRequest("s", (2,), "repair", 5600),
    ]
    assert cached_input_tokens(requests) == [0, 0, 5600 - 300 - 300]


def test_unknown_candidates_use_the_median_or_the_fallback() -> None:
    requests = [
        ReuseRequest("s", (0,), "verify", 3000, candidate_tokens=200),
        ReuseRequest("s", (1,), "verify", 3000, candidate_tokens=600),
        ReuseRequest("s", (2,), "verify", 3000),
    ]
    assert default_candidate_tokens(requests) == 400
    assert cached_input_tokens(requests)[2] == 3000 - 400
    assert default_candidate_tokens([ReuseRequest("s", (0,), "nested", 10)]) == (
        CANDIDATE_TOKENS_FALLBACK
    )


# Reference: assign_radix and its candidate calibration from scripts/metrics/
# consistent_radix_table.py, the script that produced the paper's resource columns, reduced to
# the fields it reads. Items are (session, round, turn, index, span); session[0] is the episode.


def _reference_family(span):
    phase = span["phase"]
    return phase if phase in ("verify", "correct") else "nested"


def _reference_c_hat(items):
    outs = defaultdict(list)
    for sess, rnd, _turn, _idx, s in sorted(items, key=lambda t: (t[0], t[1], t[2], t[3])):
        if s["role"] == "weak":
            outs[(sess[0], rnd)].append(s["output_tokens"])
    proxy = sorted(outs[(sess[0], rnd)][-1] for sess, rnd, _turn, _idx, s in items
                   if s["phase"] == "verify" and outs.get((sess[0], rnd)))
    return int(round(_reference_pctile(proxy, 0.50) or 1500))


def _reference_pctile(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(f)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def _reference_assign(items, c_hat, n_hat=300):
    by = defaultdict(list)
    for it in items:
        by[it[0]].append(it)
    out = [None] * len(items)
    weak_out_by_round = defaultdict(dict)
    for sess, rnd, _turn, _idx, s in items:
        if s["role"] == "weak":
            weak_out_by_round[sess[0]][rnd] = int(s.get("output_tokens") or 0)
    for sess, seq in by.items():
        seq = sorted(seq, key=lambda t: (t[1], t[2], t[3]))
        nested_max = verify_b = correct_b = 0
        for _sess, rnd, _turn, idx, s in seq:
            inn = int(s.get("input_tokens") or 0)
            fam = _reference_family(s)
            if fam == "nested":
                unc = max(0, inn - nested_max)
                nested_max = max(nested_max, inn)
            elif fam == "verify":
                c = weak_out_by_round.get(sess[0], {}).get(rnd)
                if c is None:
                    c = c_hat
                c = max(0, min(inn, int(c)))
                backbone = max(0, inn - c)
                unc = c + max(0, backbone - verify_b)
                verify_b = max(verify_b, backbone)
            else:
                note = max(0, min(inn, int(n_hat)))
                backbone = max(0, inn - note)
                unc = note + max(0, backbone - correct_b)
                correct_b = max(correct_b, backbone)
            out[idx] = max(0, inn - int(round(unc)))
    return out


def _random_items(rng):
    items = []
    for episode in range(rng.randint(1, 6)):
        rounds = rng.randint(1, 8)
        for rnd in range(rounds):
            # Weak drafts in its own session; Strong verify / repair share the Strong one.
            skip_weak = rng.random() < 0.2  # some rounds have no recorded Weak output
            for turn in range(0 if skip_weak else rng.randint(1, 4)):
                items.append(((f"e{episode}", "4b", "weak"), rnd, turn,
                              {"role": "weak", "phase": "draft",
                               "input_tokens": rng.randint(500, 60000),
                               "output_tokens": rng.randint(0, 3000)}))
            items.append(((f"e{episode}", "397b", "strong"), rnd, 10,
                          {"role": "strong", "phase": "verify",
                           "input_tokens": rng.randint(0, 80000), "output_tokens": 50}))
            if rng.random() < 0.4:
                items.append(((f"e{episode}", "397b", "strong"), rnd, 11,
                              {"role": "strong", "phase": "correct",
                               "input_tokens": rng.randint(0, 80000), "output_tokens": 500}))
    # Logs are written in time order; the reference picks "the round's last Weak output" by
    # list position, which equals round order only for chronological input.
    return [(sess, rnd, turn, idx, s) for idx, (sess, rnd, turn, s) in enumerate(items)]


def test_matches_the_paper_table_script_on_random_sessions() -> None:
    rng = random.Random(0)
    for _ in range(300):
        items = _random_items(rng)
        last_weak_out = {}
        for sess, rnd, _turn, _idx, s in sorted(items, key=lambda t: (t[0], t[1], t[2], t[3])):
            if s["role"] == "weak":
                last_weak_out[(sess[0], rnd)] = s["output_tokens"]
        requests = [
            ReuseRequest(
                session=sess,
                order=(rnd, turn),
                family={"verify": "verify", "correct": "repair"}.get(s["phase"], "nested"),
                input_tokens=s["input_tokens"],
                candidate_tokens=(
                    last_weak_out.get((sess[0], rnd)) if s["phase"] == "verify" else None
                ),
            )
            for sess, rnd, turn, idx, s in items
        ]
        expected = _reference_assign(items, _reference_c_hat(items))
        assert cached_input_tokens(requests) == expected
