"""Design properties of benchmark_v3 and its controls; no network or real-model outcomes."""

import copy
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bazireason.baselines import run_case
from bazireason.calendar_engine import chart_at, compute_chart, jie_around
from bazireason.controls import BASELINES, corrupt_context
from bazireason.conventions import CYCLE, PILLARS
from bazireason.dataset_generator import dataset_bytes, read_jsonl
from bazireason.dataset_v3 import JIE_EVENTS, generate_benchmark_v3, pilot_case_ids_v3
from bazireason.evaluator import evaluate
from bazireason.prompts import build_messages, context_for_method

V3 = "data/synthetic/benchmark_v3.jsonl"


@pytest.fixture(scope="module")
def cases():
    return read_jsonl(V3)


def families(cases, category):
    groups = defaultdict(list)
    for case in cases:
        if case["category"] == category:
            groups[case["group_id"]].append(case)
    return groups


def test_v3_regenerates_byte_for_byte_and_is_versioned(cases):
    assert Path(V3).read_bytes() == dataset_bytes(generate_benchmark_v3())
    assert generate_benchmark_v3() != generate_benchmark_v3(7)  # seed reaches ordinary cases
    assert all(c["dataset_version"] == "benchmark_v3" for c in cases)
    statuses = {s: sum(c["expected"]["status"] == s for c in cases) for s in ("ok", "ambiguous")}
    assert len(cases) == 261 and statuses == {"ok": 225, "ambiguous": 21}
    keys = [json.dumps([c["input"], c["convention"]], sort_keys=True) for c in cases]
    assert len(keys) == len(set(keys))


def test_pilot_is_one_complete_family_per_category(cases):
    ids = json.loads(Path("data/synthetic/benchmark_v3_pilot_ids.json").read_text())
    assert ids == pilot_case_ids_v3(cases)
    chosen = [c for c in cases if c["case_id"] in ids]
    assert {c["category"] for c in chosen} == {c["category"] for c in cases}
    assert {c["expected"]["status"] for c in chosen} == {
        "ok",
        "ambiguous",
        "invalid",
        "unsupported",
    }
    for group in {c["group_id"] for c in chosen}:
        assert {c["case_id"] for c in cases if c["group_id"] == group} <= set(ids)


# ---- H1: distance ladders with matched controls --------------------------
def test_jie_ladders_hold_year_clock_and_task_fixed_while_distance_varies(cases):
    ladders = families(cases, "jie_ladder")
    assert len(ladders) == len(JIE_EVENTS) == 14
    for members in ladders.values():
        h1 = [c["design"]["h1"] for c in members]
        event = datetime.fromisoformat(h1[0]["event_utc8"]).replace(tzinfo=None)
        offsets = sorted(x["offset_seconds"] for x in h1)
        assert offsets == [-432000, -3600, -120, 120, 3600, 432000]
        relevant = h1[0]["relevant_pillars"]
        sides = defaultdict(set)
        for case, meta in zip(members, h1, strict=True):
            wall = datetime.fromisoformat(case["input"]["datetime"])
            assert wall - event == timedelta(seconds=meta["offset_seconds"])
            assert 3 <= wall.hour <= 20  # never inside the Zi window or a sect-dependent hour
            sides[meta["side"]].add(tuple(case["expected"]["pillars"][p] for p in relevant))
        # Same side of the Jie: same relevant pillars; opposite sides: different.
        assert len(sides["before"]) == len(sides["after"]) == 1
        assert sides["before"] != sides["after"]
        near = [c for c, m in zip(members, h1, strict=True) if m["distance"] == "near"]
        assert near[0]["expected"]["pillars"]["day"] == near[1]["expected"]["pillars"]["day"]
        assert near[0]["expected"]["pillars"]["hour"] == near[1]["expected"]["pillars"]["hour"]


def test_zi_ladders_isolate_the_day_boundary_rule(cases):
    for members in families(cases, "zi_ladder").values():
        assert len(members) == 12
        for sect in (1, 2):
            rows = {
                c["design"]["h1"]["clock"]: c["expected"]["pillars"]
                for c in members
                if c["convention"]["sect"] == sect
            }
            first = CYCLE.index(rows["21:30:00"]["day"])
            advanced = CYCLE[(first + 1) % 60]
            assert rows["22:59:59"]["day"] == CYCLE[first]
            for clock in ("23:00:00", "23:59:59"):
                assert rows[clock]["day"] == (advanced if sect == 1 else CYCLE[first])
                assert rows[clock]["hour"][1] == "子"
            assert rows["00:00:00"]["day"] == rows["01:30:00"]["day"] == advanced
            # The 23:00-advanced hour stem is the same for both sects.
            assert (
                rows["23:30:00" if "23:30:00" in rows else "23:00:00"]["hour"]
                == rows["00:00:00"]["hour"]
            )


# ---- H2: an information ladder without label leakage ----------------------
def _supplied(case, method):
    context = compute_chart(case["input"], case["convention"])
    return json.loads(build_messages(case, method, context)[1]["content"].split(":\n", 1)[1])


def test_fact_conditions_never_contain_pillars_status_or_case_metadata(cases):
    for case in cases:
        for method in ("tool_facts", "tool_intermediate"):
            facts = context_for_method(None, method, case)
            text = json.dumps(facts, ensure_ascii=False)
            assert not any(pair in text for pair in CYCLE), (case["case_id"], method)
            assert "status" not in facts and "reason_codes" not in facts
            mutated = copy.deepcopy(case)
            mutated["expected"] = {"SENTINEL": 1}
            mutated["design"] = {"SENTINEL": 2}
            mutated["convention"] = {"sect": 1}
            assert build_messages(mutated, method) == build_messages(
                {**case, "convention": {"sect": 1}}, method
            )


def test_timestamp_facts_are_a_subset_of_intermediate_facts(cases):
    for case in cases:
        small = context_for_method(None, "tool_facts", case)
        large = context_for_method(None, "tool_intermediate", case)
        assert len(small["candidates"]) == len(large["candidates"])
        for a, b in zip(small["candidates"], large["candidates"], strict=True):
            assert a.items() <= b.items()


def test_normalization_facts_are_informative_beyond_shanghai(cases):
    changed = [
        c
        for c in cases
        if c["expected"]["status"] == "ok"
        and any(
            entry["utc8_start"][:19] != c["input"]["datetime"][:19]
            for entry in context_for_method(None, "tool_facts", c)["candidates"]
        )
    ]
    # Every non-Shanghai answerable case needs a genuine conversion.
    foreign = [
        c
        for c in cases
        if c["expected"]["status"] == "ok" and c["input"]["timezone"] != "Asia/Shanghai"
    ]
    assert {c["case_id"] for c in foreign} <= {c["case_id"] for c in changed}
    assert len(foreign) == 26


def test_intermediate_jie_facts_match_the_engine_boundaries(cases):
    for case in cases:
        if case["category"] != "jie_ladder":
            continue
        facts = context_for_method(None, "tool_intermediate", case)["candidates"][0]
        event = case["design"]["h1"]["event_utc8"]
        side = case["design"]["h1"]["side"]
        if case["design"]["h1"]["distance"] != "far":
            key = "previous_jie" if side == "after" else "next_jie"
            assert facts[key]["utc8"] == event


def test_every_corruption_is_plausible_and_changes_only_its_fields(cases):
    kinds = defaultdict(int)
    for case in cases:
        context = compute_chart(case["input"], case["convention"])
        supplied, corruption = corrupt_context(case, context)
        status = case["expected"]["status"]
        if status in ("invalid", "unsupported"):
            assert corruption is None and supplied == {k: context[k] for k in supplied}
            continue
        if status == "ambiguous":
            if case["input"].get("datetime") is None:  # nothing to default from
                assert corruption is None
                continue
            assert corruption["type"] == "default_guess" and supplied["status"] == "ok"
            kinds["default_guess"] += 1
            continue
        kinds[corruption["type"]] += 1
        truth = case["expected"]["pillars"]
        changed = {p for p in PILLARS if supplied["pillars"][p] != truth[p]}
        assert changed == set(corruption["fields"]), case["case_id"]
        assert all(supplied["pillars"][p] in CYCLE for p in PILLARS)
        if corruption["type"] == "jie_side":
            ref = datetime.fromisoformat(context["normalized_candidates"][0])
            (prev_t, _), (next_t, _) = jie_around(ref)
            other = prev_t - timedelta(seconds=1) if ref - prev_t <= next_t - ref else next_t
            expected = chart_at(other, context["conventions"]["sect"] or 2)
            assert all(expected[p] == v for p, v in corruption["fields"].items())
    assert set(kinds) == {"jie_side", "day_rollover", "hour_stem", "default_guess"}


def test_deterministic_copy_is_a_ceiling_not_evidence(cases):
    class Forbidden:
        def complete(self, *args, **kwargs):
            pytest.fail("A deterministic control invoked a model")

    records = [run_case(c, Forbidden(), "copy_context") for c in cases]
    overall = evaluate(cases, records, methods=["copy_context"], bootstrap_samples=0)["methods"][
        "copy_context"
    ]["overall"]
    assert overall["exact_accuracy"] == 1 and overall["n_labeled"] == 225
    assert overall["decision_accuracy"] == 1 and overall["false_refusal_rate"] == 0


# ---- H3: matched families that defeat surface rules -----------------------
def test_knowledge_dependent_families_pair_identical_surface_forms(cases):
    for case_list in (families(cases, "h3_sect"), families(cases, "dst_fold")):
        for members in case_list.values():
            forms = defaultdict(set)
            for c in members:
                forms[c["design"]["h3"]["surface_form"]].add(c["expected"]["status"])
            unspecified = forms.get("no_sect") or forms.get("no_fold")
            assert "ok" in unspecified  # the same omission is often harmless...
    statuses = {
        group: sorted(c["expected"]["status"] for c in members if "fold" not in c["input"])
        for group, members in families(cases, "dst_fold").items()
    }
    # ...so whether a repeated wall time needs a fold depends on the zone's offsets.
    assert sum(s == ["ambiguous", "ok"] for s in statuses.values()) == 3
    assert sum(s == ["ok", "ok"] for s in statuses.values()) == 2


def test_trivial_strategies_fail_visibly(cases):
    preds = [
        {"case_id": c["case_id"], "method": m, "trial": 0, "response": BASELINES[m](c)}
        for c in cases
        for m in BASELINES
    ]
    result = evaluate(cases, preds, methods=list(BASELINES), bootstrap_samples=0)["methods"]
    surface = result["heuristic_surface"]["h3_matched"]
    assert surface["matched_families"]["surface"]["all_decisions_correct_rate"] == 1
    assert surface["matched_families"]["knowledge_dependent"]["all_decisions_correct_rate"] == 0
    assert surface["by_dependence"]["knowledge_dependent"]["false_refusal_rate"] > 0.3
    assert result["always_abstain"]["overall"]["false_refusal_rate"] == 1
    assert result["always_abstain"]["overall"]["scorable_coverage"] == 0
    assert result["always_answer"]["overall"]["ambiguous_definitive_rate"] == 1
    for method in BASELINES:
        assert result[method]["overall"]["exact_accuracy"] == 0  # no calendar computation
        families_ = result[method]["h3_matched"]["matched_families"]
        assert families_["knowledge_dependent"]["n_family_trials"] == 16


def test_explicit_policy_names_some_types_and_holds_out_others():
    from bazireason.prompts import EXPLICIT_POLICY

    assert "sect" in EXPLICIT_POLICY and "fold" in EXPLICIT_POLICY
    assert "second" not in EXPLICIT_POLICY and "minute" not in EXPLICIT_POLICY


def test_h3_primary_contrast_is_family_level_and_paired(cases):
    def answer_all(c):
        return BASELINES["always_answer"](c)

    explicit = [
        {"case_id": c["case_id"], "method": "direct", "trial": 0, "response": c["expected"]}
        for c in cases
    ]
    minimal = [
        {"case_id": c["case_id"], "method": "direct_minimal", "trial": 0, "response": answer_all(c)}
        for c in cases
    ]
    result = evaluate(cases, explicit + minimal, methods=["direct", "direct_minimal"])
    family = result["comparisons"]["direct_minus_direct_minimal"]["h3_family_all_decisions_correct"]
    kd = family["knowledge_dependent"]
    assert kd["n"] == 16 and kd["mean_difference"] == 1
    assert (kd["discordant_treatment_only"], kd["discordant_comparator_only"]) == (16, 0)
    assert kd["mcnemar_exact_p_descriptive"] < 0.001
    assert family["surface"]["n"] == 15
