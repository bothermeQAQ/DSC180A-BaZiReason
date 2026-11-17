"""Phase 6 tests: splits and hashes, leakage, the seal, the no-answer levels, scorers, derivations.

No model is run. The sealed rows are only hashed and checked for leakage; they are not printed.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/phase5"))
sys.path.insert(0, str(ROOT / "scripts/phase6"))
SPLITS = ("dev6", "valid6", "sealed6")
PARTS = ("rules", "indep", "blind", "combo", "cf")


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def items():
    import run6

    return {(s, p): run6.load_items(s, p) for s in SPLITS for p in PARTS}


def test_phase6_splits_match_their_manifests():
    for split in SPLITS:
        manifest = json.loads((ROOT / "data/phase6" / split / "manifest.json").read_text())
        for name, info in manifest["files"].items():
            digest = hashlib.sha256((ROOT / "data/phase6" / split / name).read_bytes()).hexdigest()
            assert digest == info["sha256"], (split, name)


def test_blind_items_rebuild_from_the_writer_logs():
    import gen_blind6

    for split in (*SPLITS, "train6", "train6b"):
        if not (ROOT / f"data/phase6/blind/{split}_attempts.jsonl").exists():
            continue
        log = rows(ROOT / f"data/phase6/blind/{split}_attempts.jsonl")
        payload = "".join(
            json.dumps(i, ensure_ascii=False) + "\n" for i in gen_blind6.build(split, log)
        )
        assert payload.encode() == (ROOT / f"data/phase6/blind/{split}.jsonl").read_bytes(), split


def test_year_blocks_are_disjoint_and_eval_charts_are_new():
    from bazireason import splits6 as S

    years = S.phase6_years()
    assert not years["dev6"] & years["valid6"] and not years["valid6"] & years["sealed6"]
    assert not (years["dev6"] | years["valid6"] | years["sealed6"]) & years["train"]
    earlier = S.earlier_eval_charts()
    for split in ("valid6", "sealed6"):
        for part in ("rules", "combo", "cf"):
            for r in rows(ROOT / f"data/phase6/{split}/{part}.jsonl"):
                assert S.chart_key(r["meta"]["pillars"]) not in earlier, (split, part)


def test_training_sets_hold_no_evaluation_chart_wording_or_check_request(items):
    from bazireason import indep_rules as I
    from bazireason import ladder as L
    from bazireason import splits6 as S

    eval_charts = {S.chart_key(it["meta"]["pillars"]) for v in items.values() for it in v}
    # Blind and combination questions are held-out wordings. Counterfactual pairs of composite
    # procedures reuse a training template on purpose (they test a one-character change, not wording).
    eval_questions = {
        it["meta"]["question"] for (s, p), v in items.items() if p in ("blind", "combo") for it in v
    }
    eval_questions |= {
        it["meta"]["question"]
        for (s, p), v in items.items()
        if p == "cf"
        for it in v
        if it["family"] == "fact"
    }
    phrases = [
        t.split("{")[0] for ts in I.TEMPLATES.values() for t in ts if len(t.split("{")[0]) > 6
    ]
    phrases += [
        w.split("{")[0] for ws in S.COMBO_WORDS.values() for w in ws if len(w.split("{")[0]) > 6
    ]
    phrases += [w for w in S.CF_WORDS.values() if "{" not in w] + [
        c.split("{")[0][:20] for c in L.CHECKS.values()
    ]
    for folder in sorted((ROOT / "data/phase6/train").iterdir()):
        for path in (folder / "train.jsonl", folder / "valid.jsonl"):
            if not path.exists():  # train.jsonl is not in Git; build_train6.py --check rebuilds it
                continue
            data = rows(path)
            users = "\n".join(
                m["content"] for r in data for m in r["messages"] if m["role"] == "user"
            )
            for phrase in phrases:
                assert phrase not in users, (folder.name, phrase)
            assert not any(q in users for q in eval_questions), folder.name
            for r in data:
                pillars = (r.get("meta") or {}).get("pillars")
                if isinstance(pillars, dict) and len(pillars) == 4:
                    assert S.chart_key(pillars) not in eval_charts, (folder.name, r["id"])


def test_the_sealed_split_is_closed_without_a_selection(tmp_path, monkeypatch):
    import run6

    monkeypatch.setattr(run6, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run6.py", "--split", "sealed6", "--labels", "p4a1"])
    with pytest.raises(SystemExit):
        run6.main()
    assert not (tmp_path / "sealed_access.jsonl").exists()


def test_levels_0_to_2_carry_no_answer(items):
    from bazireason import ladder as L

    for (split, _part), its in items.items():
        if split == "sealed6":
            continue
        for it in its:
            meta = it["meta"]
            assert (
                L.leaks(L.raw_sheet(meta["pillars"], meta.get("gender"), meta["intent"])) == []
            ), it["id"]
            if meta["intent"].startswith("fact:hidden"):
                assert (
                    "藏"
                    not in L.raw_sheet(meta["pillars"], meta.get("gender"), meta["intent"]).split(
                        "\n", 1
                    )[1]
                )
    a = {"year": "乙亥", "month": "甲申", "day": "丙戌", "hour": "癸巳"}
    b = {"year": "庚午", "month": "辛巳", "day": "壬午", "hour": "甲辰"}
    for intent in ("strength", "pattern", "flow:2031", "fact:xun_kong", "combo:revealed:day"):
        assert L.rule_text(intent) == L.rule_text(intent.replace("2031", "2040")), (
            intent
        )  # chart-free by construction
        assert all(x not in L.rule_text(intent) for x in (a["day"], b["day"]))


def test_scorers_accept_the_reference_and_reject_a_wrong_conclusion(items):
    from bazireason import ladder as L

    for (split, _part), its in items.items():
        if split != "dev6":
            continue
        for it in its:
            ref = it["meta"]["expected"]
            assert L.score(it, "结论JSON: " + json.dumps(ref, ensure_ascii=False))["correct"], it[
                "id"
            ]
            wrong = L.wrong_reference(it, random.Random(it["id"]))
            if wrong is not None and wrong != ref:
                assert not L.score(it, "结论JSON: " + json.dumps(wrong, ensure_ascii=False))[
                    "correct"
                ], it["id"]


def test_small_step_derivations_match_the_rule_code_and_pass_the_claim_checker():
    from bazireason import splits6 as S
    from bazireason import steps6 as S6
    from bazireason.grounded import claim_failures
    from bazireason.p2_prompts import extract_final_json

    rng = random.Random(20261116)
    intents = ("fact:hidden:month", "fact:ten_god_stem:hour", "fact:ten_god_branch:year", "fact:life_stage:day",
               "fact:xun_kong", "fact:luck_direction", "fact:element_count", "fact:missing_element", "strength", "pattern",
               "interact", "flow:2037", "domain:marriage:2031", "domain:children:2027", "gods", "overview",
               "yongshen:扶抑", "yongshen:调候", "yongshen:格局", "ambig:yongshen_open", "star:华盖", "prediction")  # fmt: skip
    for _ in range(60):
        pillars, gender = S.random_pillars(rng), rng.choice(("male", "female"))
        for intent in intents:
            text, reference = S6.derive(intent, pillars, gender)
            assert extract_final_json(text) == reference, (intent, pillars)
            assert claim_failures(text, pillars, gender) == [], (intent, pillars)


def test_combinations_and_counterfactual_pairs_are_what_they_claim():
    from bazireason import rules as R
    from bazireason import splits6 as S

    for r in rows(ROOT / "data/phase6/dev6/combo.jsonl"):
        assert r["meta"]["intent"].startswith("combo:") and r["family"] == "combo"
    pairs = {}
    for r in rows(ROOT / "data/phase6/dev6/cf.jsonl"):
        pairs.setdefault(r["meta"]["pair"], []).append(r)
    for pair in pairs.values():
        a, b = pair
        diff = sum(
            x != y
            for p in S.PILLARS
            for x, y in zip(a["meta"]["pillars"][p], b["meta"]["pillars"][p], strict=True)
        )
        assert (
            diff == 1
            and a["meta"]["expected"] != b["meta"]["expected"]
            and a["meta"]["question"] == b["meta"]["question"]
        )
    assert S.combo_reference(
        "flow_stage", {"year": "甲子", "month": "丙寅", "day": "甲子", "hour": "甲子"}, 2026
    ) == R.life_stage("甲", "午")


# ---- round 2 (pre-registration, section 10) -----------------------------------------------------------


def test_only_the_main_system_prompt_is_swapped_and_only_for_short_prompt_models(monkeypatch):
    from bazireason import prompts6 as P6
    from bazireason.p2_prompts import P2_SYSTEM

    conv = [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": "q"}]
    other = [{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": "q"}]
    for label in ("base", "p2", "p4a1", "p5g1", "p6u0", "p6s1", "p6u1hi", "panel-p6u0-500"):
        assert P6.system_for(label) == P2_SYSTEM
        assert P6.apply(label, [conv]) == [conv]
    short = P6.swap(conv, P6.SHORT_SYSTEM)
    assert short[0]["content"] == P6.SHORT_SYSTEM and short[1] is conv[1]
    assert conv[0]["content"] == P2_SYSTEM  # the input is not modified
    assert P6.swap(other, P6.SHORT_SYSTEM) == other  # task prompts define another format
    assert len(P6.SHORT_SYSTEM) < len(P2_SYSTEM) / 5 and "结论JSON:" in P6.SHORT_SYSTEM
    parents = {"u2": {"system": "short"}, "u3": {"system": "short", "parent": "u2"}}
    monkeypatch.setattr(P6, "manifest", lambda recipe: parents.get(recipe, {}))
    assert P6.system_for("p6u3") == P6.SHORT_SYSTEM == P6.system_for("panel-p6u2-1000")
    assert P6.apply("p6u2", [conv, other]) == [short, other]
    assert P6.lineage("u3") == ["u3", "u2"] and P6.lineage("u1") == ["u1"]


def test_round2_builds_swap_the_prompt_block_sample_b_and_follow_the_weights():
    import build_train6 as B

    from bazireason import prompts6 as P6
    from bazireason import train_data6 as T6
    from bazireason.p2_prompts import P2_SYSTEM

    exclude = B.exclusions()
    sample = T6.fit_sample_b(B.FIT_B_SEED, 3, exclude)
    assert [r["meta"]["head"] for r in sample] == [h for h in T6.WEIGHTS for _ in range(3)]
    assert all(T6.head_of(r["meta"]["intent"]) == r["meta"]["head"] for r in sample)
    assert all(r["messages"][0]["content"] == P2_SYSTEM for r in sample)
    assert not {T6.chart_of(r) for r in sample} & exclude
    paraphrases = [
        json.loads(x)
        for x in (ROOT / "data/phase6/blind/train6.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    block = frozenset(T6.chart_of(r) for r in sample + paraphrases[:5])
    kwargs = {"exclude": exclude, "paraphrases": paraphrases, "drop": ("selfcheck",)}
    long_rows = T6.build("s3", 120_000, 7, **kwargs)
    short_rows = T6.build("s3", 120_000, 7, system=P6.SHORT_SYSTEM, block=block, **kwargs)
    # A round-2 build leaves the default path as it was: the earlier sets still rebuild bit for bit.
    assert T6.digest(long_rows) == T6.digest(T6.build("s3", 120_000, 7, **kwargs))
    systems = {r["messages"][0]["content"] for r in short_rows}
    assert P2_SYSTEM not in systems and P6.SHORT_SYSTEM in systems and len(systems) > 1
    assert all(r["messages"][0]["content"] != P6.SHORT_SYSTEM for r in long_rows)
    assert len(short_rows) > 1.2 * len(long_rows)  # the same token budget holds more rows
    assert not any(T6.chart_of(r) in block for r in short_rows)
    weights = dict.fromkeys(T6.WEIGHTS, 0.05) | {"flow": 0.9}
    options = {"system": P6.SHORT_SYSTEM, "weights": weights, "block": block}
    weighted = T6.build("s3", 120_000, 8, **options, **kwargs)
    assert T6.digest(weighted) == T6.digest(T6.build("s3", 120_000, 8, **options, **kwargs))
    procedures = [r for r in weighted if r["family"] == "procedure"]
    heads = [T6.head_of(r["meta"]["intent"]) for r in procedures]
    assert heads.count("flow") > len(heads) / 2
    # Derivation version 2 is used only when asked for, and the default is restored afterwards.
    second = T6.build("s3", 120_000, 8, steps=2, **options, **kwargs)
    assert any("相克次序" in r["messages"][-1]["content"] for r in second)
    assert not any("相克次序" in r["messages"][-1]["content"] for r in weighted)
    assert T6.digest(long_rows) == T6.digest(T6.build("s3", 120_000, 7, **kwargs))
    assert {r["meta"]["source"] for r in procedures} == {"template", "paraphrase"}
    assert not any(T6.chart_of(r) in block for r in weighted)


def test_error_weights_score_only_charts_the_parent_never_trained_on(tmp_path, monkeypatch):
    import build_train6 as B

    from bazireason import train_data6 as T6

    heads = list(T6.WEIGHTS)
    seen, fresh = ("甲子", "乙丑", "丙寅", "丁卯"), ("戊辰", "己巳", "庚午", "辛未")

    def reply(head, chart, ok):
        pillars = dict(zip(("year", "month", "day", "hour"), chart, strict=True))
        return {"row": {"meta": {"head": head, "pillars": pillars}}, "reply": "", "correct": ok}

    rows = [reply(heads[0], fresh, False)] * 8 + [reply(heads[0], fresh, True)] * 2
    rows += [reply(heads[1], fresh, True)] * 100 + [reply(heads[1], seen, False)] * 5
    path = tmp_path / "replies.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    monkeypatch.setattr(B, "training_charts", lambda recipe: {"".join(seen)})
    out = B.error_weights(path, "u2")
    assert out["left_out"] == 5
    assert out["scored_wrong"][heads[0]] == [10, 8] and out["scored_wrong"][heads[1]] == [100, 0]
    assert out["weights"][heads[0]] == 0.75  # (8 + 1) / (10 + 2)
    assert out["weights"][heads[1]] == B.ERROR_FLOOR  # (0 + 1) / (100 + 2) is under the floor
    assert set(out["weights"]) == set(T6.WEIGHTS)


# Level 4 asks about years of the chart's life (12.11) and weights calendar rows by family (12.13)
L4 = "--years life --calendar-weight-by family"


def test_round2_loop_structure_and_selection_follow_the_preregistered_rules(
    tmp_path, monkeypatch, capsys
):
    import decide6 as D

    right = {}  # (split, label) -> number of right answers out of 100

    def records(split, label, level="L0"):
        k = right.get((split, label))
        if k is None:
            return None
        return [{"id": i, "lab": label, "score": {"correct": i < k}} for i in range(100)]

    size = {"p6u1hi": 33_000_000}
    log = []
    parents = {"u3": {"parent": "u2"}, "u3v": {"parent": "u2"}, "u4": {"parent": "u3v"}}
    monkeypatch.setattr(D.M, "rule_records", records)
    monkeypatch.setattr(D, "parameters", lambda lab: size.get(lab, 14_680_064))
    monkeypatch.setattr(D, "record", log.append)
    monkeypatch.setattr(D, "recorded", lambda: log)
    monkeypatch.setattr(D.P6, "manifest", lambda recipe: parents.get(recipe, {}))
    monkeypatch.setattr(
        D, "paired", lambda split, a, b, **kw: {"difference": 0.0, "interval95": gap}
    )

    def said():
        return capsys.readouterr().out.strip()

    # The first level is a fork from the best of U1, U1-hi and U2 on dev6.
    right.update({("dev6", "p6u1"): 60, ("dev6", "p6u2"): 62})
    gap = [0.01, 0.05]
    D.nxt()
    assert said() == "u3 u2\nu3v u2 --steps 2"
    right[("dev6", "p6u1hi")] = 62  # a tie goes to the smaller adapter
    D.nxt()
    assert said().splitlines()[0] == "u3 u2"
    right[("dev6", "p6u1")] = 62  # then to the earlier run
    D.nxt()
    assert said().splitlines()[0] == "u3 u1"
    right[("dev6", "p6u1")] = 60
    # A level with a missing arm (a restart, or a failed arm) asks for that arm once more, with
    # the same parent; if it is still missing after that, the level stands as it is.
    right[("dev6", "p6u3")] = 64
    D.nxt()
    assert said() == "u3v u2 --steps 2"
    assert (
        log[-1]["decision"] == "complete" and log[-1]["level"] == 3 and log[-1]["runs"] == ["u3v"]
    )
    D.nxt()
    assert said() == f"u4 u3 {L4}\nu4c u3 --compose-share 0.10 {L4}"
    del right[("dev6", "p6u3")]
    right[("dev6", "p6u3v")] = 65
    D.nxt()
    assert said() == "u3 u2"  # the same for the other arm
    # The chain continues from the better arm of the fork (a tie goes to U3).
    right.update({("dev6", "p6u3"): 64, ("dev6", "p6u3v"): 65})
    D.nxt()
    # the second level has two arms too, and both ask about years of the chart's life (12.11)
    assert said() == f"u4 u3v {L4}\nu4c u3v --compose-share 0.10 {L4}"
    right[("dev6", "p6u3v")] = 64
    D.nxt()
    assert said().splitlines()[0] == f"u4 u3 {L4}"
    right[("dev6", "p6u3v")] = 65
    # It goes on from the latest level, and ends after two levels in a row without a new best.
    right[("dev6", "p6u4")] = 64  # not above 65: first level without a new best
    D.nxt()
    assert said() == f"u4c u3v --compose-share 0.10 {L4}"  # the missing arm, once
    D.nxt()
    assert said() == "u5 u4"
    right[("dev6", "p6u5")] = 65  # equal is not a new best: second in a row
    D.nxt()
    assert said() == "stop"
    right[("dev6", "p6u5")] = 66  # a new best resets the count
    D.nxt()
    assert said() == "u6 u5"
    for k, lab in enumerate(("p6u6", "p6u7", "p6u8", "p6u9", "p6u10")):
        right[("dev6", lab)] = 67 + k
    D.nxt()
    assert said() == "stop"  # at most eight levels
    for lab in ("p6u5", "p6u6", "p6u7", "p6u8", "p6u9", "p6u10"):
        del right[("dev6", lab)]
    # Structure: U1-hi's only when it beats U1 with a lower bound above 0.
    shape = {"layers": 36, "rank": 16, "grad_checkpoint": True}
    monkeypatch.setattr(D.build_train6, "structure_of", lambda recipe: shape)
    D.structure()
    assert said() == "--layers 36 --rank 16 --grad-checkpoint"
    gap = [-0.01, 0.05]
    D.structure()
    assert said() == ""
    # Selection: every round-2 run is a candidate; the calendar gate, then the highest valid6
    # accuracy, then the smaller adapter, then the earlier run.
    monkeypatch.setattr(D, "ROOT", tmp_path)
    (tmp_path / "screening.json").write_text(json.dumps({"best_s_run_on_dev6": "p6s2"}))
    valid = {"p6s2": 55, "p6u1": 63, "p6u1hi": 66, "p6u2": 66, "p6u3": 70, "p6u4": 66}
    right.update({("valid6", lab): k for lab, k in valid.items()})
    lower = {"p6u3": -0.09}  # U3 fails the calendar gate (lower bound under -7 points)
    monkeypatch.setattr(D.M, "load", lambda split, lab, part, level="L0": records("valid6", lab) or records("dev6", "p6u1"))  # fmt: skip
    monkeypatch.setattr(D.M, "scores", lambda recs: recs)
    monkeypatch.setattr(D, "paired_difference", lambda a, b: {"difference": 0.0, "interval95": [lower.get(b[0]["lab"], -0.03), 0.03]})  # fmt: skip
    D.select()
    chosen = json.loads((tmp_path / "selection.json").read_text())
    assert list(chosen["candidates"]) == ["p6s2", "p6u1", "p6u1hi", "p6u2", "p6u3", "p6u4"]
    assert chosen["passing_calendar_gate"] == ["p6s2", "p6u1", "p6u1hi", "p6u2", "p6u4"]
    assert chosen["selected"] == "p6u2"  # 66 three times: smaller than U1-hi, earlier than U4
    assert log and all(e["decision"] in ("next", "complete", "structure") for e in log)


def test_derivation_version_2_keeps_every_answer_and_passes_the_claim_checker():
    from bazireason import deep_rules as D
    from bazireason import rules as R
    from bazireason import splits6 as S
    from bazireason import steps6 as S6
    from bazireason.conventions import BRANCHES, STEMS
    from bazireason.grounded import claim_failures
    from bazireason.p2_prompts import extract_final_json

    for day in STEMS:  # the position of the month's element gives the rule code's state
        for branch in BRANCHES:
            position = D.element_class(day, R.BRANCH_ELEMENT[branch])
            assert S6.STATE_OF_POSITION[position] == R.seasonal_state(R.STEM_ELEMENT[day], branch)
    rng = random.Random(20261117)
    intents = ("strength", "flow:2037", "domain:marriage:2031", "domain:wealth:2029", "overview",
               "yongshen:扶抑", "interact", "pattern", "gods", "fact:xun_kong", "fact:life_stage:month")  # fmt: skip
    changed = set()
    for _ in range(40):
        pillars, gender = S.random_pillars(rng), rng.choice(("male", "female"))
        for intent in intents:
            first, reference = S6.derive(intent, pillars, gender)
            S6.set_steps(2)
            try:
                second, same = S6.derive(intent, pillars, gender)
            finally:
                S6.set_steps(1)
            assert same == reference == extract_final_json(second), (intent, pillars)
            assert claim_failures(second, pillars, gender) == [], (intent, pillars)
            if first != second:
                changed.add(":".join(intent.split(":")[:2]) if intent.startswith("fact") else intent.split(":")[0])  # fmt: skip
    # gods and 旬空 are the same in both versions; the others changed (10.5 and 12.6)
    assert changed == {"strength", "flow", "domain", "overview", "yongshen", "interact", "pattern",
                       "fact:life_stage"}  # fmt: skip
    assert S6.steps_version() == 1


def test_clean_fit_leaves_out_questions_whose_chart_was_trained_on(tmp_path, monkeypatch):
    import fit_clean as F

    def reply(chart, intent, ok):
        pillars = dict(zip(("year", "month", "day", "hour"), chart, strict=True))
        return {"row": {"meta": {"intent": intent, "pillars": pillars}}, "correct": ok}

    seen, fresh = ("甲子", "乙丑", "丙寅", "丁卯"), ("戊辰", "己巳", "庚午", "辛未")
    rows = [reply(seen, "pattern", True)] * 6 + [reply(fresh, "pattern", True)] * 3
    rows += [reply(fresh, "fact:xun_kong:day", False)]
    (tmp_path / "own_replies_p6u1.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F.build_train6, "training_charts", lambda recipe: {"".join(seen)})
    out = F.clean("p6u1")
    assert (out["n"], out["correct"], out["share"]) == (4, 3, 0.75)
    assert out["left_out_chart_in_training"] == 6
    assert out["by_type"] == {"fact:xun_kong": [0, 1], "pattern": [3, 3]}


# ---- the larger-base pilot (pre-registration, section 11) ---------------------------------------------


def test_the_pilot_config_is_s1s_with_only_the_base_and_batch_changed():
    import p8_config as P

    def keys(text):  # "key: value" lines (top level and nested); comments count as "#"
        out = {}
        for line in text.splitlines():
            if line.strip().startswith("#"):
                out.setdefault("#", []).append(line)
            elif ":" in line:
                key, _, value = line.strip().partition(":")
                out[key] = value.split("#")[0].strip()
        return out

    s1 = keys(P.SOURCE.read_text())
    for batch, checkpoint in ((2, False), (1, False), (1, True)):
        pilot = keys(P.build(batch, checkpoint))
        changed = {k for k in s1.keys() | pilot.keys() if s1.get(k) != pilot.get(k)}
        expected = {
            "#",
            "run_name",
            "model",
            "repo",
            "revision",
            "quantization",
            "adapter_dir",
            "log_dir",
        }
        if batch == 1:
            expected |= {"batch_size", "grad_accumulation_steps"}
        if checkpoint:
            expected.add("grad_checkpoint")
        assert changed == expected, (batch, checkpoint, changed)
        assert pilot["model"] == "models/qwen3-8b-mlx-4bit"
        assert int(pilot["batch_size"]) * int(pilot["grad_accumulation_steps"]) == 4  # S1's updates
        assert pilot["train"] == s1["train"] and pilot["seed"] == s1["seed"]
        assert pilot["enable_thinking"] == "false" and pilot["mask_prompt"] == "true"


def test_an_adapter_is_loaded_on_the_base_it_was_trained_on(tmp_path, monkeypatch):
    import types

    import run6

    default = "models/qwen3-4b-instruct-2507-mlx-4bit"
    for name, model, layers in (("bazireason-qwen3-8b-lora-p6-s1", "models/qwen3-8b-mlx-4bit", 16),
                                ("bazireason-qwen3-4b-lora-p6-u1-hi", default, 36)):  # fmt: skip
        (tmp_path / name).mkdir()
        config = {"model": model, "num_layers": layers, "lora_parameters": {"rank": 16}}
        (tmp_path / name / "adapter_config.json").write_text(json.dumps(config))
    loaded = []

    class Stub:
        def __init__(self, model_path, adapter_path=None, label=None, enable_thinking=False):
            self.adapter_path = str(adapter_path)
            loaded.append((model_path, Path(adapter_path).name, enable_thinking))

    # The real modules need mlx; the test environment has none, so stand-ins are used.
    bank_module = types.SimpleNamespace(
        ADAPTER_ROOT=tmp_path, DEFAULT_MODEL=default, AdapterBank=None
    )
    monkeypatch.setitem(sys.modules, "bazireason.adapter_bank", bank_module)
    monkeypatch.setitem(
        sys.modules, "bazireason.mlx_backend", types.SimpleNamespace(MLXChatModel=Stub)
    )
    bank = run6.Bank()
    bank.get("p8s1")
    bank.get("p8s1")  # reused, not loaded twice
    bank.get("p6u1hi")
    assert loaded == [
        ("models/qwen3-8b-mlx-4bit", "bazireason-qwen3-8b-lora-p6-s1", False),
        (default, "bazireason-qwen3-4b-lora-p6-u1-hi", False),
    ]


# ---- the recovery (pre-registration, section 12) ------------------------------------------------------


def test_training_writer_questions_on_evaluation_charts_are_known_and_left_out():
    import build_train6 as B

    from bazireason import splits6 as S

    exclude = B.exclusions()
    on_eval = {}
    for path in sorted((ROOT / "data/phase6/blind").glob("train6*.jsonl")):
        if path.name.endswith("_attempts.jsonl"):
            continue
        items = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        on_eval[path.stem] = sum(S.chart_key(it["meta"]["pillars"]) in exclude for it in items)
    # train6 feeds S2 and S3 and has none, so the filter of section 12.2 leaves them bit for bit.
    assert on_eval["train6"] == 0
    assert all(n <= 2 for n in on_eval.values()), on_eval  # a rare recurrence of four pillars
    source = (ROOT / "scripts/phase6/build_train6.py").read_text(encoding="utf-8")
    assert 'S.chart_key(p["meta"]["pillars"]) not in exclude' in source


def test_u2_keeps_the_short_prompt_unless_the_twin_is_clearly_worse(monkeypatch, capsys):
    import decide6 as D

    log = []
    monkeypatch.setattr(D, "record", log.append)
    bounds = {}

    def paired(split, a, b, level="L0", parts=None):
        assert (split, a, b) == ("dev6", "p6s2", "p6s2s")
        key = "calendar" if parts == ("calendar",) else "rules"
        return {"difference": 0.0, "interval95": bounds[key]}

    monkeypatch.setattr(D, "paired", paired)
    for rules, calendar, expected in (
        ([-0.02, 0.03], [-0.05, 0.04], "--system short"),
        ([-0.06, -0.01], [-0.05, 0.04], "--system long"),  # clearly worse on the rule items
        ([0.01, 0.06], [-0.12, -0.02], "--system long"),  # clearly worse on the calendar rows
    ):
        bounds.update(rules=rules, calendar=calendar)
        D.prompt()
        assert capsys.readouterr().out.strip() == expected
    assert [e["decision"] for e in log] == ["prompt"] * 3


def test_no_queue_script_can_report_success_after_a_failed_step():
    """Every GPU step of a queue is followed by a stop (the U1 queue of 15 November was not)."""
    for name in ("queue_s2s.sh", "queue_main.sh", "queue_sealed6.sh", "queue_u1hi.sh"):
        text = (ROOT / "scripts/phase6" / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            step = line.strip()
            if "run $PY" not in step or step.startswith(("#", "P=", "if run", "elif run")):
                continue
            guarded = any(k in step for k in ("|| fail", "|| return 1", "|| exit 1"))
            diagnostic = (
                "--sample b" in step and "own_replies_${L}_b" in step
            )  # the last, optional loop
            # Calendar weights are optional by 12.5: without the sample the build goes on.
            optional = "--sample c" in step and "no calendar fit sample" in step
            assert guarded or diagnostic or optional, (name, step[:90])


def test_calendar_weights_follow_the_measured_errors(tmp_path):
    import build_train6 as B

    from bazireason import train_data6 as T6

    def reply(family, status, ok):
        row = {"task": "chart", "family": family, "meta": {"expected": {"status": status}}}
        return {"row": row, "reply": "", "correct": ok}

    rows = [reply("zi_transition", "ambiguous", False)] * 9 + [
        reply("zi_transition", "ambiguous", True)
    ]
    rows += [reply("chart_plain", "ok", True)] * 98
    path = tmp_path / "c.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    out = B.calendar_weights(path)
    raw = {"zi_transition:ambiguous": 10 / 12, "chart_plain:ok": B.ERROR_FLOOR}  # 1/100 is floored
    mean = sum(raw.values()) / len(raw)
    assert out["weights"] == {cell: round(v / mean, 4) for cell, v in raw.items()}
    assert out["scored_wrong"] == {"zi_transition:ambiguous": [10, 9], "chart_plain:ok": [98, 0]}
    other = {"task": "verify", "family": "verify", "meta": {"expected": {"status": "ok"}}}
    assert T6.calendar_cell(other) is None  # other calendar tasks keep weight 1


def test_calendar_weights_by_family_keep_the_balance_inside_a_family(tmp_path):
    """Preregistration 12.13: from level 4 a cell's error is that of its whole family."""
    import build_train6 as B

    def reply(family, status, ok):
        row = {"task": "chart", "family": family, "meta": {"expected": {"status": status}}}
        return {"row": row, "reply": "", "correct": ok}

    rows = [reply("jie_window", "ok", False)] * 25 + [reply("jie_window", "ok", True)] * 15
    rows += [reply("jie_window", "needs_tool", False)] * 7
    rows += [reply("jie_window", "needs_tool", True)] * 33
    rows += [reply("chart_plain", "ok", True)] * 40
    path = tmp_path / "c.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    cell, family = B.calendar_weights(path), B.calendar_weights(path, "family")
    assert cell == B.calendar_weights(path, "cell") and "by" not in cell  # the default, as before
    assert cell["weights"]["jie_window:ok"] > 3 * cell["weights"]["jie_window:needs_tool"]
    w = family["weights"]
    assert family["by"] == "family" and w["jie_window:ok"] == w["jie_window:needs_tool"]
    pooled = 33 / 82  # jie_window: 32 wrong of 80
    raw = {
        "chart_plain:ok": B.ERROR_FLOOR,
        "jie_window:needs_tool": pooled,
        "jie_window:ok": pooled,
    }
    mean = sum(raw.values()) / len(raw)
    assert w == {c: round(v / mean, 4) for c, v in raw.items()}
    assert family["scored_wrong"] == cell["scored_wrong"]  # the counts stay per kind


def test_weighted_calendar_rows_favour_the_failing_cells_and_skip_the_fit_sample():
    from collections import Counter

    import build_train6 as B

    from bazireason import ladder as LD
    from bazireason import train_data6 as T6

    sample = T6.fit_sample_c(B.FIT_C_SEED, 3)
    cells = Counter(T6.calendar_cell(r) for r in sample)
    assert None not in cells and max(cells.values()) == 3 and len(cells) >= 15
    assert all(LD.score(r, r["messages"][-1]["content"])["correct"] for r in sample)  # scorable
    seed = 2026111799
    weights = dict.fromkeys(cells, 0.3) | {"zi_transition:ambiguous": 6.0}
    plain = T6.calendar_rows(150_000, seed, False)
    pool_texts = frozenset(T6.first_user(r) for r in plain[:20])  # stand-ins for fit requests
    weighted = T6.calendar_rows(150_000, seed, False, weights=weights, block_texts=pool_texts)
    count = lambda rows: sum(T6.calendar_cell(r) == "zi_transition:ambiguous" for r in rows)  # noqa: E731
    assert count(weighted) > 3 * max(1, count(plain))
    assert not any(T6.first_user(r) in pool_texts for r in weighted)
    again = T6.calendar_rows(150_000, seed, False, weights=weights, block_texts=pool_texts)
    assert [r["id"] for r in weighted] == [r["id"] for r in again]


def test_composition_rows_match_the_rule_code_and_keep_the_held_out_kinds_out(items):
    from bazireason import compose6 as C
    from bazireason import splits6 as S
    from bazireason import train_data6 as T6
    from bazireason.grounded import claim_failures
    from bazireason.p2_prompts import extract_final_json

    assert not set(C.KINDS) & set(S.COMBO_KINDS)
    held_out = {
        it["meta"]["question"] for (_, part), v in items.items() if part == "combo" for it in v
    }
    prefixes = [
        w.split("{")[0] for ws in S.COMBO_WORDS.values() for w in ws if len(w.split("{")[0]) > 6
    ]
    rng = random.Random(20261119)
    for _ in range(50):
        pillars, gender = S.random_pillars(rng), rng.choice(("male", "female"))
        for kind in C.KINDS:
            question, hint, text, reference = C.build(kind, pillars, gender, rng)
            assert extract_final_json(text) == reference, (kind, pillars)
            assert claim_failures(text, pillars, gender) == [], (kind, pillars)
            assert question not in held_out and not any(p in question for p in prefixes)
    charts = T6.Charts(5, frozenset())
    rows = T6.compose_rows(30_000, charts, random.Random(5))
    assert {r["family"] for r in rows} == {"compose"}
    assert {r["meta"]["intent"].split(":")[1] for r in rows} == set(C.KINDS)


def test_composition_rows_take_their_share_from_the_procedure_rows():
    import build_train6 as B

    from bazireason import train_data6 as T6

    exclude = B.exclusions()
    paraphrases = [
        json.loads(x)
        for x in (ROOT / "data/phase6/blind/train6.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    plain = T6.build("s2", 120_000, 9, exclude, paraphrases)
    mixed = T6.build("s2", 120_000, 9, exclude, paraphrases, compose_share=0.10)
    assert not any(r["family"] == "compose" for r in plain)
    tokens = lambda rows, fam: sum(T6.tokens(r) for r in rows if r["family"] == fam)  # noqa: E731
    total = sum(T6.tokens(r) for r in mixed)
    assert 0.08 < tokens(mixed, "compose") / total < 0.13
    assert tokens(mixed, "procedure") < tokens(plain, "procedure")
    # the calendar part does not depend on the arm: both arms of a level get the same requests
    cal = lambda rows: [T6.first_user(r) for r in rows if T6.calendar_cell(r)]  # noqa: E731
    assert sorted(cal(plain)) == sorted(cal(mixed))
    assert T6.digest(plain) == T6.digest(T6.build("s2", 120_000, 9, exclude, paraphrases))


def test_results_table_holds_the_selection_split_back_until_selection(tmp_path, monkeypatch):
    import make_report6 as M

    for label, part in (("p6u2", "rules"), ("p2", "calendar")):
        (tmp_path / "valid6" / label).mkdir(parents=True)
        (tmp_path / "valid6" / label / f"{part}_L0.jsonl").write_text("")
    monkeypatch.setattr(M, "ROOT", tmp_path)
    M.main()
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    held = text.split("## Split `valid6`")[1]
    assert "Phase 2, U2" in held and "selection.json" in held
    assert "###" not in held and "%" not in held and "|" not in held
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid6_scored_before_selection"] == ["p2", "p6u2"]
    # once the selection exists, the split is written out like the others
    (tmp_path / "selection.json").write_text(
        json.dumps({"candidates": {}, "selected": "p6u2", "calendar_claim_failed": False})
    )
    M.main()
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "for selection only, so its tables" not in text
    assert "valid6_scored_before_selection" not in (tmp_path / "summary.json").read_text()


def test_selection_split_counts_stay_out_of_the_log_until_selection(tmp_path, monkeypatch):
    import run6

    monkeypatch.setattr(run6, "ROOT", tmp_path)
    records = [{"score": {"correct": c}, "follows_wrong": not c} for c in (True, False, True)]
    stats = {"generated": 3}
    shown = run6.tally("p6u2", "dev6", "rules", "L0", records, stats)
    assert shown == {"label": "p6u2", "split": "dev6", "part": "rules", "level": "L0", "n": 3,
                     "correct": 2, "stats": stats}  # fmt: skip
    assert run6.tally("p6u2", "dev6", "rules", "L3w", records, stats)["follows_wrong"] == 1
    held = run6.tally("p6u2", "valid6", "rules", "L0", records, stats)
    assert "correct" not in held and held["n"] == 3
    assert run6.tally("p6u2", "sealed6", "rules", "L0", records, stats)["correct"] == 2
    (tmp_path / "selection.json").write_text("{}")
    assert run6.tally("p6u2", "valid6", "rules", "L0", records, stats)["correct"] == 2


def test_second_reading_of_h9_counts_replies_with_wrong_statements(monkeypatch):
    import make_report6 as M

    recs = [
        {"id": f"x{i}", "intent": "domain:wealth:2000" if i < 4 else "flow:2000", "part": "rules"}
        for i in range(10)
    ]
    flags = {
        "p6u3": {"wrong_reply_ids": ["x0", "x1", "x5"], "wrong_hidden_ids": ["x0", "x1", "x5"]},
        "p6u3v": {"wrong_reply_ids": ["x5"], "wrong_hidden_ids": ["x5"]},
    }
    monkeypatch.setattr(M, "rule_records", lambda split, lab, level="L0": recs)
    monkeypatch.setattr(M, "claim_counts", lambda split, lab: flags[lab])
    pair = M.statement_pair("dev6", "p6u3", "p6u3v")
    every = pair["all rule items: replies with a wrong statement"]
    assert (every["a"], every["b"], every["n"]) == (3, 1, 10)
    assert abs(every["difference"]["difference"] + 0.2) < 1e-9
    # only area-of-life replies count in the second row: x5 is a flow-year item
    area = pair["area-of-life items: replies with a wrong hidden-stem statement"]
    assert (area["a"], area["b"], area["n"]) == (2, 0, 4)
    lines, summary = [], {}
    M.fork_statements_section(lines, summary, "dev6")
    assert "second reading" in lines[0] and len(lines) == 6
    assert summary["h9_statements"]["dev6"] == pair


def test_question_years_of_the_chart_life_only_when_asked():
    import build_train6 as B

    from bazireason import train_data6 as T6

    exclude = B.exclusions()
    paraphrases = [
        json.loads(x)
        for x in (ROOT / "data/phase6/blind/train6.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    def years(rows):
        out = {"flow": [], "domain": []}
        for r in rows:
            intent = str(r["meta"].get("intent", ""))
            if r["meta"].get("source") == "template" and intent.startswith(("flow:", "domain:")):
                birth = T6._BIRTH["".join(r["meta"]["pillars"].values())]
                out[intent.split(":")[0]].append((birth, int(intent.split(":")[-1])))
        return out

    fixed = years(T6.build("s2", 300_000, 11, exclude, paraphrases))
    life = years(T6.build("s2", 300_000, 11, exclude, paraphrases, years="life"))
    assert T6._YEARS[0] == "fixed"  # the setting does not outlive the build
    assert fixed["flow"] and fixed["domain"] and life["flow"] and life["domain"]
    assert all(2024 <= y < 2050 for _, y in fixed["flow"])
    assert all(2024 <= y < 2046 for _, y in fixed["domain"])
    assert all(y == 2160 or b + 8 <= y < b + 70 for b, y in life["flow"])
    assert all(y == 2150 or b + 16 <= y < b + 50 for b, y in life["domain"])
    assert min(y for _, y in life["flow"] + life["domain"]) < 2000  # not the fixed window
    # the default is the fixed window, so every earlier set rebuilds unchanged
    again = T6.build("s2", 300_000, 11, exclude, paraphrases)
    assert T6.digest(again) == T6.digest(
        T6.build("s2", 300_000, 11, exclude, paraphrases, years="fixed")
    )
    with pytest.raises(ValueError):
        T6.build("s2", 10_000, 11, exclude, paraphrases, years="wide")
