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

    for split in (*SPLITS, "train6"):
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
