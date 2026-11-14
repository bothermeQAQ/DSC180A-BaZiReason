"""Reproduce the numbers of docs/phase6_capability_audit.md from saved records (no model).

    PYTHONPATH=src .venv/bin/python scripts/phase6/audit_repro.py   # results/phase6/audit_repro.json

Every number in the audit's Part I is recomputed here; the output lists each with the value
quoted in the audit, so a difference is visible.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
import run5  # noqa: E402

from bazireason import workflow as W  # noqa: E402
from bazireason.claim_check import check_claims  # noqa: E402

P5 = Path("results/phase5")
P4 = Path("results/phase4/final_test")
OUT = Path("results/phase6/audit_repro.json")
PARTS = ("rules", "indep", "blind")
LOOKUPS = ("day_master", "hidden", "pillar", "spouse_star")


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def p5(split, label, part, cond):
    path = P5 / split / label / f"{part}_{cond}.jsonl"
    return rows(path) if path.exists() else None


def correct(recs, final=False):
    return sum(bool((r["final_score"] if final else r["score"])["correct"]) for r in recs)


def ladder():
    out = {}
    for split in ("valid", "sealed"):
        cells = {}
        for label, cond, final in (("base", "w0", False), ("p4a1", "w0", False), ("p5g1", "w0", False),
                                   ("p5g1", "gf", False), ("p5g1", "gr", False), ("p5g1", "grv", True)):  # fmt: skip
            recs = [r for p in PARTS for r in (p5(split, label, p, cond) or [])]
            if recs:
                cells[f"{label}/{cond}{'/final' if final else ''}"] = [
                    correct(recs, final),
                    len(recs),
                ]
        out[split] = cells
    grv = [r for p in PARTS for r in p5("sealed", "p5g1", p, "grv")]
    out["sealed_g1_repairs"] = {
        "repaired": sum(bool(r["repaired"]) for r in grv),
        "fallback": sum(bool(r["fallback"]) for r in grv),
        "repaired_and_right_without_fallback": sum(bool(r["repaired"]) and not r["fallback"] and bool(r["final_score"]["correct"]) for r in grv),
    }  # fmt: skip
    return out


def statements_a1_sealed():
    items = {i["id"]: i for p in PARTS for i in run5.load_items("sealed", p)}
    checked, wrong = Counter(), Counter()
    errors = Counter()
    family = defaultdict(lambda: [0, 0])
    template_vs_blind = defaultdict(lambda: [0, 0, 0, 0])
    for part in PARTS:
        for r in p5("sealed", "p4a1", part, "w0"):
            meta = items[r["id"]]["meta"]
            claims = check_claims(r["raw_output"], meta["pillars"], meta.get("gender"))
            bad = [c for c in claims if not c["ok"]]
            for c in claims:
                checked[c["kind"]] += 1
                wrong[c["kind"]] += not c["ok"]
            ok = bool(r["score"]["correct"])
            fam = r["intent_oracle"].split(":")[0]
            family[fam][0] += ok
            family[fam][1] += 1
            if part in ("rules", "blind"):
                k = 0 if part == "rules" else 2
                template_vs_blind[fam][k] += ok
                template_vs_blind[fam][k + 1] += 1
            if not ok:
                if not r["score"].get("parsed"):
                    errors["no_readable_conclusion"] += 1
                elif W.has_loop(r["raw_output"]):
                    errors["loop"] += 1
                elif bad:
                    errors["wrong_with_a_wrong_statement"] += 1
                else:
                    errors["wrong_no_wrong_checkable_statement"] += 1
            elif bad and fam == "ambig":
                errors["ambig_right_with_a_wrong_statement"] += 1
    lookups = [sum(checked[k] for k in LOOKUPS), sum(wrong[k] for k in LOOKUPS)]
    rules = p5("sealed", "p4a1", "rules", "w0")
    seen = [r for r in rules if not r["score"].get("unseen_wording")]
    unseen = [r for r in rules if r["score"].get("unseen_wording")]
    return {
        "statement_kinds": {k: [wrong[k], checked[k]] for k, _ in checked.most_common()},
        "lookups_wrong_of_checked": [lookups[1], lookups[0]],
        "error_kinds": dict(errors),
        "family_accuracy": {k: v for k, v in sorted(family.items(), key=lambda kv: -kv[1][1])},
        "template_vs_blind": {k: v for k, v in template_vs_blind.items() if v[1] and v[3]},
        "template_rules_seen_unseen": [[correct(seen), len(seen)], [correct(unseen), len(unseen)]],
    }


def evidence_g1_valid():
    fam = defaultdict(Counter)
    for cond in ("w0", "gf", "go"):
        for part in PARTS:
            for r in p5("valid", "p5g1", part, cond):
                f = r["intent_oracle"].split(":")[0]
                fam[f][cond] += bool(r["score"]["correct"])
                fam[f]["n"] += 1 if cond == "w0" else 0
    dev = {c: sum(correct(p5("dev", "p4a1", p, c)) for p in PARTS) for c in ("w0", "gf")}
    return {"g1_valid_by_family": {k: dict(v) for k, v in fam.items()}, "a1_dev_w0_vs_gf": dev}


def self_correction_phase4():
    out = {}
    for path in sorted(P4.glob("*_w1.jsonl")):
        recs = rows(path)
        wrong = [r for r in recs if not r.get("first_correct")]
        out[path.name] = {"first_wrong": len(wrong), "fixed": sum(bool(r["score"]["correct"]) for r in wrong),
                          "broke": sum(bool(r.get("first_correct")) and not r["score"]["correct"] for r in recs)}  # fmt: skip
    by_model = defaultdict(lambda: [0, 0])
    for name, v in out.items():
        by_model[name.split("_")[0]][0] += v["fixed"]
        by_model[name.split("_")[0]][1] += v["first_wrong"]
    return {"files": out, "fixed_of_first_wrong_by_model": dict(by_model)}


def calendar_sealed():
    out, twins = {}, {}
    for label in ("p2", "p3r3", "p4a1", "p5g1"):
        sc = [r["score"] for r in p5("sealed", label, "calendar", "w0")]
        ok = [s for s in sc if s.get("expected_status") == "ok"]
        other = [s for s in sc if s.get("expected_status") not in (None, "ok")]
        bda = (
            sum(bool(s.get("status_correct")) for s in ok) / len(ok)
            + sum(bool(s.get("status_correct")) for s in other) / len(other)
        ) / 2
        out[label] = {"answerable_refused": [sum(bool(s.get("false_refusal")) for s in sc), len(ok)],
                      "definite_on_nonanswer": [sum(bool(s.get("definitive_on_nonanswer")) for s in sc), len(other)],
                      "balanced_decision": round(bda, 4)}  # fmt: skip
        jw = [s for s in sc if s.get("family") == "jie_window"]
        twins[label] = {st: [sum(bool(s["correct"]) for s in jw if s["expected_status"] == st), sum(s["expected_status"] == st for s in jw)]
                        for st in ("needs_tool", "ok")}  # fmt: skip
    return {"decisions": out, "jie_window_twins": twins}


def training_wordings():
    fam_q = defaultdict(Counter)
    for d in (
        "data/phase3/train_v1",
        "data/phase3/train_v2",
        "data/phase3/train_v3",
        "data/phase4/train_v1",
    ):
        for r in rows(f"{d}/train.jsonl"):
            f = r["family"]
            if f.startswith("replay") or f in ("drill", "mcq3", "strict_audit", "claims", "dialog"):
                continue
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            lines = [
                ln
                for ln in user.split("\n")
                if ln.strip()
                and not ln.startswith("结论JSON")
                and not ln.startswith("[")
                and "引擎" not in ln
            ]
            q = lines[-1] if lines else ""
            q = re.sub(r"^八字：.*?命。|^八字：.*?）。", "", q)
            q = re.sub(r"\d{4}", "YYYY", q)
            q = re.sub(r"[甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥]", "〇", q)
            fam_q[f][q[:70]] += 1
    return {f: {"wordings": len(c), "rows": sum(c.values())} for f, c in sorted(fam_q.items())}


def phase4_final():
    out = {}
    for label in ("base", "p2", "p3r2", "p3r3", "p4a1"):
        ind, p3, p2 = (rows(P4 / f"{label}_{p}.jsonl") for p in ("indep", "p3", "p2"))
        seen = [r for r in p3 if not r["score"].get("unseen_wording")]
        unseen = [r for r in p3 if r["score"].get("unseen_wording")]
        out[label] = {"indep": [correct(ind), len(ind)], "rule_rows_seen": [correct(seen), len(seen)],
                      "rule_rows_unseen": [correct(unseen), len(unseen)], "calendar": [correct(p2), len(p2)]}  # fmt: skip
    return out


def main():
    result = {
        "ladder": ladder(),
        "a1_alone_sealed": statements_a1_sealed(),
        "evidence": evidence_g1_valid(),
        "self_correction_phase4_w1": self_correction_phase4(),
        "calendar_sealed": calendar_sealed(),
        "training_wordings_phase3_4": training_wordings(),
        "phase4_final_test": phase4_final(),
        "audit_quotes": {
            "valid ladder": "base 64, A1 347, G1 341, G1 gf 366, G1 gr 461, final 491 (of 498)",
            "sealed": "A1 alone 688, G1 gr 918, final 991 (of 1005); repaired 73, fallback 5",
            "lookups": "0 of 1,884 wrong",
            "xun_kong": "35/51",
            "pattern": "55/141",
            "element_count": "24/73",
            "errors": "195 no wrong checkable statement, 117 with one, 5 no conclusion, 0 loops",
            "self-correction": "A1 19/556, R3 17/631, P2 11/941",
            "calendar": "refused P2 56, R3 15, A1 198, G1 125 of 532; jie_window twins 5/28, 1/30, 39/1, 31/6",
        },  # fmt: skip
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("ladder",)}, ensure_ascii=False))
    print(
        json.dumps(result["a1_alone_sealed"]["lookups_wrong_of_checked"]),
        result["a1_alone_sealed"]["error_kinds"],
    )
    print(result["self_correction_phase4_w1"]["fixed_of_first_wrong_by_model"])
    print(json.dumps(result["calendar_sealed"], ensure_ascii=False))


if __name__ == "__main__":
    main()
