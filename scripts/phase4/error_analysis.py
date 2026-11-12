"""Phase 4 error analysis on saved outputs only (no model is run).

    python scripts/phase4/error_analysis.py

Every split read here has already been scored and inspected in Phase 3, so in Phase 4
it is development material. The untouched Phase 4 test split is never read here.
Writes results/phase4/error_analysis.json and prints the headline counts.
"""

import collections
import json
import re
from pathlib import Path

H = Path("results/phase3/heldout")
OUT = Path("results/phase4/error_analysis.json")
P2_SPLITS = {
    "check": ("data/phase3/p2check_v1/test.jsonl", "p2check_p2", "p2check_p3r2", "p2check_p3r3"),
    "first": (
        "data/phase2/train_v1/test.jsonl",
        "../../phase2/heldout/tuned_v2",
        "p2test_p3r2",
        "p2test_p3r3",
    ),
    "fresh": (
        "data/phase2/train_v2/test.jsonl",
        "../../phase2/heldout/fresh_tuned_v2",
        "p2fresh_p3r2",
        "p2fresh_p3r3",
    ),
}
MODELS = ("p2", "p3r2", "p3r3")
# Signature phrases of each rule procedure in an answer.
SIGNATURES = {
    "strength": re.compile(r"得令[：:]"),
    "element_count": re.compile(r"合计[：:]\s*木"),
    "god_table": re.compile(r"十神表"),
    "branch_card": re.compile(r"卡片[：:]|支卡片"),
    "flow": re.compile(r"流年[：:]"),
    "pattern": re.compile(r"取格|透干.{0,6}格"),
}
EXPECTED_PROCEDURE = {
    "rule_elements": "element_count",
    "rule_strength": "strength",
    "p3_strength": "strength",
    "rule_flow_year": "flow",
    "p3_flow": "flow",
}
LOOP = re.compile(r"(.{20,}?)\1\1", re.S)
TITLE = re.compile(r"《([^》]{1,12})》")


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def output(stem):
    path = H / f"{stem}.jsonl"
    return {row["id"]: row for row in load(path)}


def calendar_and_rules():
    """Issues 1, 2 and 5 on the three Phase 2-style splits."""
    fam = collections.defaultdict(lambda: collections.Counter())
    lost = collections.Counter()
    refusals = collections.Counter()
    rule_by_kind = collections.defaultdict(collections.Counter)
    confusion = collections.Counter()
    for _, (data, *stems) in P2_SPLITS.items():
        rows = {row["id"]: row for row in load(data)}
        outs = dict(zip(MODELS, (output(s) for s in stems), strict=True))
        for rid, row in rows.items():
            for model in MODELS:
                score = outs[model][rid]["score"]
                fam[row["family"]][(model, "n")] += 1
                fam[row["family"]][(model, "k")] += bool(score["correct"])
                if score.get("false_refusal"):
                    refusals[(model, row["family"])] += 1
                if row["family"] == "rule":
                    rule_by_kind[row["task"]][model] += bool(score["correct"])
                    want = EXPECTED_PROCEDURE.get(row["task"])
                    text = outs[model][rid]["raw_output"] or ""
                    used = [k for k, sig in SIGNATURES.items() if sig.search(text)]
                    if want and used and want not in used:
                        confusion[(model, row["task"], "+".join(used))] += 1
            p2_ok = outs["p2"][rid]["score"]["correct"]
            r3_ok = outs["p3r3"][rid]["score"]["correct"]
            if p2_ok and not r3_ok:
                lost[(row["family"], row["task"])] += 1
    families = {
        f: {m: f"{c[(m, 'k')]}/{c[(m, 'n')]}" for m in MODELS} for f, c in sorted(fam.items())
    }
    worse = {
        f: v
        for f, v in families.items()
        if int(v["p3r3"].split("/")[0]) < int(v["p2"].split("/")[0])
    }
    return {
        "families_all_three_splits": families,
        "families_where_round3_below_phase2": worse,
        "phase2_right_round3_wrong_by_task": {f"{f}/{t}": n for (f, t), n in lost.most_common()},
        "false_refusals": {f"{m}/{f}": n for (m, f), n in sorted(refusals.items())},
        "phase2_rule_tasks_by_kind": {k: dict(v) for k, v in sorted(rule_by_kind.items())},
        "procedure_confusion_phase2_rules": {
            f"{m}/{t}/{u}": n for (m, t, u), n in confusion.most_common()
        },
    }


def phase3_split():
    """Issues 2 and 3 on the Phase 3 test split (inspected in Phase 3)."""
    rows = {row["id"]: row for row in load("data/phase3/train_v1/test.jsonl")}
    out = {}
    for model in ("p3r2", "p3r3"):
        res = output(f"p3test_{model}")
        wording = collections.Counter()
        confusion = collections.Counter()
        tool_vs_bare = collections.Counter()
        for rid, r in res.items():
            s = r["score"]
            if s["family"] == "strict_shift":
                continue
            key = "unseen" if s.get("unseen_wording") else "seen"
            wording[(key, "n")] += 1
            wording[(key, "k")] += bool(s["correct"])
            tool_vs_bare[(s["mode"], "n")] += 1
            tool_vs_bare[(s["mode"], "k")] += bool(s["correct"])
            want = EXPECTED_PROCEDURE.get(rows[rid]["task"])
            used = [k for k, sig in SIGNATURES.items() if sig.search(r["raw_output"] or "")]
            if want and used and want not in used:
                confusion[(rows[rid]["task"], "+".join(used))] += 1
        out[model] = {
            "seen_wording": f"{wording[('seen', 'k')]}/{wording[('seen', 'n')]}",
            "unseen_wording": f"{wording[('unseen', 'k')]}/{wording[('unseen', 'n')]}",
            "by_mode": {
                m: f"{tool_vs_bare[(m, 'k')]}/{tool_vs_bare[(m, 'n')]}"
                for m in sorted({m for m, _ in tool_vs_bare})
            },
            "procedure_confusion": {f"{t}/{u}": n for (t, u), n in confusion.items()},
        }
    return out


def benchmark_v3():
    """Issues 4 and 5: strict benchmark_v3, by condition."""
    out = {}
    for label, path in (
        ("p2", "results/phase2/v3/tuned_v2_strict.jsonl.scores.json"),
        ("p3r2", "results/phase3/v3/p3r2_strict.jsonl.scores.json"),
        ("p3r3", "results/phase3/v3/p3r3_strict.jsonl.scores.json"),
    ):
        if not Path(path).exists():
            continue
        methods = json.loads(Path(path).read_text())["methods"]
        out[label] = {
            m: {
                k: methods[m]["overall"].get(k)
                for k in (
                    "decision_accuracy",
                    "false_refusal_rate",
                    "ambiguous_definitive_rate",
                    "corruption_following_rate",
                    "corruption_following_by_type",
                )
                if methods[m]["overall"].get(k) is not None
            }
            for m in methods
        }
    return out


def baziqa():
    """Issues 6 and 7: loops, titles, unanswered and reply length, no answer keys read."""
    out = {}
    for label, folder, prefix in (
        ("p2", "p2", "tuned"),
        ("p3r2", "p3", "tuned"),
        ("p3r3", "p3r3", "tuned"),
    ):
        for tools in ("no_tools", "tools"):
            path = Path(f"results/phase3/baziqa1024/{folder}/{prefix}_{tools}.jsonl")
            if not path.exists():
                continue
            rows = load(path)
            loops = [r for r in rows if LOOP.search(r["raw_output"] or "")]
            out[f"{label}_{tools}"] = {
                "replies": len(rows),
                "loops": len(loops),
                "loops_without_answer": sum(not r["parseable"] for r in loops),
                "unanswered": sum(not r["parseable"] for r in rows),
                "truncated": sum(bool(r.get("truncated")) for r in rows),
                "mean_tokens": round(sum(r["completion_tokens"] for r in rows) / len(rows)),
                "titles": dict(
                    collections.Counter(
                        t for r in rows for t in TITLE.findall(r["raw_output"] or "")
                    ).most_common(8)
                ),
            }
    return out


def main():
    report = {
        "note": "Saved Phase 3 outputs only; every split here was inspected before Phase 4.",
        "calendar_and_rules": calendar_and_rules(),
        "phase3_split": phase3_split(),
        "benchmark_v3_strict": benchmark_v3(),
        "baziqa_replies": baziqa(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:9000])


if __name__ == "__main__":
    main()
