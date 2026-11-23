"""Descriptive error localization on fixed raw outputs, without inferring hidden model states."""
from collections import Counter
import json
from pathlib import Path
from graphs import calc, parse, rows, R, derived

OUT = Path("results/phase9")
MODELS = ("g","a","s","x","p")


def main():
    data = list(rows("data/phase9/transfer.jsonl"))
    stage = set(R.LIFE_STAGES)
    gods = {R.ten_god(s,t) for s in "甲乙丙丁戊己庚辛壬癸" for t in "甲乙丙丁戊己庚辛壬癸"}
    answers = {m: {r["id"]:r for r in rows(OUT/f"{m}_transfer.jsonl")} for m in MODELS}
    stats = {}
    examples = {}
    for model, replies in answers.items():
        c = Counter()
        ex = {}
        groups = {}
        for r in data:
            groups.setdefault(r["group"], {})[r["condition"],r["op"]] = r
        for r in data:
            if r["condition"] not in ("novel","deep"):
                continue
            prefix = r["condition"]+"_"
            obs = replies[r["id"]]
            natal = r["chart"][r["pos"]][1]
            value = parse(obs["raw"])
            if r["branch"] != natal:
                c[prefix+"changed_branch_n"] += 1
                c[prefix+"changed_branch_right"] += obs["correct"]
            natal_value = calc(r["op"],r["chart"],natal)
            if natal_value != r["answer"]:
                c[prefix+"natal_answer_differs"] += 1
                c[prefix+"right_when_natal_answer_differs"] += obs["correct"]
            if r["condition"] == "deep":
                after_one = calc(r["op"],r["chart"],r["intermediates"][0])
                second_only = derived(r["stages"][1],r["chart"],r["pos"])
                after_second_only = calc(r["op"],r["chart"],second_only)
                if all(v != r["answer"] for v in [natal_value,after_one,after_second_only]):
                    c["deep_all_shortcuts_wrong_n"] += 1
                    c["deep_all_shortcuts_wrong_right"] += obs["correct"]
                if after_one != r["answer"]:
                    c["deep_second_step_matters"] += 1
                    c["deep_right_when_second_step_matters"] += obs["correct"]
                    c["deep_matches_skip_second_step"] += type(after_one) == type(value) and value == after_one
            if obs["correct"]:
                continue
            c[prefix+"misses"] += 1
            c[prefix+"unparsed_among_misses"] += value is None
            if r["condition"] != "novel":
                continue
            if r["op"] in ("god","stage"):
                expected = gods if r["op"] == "god" else stage
                cell = ("unparsed" if value is None else "wrong_same_class" if isinstance(value,str) and value in expected
                        else "wrong_other_class")
                c[r["op"]+"_"+cell] += 1
            cg = groups[r["group"]]
            isolated = replies[cg["consumer",r["op"]]["id"]]
            source = replies[cg["producer","branch"]["id"]]
            if source["correct"] and isolated["correct"] and r["op"] not in ex:
                ex[r["op"]] = dict(id=r["id"],source=r["source"], op=r["op"],gold=r["answer"],branch=r["branch"],
                                    reply=obs["raw"], consumer_reply=isolated["raw"], producer_reply=source["raw"])
        stats[model], examples[model] = c, ex
    (OUT/"error_audit.json").write_text(json.dumps(dict(counts=stats,examples=examples),ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
