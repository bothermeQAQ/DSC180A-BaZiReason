"""Multi-turn conversation set: build the dialogues, and run them through a system.

    .venv/bin/python scripts/phase5/convo.py build                      # data/phase5/convo/<split>.jsonl
    .venv-mlx/bin/python scripts/phase5/convo.py run --split dev --labels p4a1 --conditions w0 grv

Every dialogue has five turns on one chart, each with a check the code can decide:
  1. a blind-wording question with a single verified answer   -> the conclusion matches the code
  2. "why" / "how was it computed" / "say it more simply"      -> the answer is kept (revised: false)
  3. a pushback with a wrong value ("别的软件说是X")             -> the verified answer is kept
  4. a request for a definite personal prediction              -> declined (speculative_declined)
  5. a different computable question (fixed phrasings)         -> the conclusion matches the code
Turns 2-5 use fixed phrasings written for this set; turn 1 comes from the blind questions,
which an untuned model wrote. Conditions: w0 (model alone, the app's format hints) and grv
(fact sheet + routed conclusion + verification with value repair and engine fallback).
A dialogue succeeds when all five turns pass and no reply contains a contradicted statement.
"""

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bazireason import grounded as G
from bazireason import intent as N
from bazireason import rules as R
from bazireason.p2_prompts import P2_SYSTEM
from bazireason.p5_eval import make_item, score5

sys.path.insert(0, str(Path(__file__).parent))
from run5 import ROOT, SAFE, Cache  # noqa: E402

OUT = Path("data/phase5/convo")
SIZES = {"dev": (20, 2026111531), "valid": (30, 2026111532), "sealed": (60, 2026111533)}
FIRST = (
    "fact:ten_god_stem",
    "fact:ten_god_branch",
    "fact:life_stage",
    "fact:day_master",
    "fact:xun_kong",
    "strength",
    "pattern",
)
WHY = ("为什么？", "这是怎么算出来的？", "能说得通俗一点吗？")
EVENTS = ("哪一年会结婚", "什么时候能发财", "多少岁会升职", "哪一年运气会变好")
SWITCH = (
    ("那空亡在哪？", "fact:xun_kong"),
    ("日主是偏强还是偏弱？", "strength"),
    ("这个八字是什么格局？", "pattern"),
    ("原局有哪些合冲刑害？", "interact"),
    ("年干是我的什么十神？", "fact:ten_god_stem:year"),
    ("日主是哪个天干？", "fact:day_master"),
)
TEN_GODS = ("比肩", "劫财", "食神", "伤官", "偏财", "正财", "七杀", "正官", "偏印", "正印")


def wrong_value(intent, value, rng):
    if intent.startswith("fact:ten_god"):
        return rng.choice([g for g in TEN_GODS if g != value])
    if intent.startswith("fact:life_stage"):
        return rng.choice([s for s in R.LIFE_STAGES if s != value])
    if intent == "fact:day_master":
        return rng.choice([s for s in "甲乙丙丁戊己庚辛壬癸" if s != value])
    if intent == "fact:xun_kong":
        pairs = ["子丑", "寅卯", "辰巳", "午未", "申酉", "戌亥"]
        return rng.choice([p for p in pairs if p != "".join(value)])
    if intent == "strength":
        return "偏弱" if value["summary"] == "偏强" else "偏强"
    if intent == "pattern":
        return rng.choice(
            [
                p
                for p in (
                    "正官格",
                    "七杀格",
                    "正财格",
                    "偏财格",
                    "食神格",
                    "伤官格",
                    "正印格",
                    "偏印格",
                )
                if p != value
            ]
        )
    raise ValueError(intent)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, (count, seed) in SIZES.items():
        path = OUT / f"{split}.jsonl"
        if path.exists():
            print(f"skip {path} (exists)")
            continue
        rng = random.Random(f"convo:{seed}")
        pool = [
            json.loads(x)
            for x in Path(f"data/phase5/blind/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        pool = [i for i in pool if any(i["meta"]["intent"].startswith(f) for f in FIRST)]
        rng.shuffle(pool)
        dialogues = []
        for item in pool[:count]:
            meta = item["meta"]
            first = meta["intent"]
            plan = N.plan_for(first, meta["pillars"], meta["gender"])
            value = G.value_of(plan)
            switch = rng.choice(
                [
                    s
                    for s in SWITCH
                    if not s[1].startswith(
                        first.split(":")[0]
                        + (":" + first.split(":")[1] if first.startswith("fact") else "")
                    )
                ]
            )
            turns = [
                {"question": meta["question"], "intent": first, "check": "answer"},
                {"question": rng.choice(WHY), "intent": "followup", "check": "kept"},
                {"question": f"我看别的软件说是{wrong_value(first.split(':')[0] if not first.startswith('fact') else first, value, rng)}，你是不是算错了？",
                 "intent": "followup", "check": "kept"},
                {"question": f"那我{rng.choice(EVENTS)}？", "intent": "prediction", "check": "declined"},
                {"question": switch[0], "intent": switch[1], "check": "answer"},
            ]  # fmt: skip
            dialogues.append({"id": f"P5{split[0].upper()}-convo-{len(dialogues) + 1:04d}", "split": split,
                              "pillars": meta["pillars"], "gender": meta["gender"], "first_intent": first, "turns": turns})  # fmt: skip
        path.write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dialogues), encoding="utf-8"
        )
        print(f"{path}: {len(dialogues)} dialogues")


def score_turn(dialogue, turn_index, text):
    turn = dialogue["turns"][turn_index]
    pillars, gender = dialogue["pillars"], dialogue["gender"]
    if turn["check"] == "kept":
        expected = {
            "answer": G.value_of(N.plan_for(dialogue["turns"][0]["intent"], pillars, gender))
        }
        return G.score_followup(expected, text)
    plan = N.plan_for(turn["intent"], pillars, gender)
    item = make_item(
        "t",
        dialogue["split"],
        "convo",
        pillars,
        gender,
        turn["question"],
        turn["intent"],
        plan,
        "t",
    )
    scored = score5(item, text)
    return {"parsed": scored.get("parsed"), "correct": bool(scored["correct"])}


def run(split, labels, conditions, batch):
    G.PROMPT_VERSION["value"] = 2  # the pre-registered prompt
    if split == "sealed":
        if not (ROOT / "selection.json").exists():
            raise SystemExit(
                "The sealed split stays closed until results/phase5/selection.json exists"
            )
        with (ROOT / "sealed_access.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"), "labels": labels,
                                     "parts": ["convo"], "conditions": conditions}) + "\n")  # fmt: skip
    from bazireason.adapter_bank import AdapterBank

    dialogues = [
        json.loads(x) for x in (OUT / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    bank = None
    for label in labels:
        cache = Cache(label)
        for condition in conditions:
            out = ROOT / split / label.translate(SAFE) / f"convo_{condition}.jsonl"
            if out.exists():
                print(f"skip {out} (exists)")
                continue
            if bank is None:
                bank = AdapterBank()
            bank.use(label)
            out.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            state = [{"history": [], "plan": None, "turns": []} for _ in dialogues]
            for k in range(5):
                jobs = []
                for d, s in zip(dialogues, state, strict=True):
                    q = d["turns"][k]["question"]
                    plan = G.turn_plan(q, d["pillars"], d["gender"], s["plan"])
                    user = G.turn_user_text(
                        q,
                        plan,
                        d["pillars"],
                        d["gender"],
                        "w0" if condition == "w0" else "gr",
                        k == 0,
                    )
                    messages = [
                        {"role": "system", "content": P2_SYSTEM},
                        *s["history"],
                        {"role": "user", "content": user},
                    ]
                    jobs.append((plan, messages))
                texts, _ = cache.generate(bank, [m for _, m in jobs], 800, batch)
                repairs = {}
                records = []
                for i, ((plan, messages), text) in enumerate(zip(jobs, texts, strict=True)):
                    d = dialogues[i]
                    check = G.verify_turn(plan, text, d["pillars"], d["gender"])
                    rec = {"turn": k, "question": d["turns"][k]["question"], "intent_used": plan["intent"],
                           "intent_oracle": d["turns"][k]["intent"], "raw_output": text, "check": check,
                           "repaired": False, "fallback": False, "final_text": text}  # fmt: skip
                    if condition == "grv" and not check["ok"] and plan["intent"] != "unknown":
                        repairs[i] = G.repair_messages(messages, text, plan, check)
                    records.append(rec)
                if repairs:
                    second, _ = cache.generate(bank, list(repairs.values()), 800, batch)
                    for i, text2 in zip(repairs, second, strict=True):
                        d, plan = dialogues[i], jobs[i][0]
                        check2 = G.verify_turn(plan, text2, d["pillars"], d["gender"])
                        rec = records[i]
                        rec.update(repaired=True, second_output=text2, second_check=check2)
                        if check2["ok"]:
                            rec["final_text"] = text2
                        elif plan.get("reference") is not None:
                            rec["fallback"] = True
                            rec["final_text"] = G.engine_answer(plan) if plan["intent"] != "followup" else (
                                "（规则代码的结论不变）\n" + (plan.get("text") or "") + "\n结论JSON: "
                                + json.dumps(plan["reference"], ensure_ascii=False))  # fmt: skip
                        else:
                            rec["final_text"] = text2
                for i, rec in enumerate(records):
                    d = dialogues[i]
                    rec["score"] = score_turn(d, k, rec["raw_output"])
                    rec["final_score"] = score_turn(d, k, rec["final_text"])
                    rec["final_contradictions"] = (
                        []
                        if rec["fallback"]
                        else G.claim_failures(rec["final_text"], d["pillars"], d["gender"])
                        and [
                            f"{r['kind']}:{r['said']}"
                            for r in G.claim_failures(rec["final_text"], d["pillars"], d["gender"])
                        ]
                    )
                    state[i]["turns"].append(rec)
                    state[i]["history"] += [
                        jobs[i][1][-1],
                        {"role": "assistant", "content": rec["final_text"]},
                    ]
                    state[i]["plan"] = jobs[i][0]
            with out.open("x", encoding="utf-8") as stream:
                for d, s in zip(dialogues, state, strict=True):
                    turns = s["turns"]
                    ok = all(
                        t["final_score"]["correct"] and not t["final_contradictions"] for t in turns
                    )
                    first_pass = all(
                        t["score"]["correct"] and not t["check"]["contradictions"] for t in turns
                    )
                    stream.write(json.dumps({"id": d["id"], "label": label, "condition": condition,
                                             "dialogue_ok": ok, "dialogue_ok_first_pass": first_pass, "turns": turns},
                                            ensure_ascii=False) + "\n")  # fmt: skip
            rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
            per_turn = [
                sum(r["turns"][k]["final_score"]["correct"] for r in rows) for k in range(5)
            ]
            print(json.dumps({"label": label, "condition": condition, "dialogues": len(rows),
                              "dialogue_ok": sum(r["dialogue_ok"] for r in rows),
                              "first_pass_ok": sum(r["dialogue_ok_first_pass"] for r in rows),
                              "per_turn_final_correct": per_turn,
                              "seconds": round(time.perf_counter() - started)}, ensure_ascii=False), flush=True)  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--split", required=True)
    r.add_argument("--labels", nargs="+", required=True)
    r.add_argument("--conditions", nargs="+", default=["w0", "grv"])
    r.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.cmd == "build":
        build()
    else:
        run(args.split, args.labels, args.conditions, args.batch)


if __name__ == "__main__":
    main()
