"""Natural-wording questions for Phase 6, written by untuned local models (never by the author).

    .venv-mlx/bin/python scripts/phase6/gen_blind6.py --splits dev6 valid6 sealed6   # test writer
    .venv-mlx/bin/python scripts/phase6/gen_blind6.py --splits train6                # training writer
    .venv/bin/python scripts/phase6/gen_blind6.py --check                             # rebuild items from logs

Two writers, kept apart:
  * evaluation (dev6, valid6, sealed6): Qwen3.5-4B, with the Phase 5 descriptions and filters
    (scripts/phase5/gen_blind.py), on charts from the Phase 6 blocks;
  * training (train6): the untuned base Qwen3-4B-Instruct, with a different set of
    descriptions and personas written for training, on charts from the training years.
No writer sees a project template, rule card or training row. Labels come from the rule code
(bazireason.intent.plan_for). Every attempt is kept in data/phase6/blind/<split>_attempts.jsonl.
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, "scripts/phase5")
sys.path.insert(0, "scripts/phase6")
import gen_blind as GB  # noqa: E402

from bazireason import intent as N  # noqa: E402
from bazireason import splits6 as S  # noqa: E402
from bazireason import train_data as TD  # noqa: E402
from bazireason.indep_rules import eight_char  # noqa: E402
from bazireason.p5_eval import make_item  # noqa: E402

OUT = Path("data/phase6/blind")
TEST_WRITER = "models/qwen3.5-4b-mlx-4bit"
TRAIN_WRITER = "models/qwen3-4b-instruct-2507-mlx-4bit"
CLASSES = tuple(c for c in GB.CLASSES if c != "day_master")  # the head line names the day master
PLAN = {
    "dev6": (2026111641, 10),
    "valid6": (2026111642, 10),
    "sealed6": (2026111643, 21),
    "train6": (2026111649, 180),
    "train6b": (2026111650, 220),  # more training wordings for the full-budget run
}
PERSONAS = (
    "你是一位刚开始了解八字的年轻人，正在向一个八字助手请教。",
    "你是一位研究命理多年的爱好者，说话比较专业，正在和八字助手讨论。",
    "你在帮家里人问八字的问题，说话很口语。",
    "你是一个说话简短直接的人，正在用手机和八字助手聊天。",
)
ASK_TRAIN = (
    "这是{owner}的八字：{chart}，{sex}。请你自然地向助手提一个问题，想弄清楚的是：{desc}。只输出这一句提问。",
    "八字：{chart}（{sex}）。用你自己的说法问助手一句话，问的内容是：{desc}。不要回答，也不要解释，只写问题。",
)
OWNERS = ("我", "我朋友", "我家孩子", "我爱人")
POS = GB.POS_ZH
TRAIN_DESC = {
    "ten_god_stem": (
        "{p}干{s}和日主{dm}之间是什么十神关系",
        "站在日主{dm}的角度，{p}柱天干{s}算哪一种十神",
    ),
    "ten_god_branch": (
        "{p}支{b}的本气对日主{dm}来说是哪种十神",
        "只看{p}支{b}里的主气，它是日主{dm}的什么十神",
    ),
    "hidden": ("{p}支{b}里藏了几个天干、分别是哪几个", "{p}柱地支{b}的人元（藏干）有哪些"),
    "life_stage": (
        "日主{dm}落在{p}支{b}，长生十二宫里算哪一宫",
        "日主{dm}到了{p}支{b}是长生、帝旺还是别的什么状态",
    ),
    "xun_kong": ("日柱所在的那一旬里哪两个地支落空（空亡）", "八字里的空亡是哪两个字"),
    "luck_direction": ("大运应该往前顺着排还是往后倒着排", "大运的排法是顺排还是逆排"),
    "element_count": ("四柱八个字里五行各出现几次", "把八个字按金木水火土分一分，每样有几个"),
    "missing_element": ("八个字里有没有哪一种五行一个都没有", "五行齐不齐，缺哪一行"),
    "strength": ("日主到底是身强还是身弱", "这个盘的日元旺不旺，算强还是算弱"),
    "pattern": ("按月令来看这个八字成什么格", "这个命局属于哪一种格局"),
    "interact": ("四柱之间有没有相合、相冲、相刑、相害、相破", "干支之间都有哪些刑冲合害的关系"),
    "gods": (
        "年、月、时三个天干各是什么十神，凑在一起有没有特别的组合",
        "天干上的十神分别是什么，能组成什么搭配",
    ),
    "yongshen_open": ("这个八字的用神和喜神是什么", "八字喜什么五行、用神是哪个"),
    "no_rule": ("这个八字是不是好命、以后富不富贵", "命好不好，整体看算上等还是下等"),
    "flow": ("{year}年这个流年和原局之间有什么关系", "到了{year}年，流年干支跟八字会产生哪些作用"),
    "star": ("{star}落在八字的哪个地支上", "八字里的{star}在哪一支"),
}
TRAIN_SCHOOL = {"扶抑": ("按扶抑法，这个八字喜哪些十神", "用扶抑的思路看，日主需要哪几类五行帮扶或克泄"),
                "调候": ("按调候的思路，这个八字需要哪种五行来调节寒暖", "从调候看，出生季节需要什么五行"),
                "格局": ("按格局派的说法，格局用神是哪个十神，要顺用还是逆用", "以格局论，用神该顺用还是逆用")}  # fmt: skip
TRAIN_DOMAIN = {"marriage": ("按传统说法，看{d}要看哪些十神和宫位，这个盘里怎么样", "{d}方面传统上看什么，这个八字里的情况如何"),
                "career": ("按传统说法，{d}看哪些十神，这个盘里有没有", "{d}运传统上怎么看，盘里是什么情况"),
                "wealth": ("按传统说法，{d}看哪些十神，盘里情况如何", "{d}传统上看哪几个十神，在这个八字里出现了吗")}  # fmt: skip
TRAIN_EVENTS = (
    "哪一年能遇到正缘",
    "什么时候能买房",
    "多少岁会转运",
    "以后会不会大富大贵",
    "哪年适合跳槽",
    "什么时候会有孩子",
)


def describe_train(cls, pillars, rng):
    """(intent, description, need): the same intents and filters as the test writer, other words."""
    intent, _, need = GB.describe(cls, pillars, rng)
    dm = pillars["day"][0]
    if cls in ("ten_god_stem", "ten_god_branch", "hidden", "life_stage"):
        pos = intent.split(":")[2]
        desc = rng.choice(TRAIN_DESC[cls]).format(
            p=POS[pos], s=pillars[pos][0], b=pillars[pos][1], dm=dm
        )
    elif cls == "yongshen":
        desc = rng.choice(TRAIN_SCHOOL[intent.split(":")[1]])
    elif cls == "flow":
        desc = rng.choice(TRAIN_DESC["flow"]).format(year=intent.split(":")[1])
    elif cls == "star":
        desc = rng.choice(TRAIN_DESC["star"]).format(star=intent.split(":")[1])
    elif cls == "prediction":
        desc = rng.choice(TRAIN_EVENTS)
    elif cls == "domain":
        domain = intent.split(":")[1]
        desc = rng.choice(TRAIN_DOMAIN[domain]).format(d=GB.DOMAINS[domain])
    else:
        desc = rng.choice(TRAIN_DESC[cls])
    return intent, desc, need


def charts_for(split, seed, count):
    if split.startswith("train"):
        years = sorted(TD.split_years()["train"])
        blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
        rng = random.Random(f"p6-train-charts:{seed}")
        out = []
        while len(out) < count:
            day = date(rng.choice(years), 1, 1) + timedelta(days=rng.randrange(365))
            if day in blocked or day.year not in years:
                continue
            c = eight_char(
                datetime(day.year, day.month, day.day, rng.randrange(24), rng.randrange(60))
            )
            out.append(({"year": c.getYear(), "month": c.getMonth(), "day": c.getDay(), "hour": c.getTime()},
                        rng.choice(("male", "female"))))  # fmt: skip
        return out
    sys.path.insert(0, "scripts/phase6")
    import build_eval6

    exclude = S.earlier_eval_charts() if split != "dev6" else set()
    return build_eval6.block_charts(split, seed, count, exclude)


def generate(split):
    from bazireason.mlx_backend import MLXChatModel

    seed, per_class = PLAN[split]
    rng = random.Random(f"blind6:{seed}")
    train = split.startswith("train")
    slots = []
    for index, (pillars, gender) in enumerate(charts_for(split, seed, per_class * len(CLASSES))):
        cls = CLASSES[index % len(CLASSES)]
        intent, desc, need = (describe_train if train else GB.describe)(cls, pillars, rng)
        slots.append({"slot": index, "cls": cls, "intent": intent, "desc": desc, "need": need, "pillars": pillars,
                      "gender": gender, "persona": rng.randrange(len(PERSONAS)), "ask": rng.randrange(len(ASK_TRAIN)),
                      "owner": rng.choice(OWNERS), "done": False})  # fmt: skip
    writer = MLXChatModel(TRAIN_WRITER if train else TEST_WRITER, None, label="writer")
    log = []
    for attempt in range(4):
        todo = [s for s in slots if not s["done"]]
        if not todo:
            break
        convs = []
        for s in todo:
            chart = " ".join(
                f"{POS[n]}柱{s['pillars'][n]}" for n in ("year", "month", "day", "hour")
            )
            sex = "男" if s["gender"] == "male" else "女"
            if train:
                system = PERSONAS[s["persona"]]
                user = ASK_TRAIN[s["ask"]].format(
                    owner=s["owner"], chart=chart, sex=sex + "命", desc=s["desc"]
                )
            else:
                system, user = GB.SYSTEM, GB.ASK.format(chart=chart, sex=sex, desc=s["desc"])
            convs.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
        texts, _ = writer.generate_batch(
            convs, 80, temperature=0.8, top_p=0.9, seed=seed + attempt, batch=32
        )
        for s, raw in zip(todo, texts, strict=True):
            question = GB.clean(raw)
            plan = N.plan_for(s["intent"], s["pillars"], s["gender"])
            ok, reason = GB.passes(question, s["need"], plan)
            s["done"] = ok
            log.append({"slot": s["slot"], "attempt": attempt, "cls": s["cls"], "intent": s["intent"],
                        "pillars": s["pillars"], "gender": s["gender"], "description": s["desc"],
                        "raw": raw, "question": question, "accepted": ok, "reason": reason})  # fmt: skip
    return log


def build(split, log):
    tag = {"dev6": "D", "valid6": "V", "sealed6": "S", "train6": "T", "train6b": "B"}[split]
    writer = Path(TRAIN_WRITER if split.startswith("train") else TEST_WRITER).name
    items = []
    for record in log:
        if not record["accepted"]:
            continue
        plan = N.plan_for(record["intent"], record["pillars"], record["gender"])
        items.append(make_item(f"P6{tag}-blind-{len(items) + 1:05d}", split, f"blind:{writer}", record["pillars"],
                               record["gender"], record["question"], record["intent"], plan, f"blind-{record['slot']}"))  # fmt: skip
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["dev6", "valid6", "sealed6", "train6"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        log_path, items_path = OUT / f"{split}_attempts.jsonl", OUT / f"{split}.jsonl"
        if args.check:
            if not log_path.exists():
                continue
            log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines()]
            data = "".join(
                json.dumps(i, ensure_ascii=False) + "\n" for i in build(split, log)
            ).encode()
            print(f"{items_path}: {'ok' if items_path.read_bytes() == data else 'MISMATCH'}")
            continue
        if items_path.exists():
            print(f"skip {items_path} (exists)")
            continue
        log = generate(split)
        log_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log), encoding="utf-8"
        )
        items = build(split, log)
        data = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items).encode()
        items_path.write_bytes(data)
        print(json.dumps({"split": split, "items": len(items), "slots": PLAN[split][1] * len(CLASSES),
                          "sha256": hashlib.sha256(data).hexdigest()[:16]}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
