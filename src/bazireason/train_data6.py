"""Phase 6 training corpora: one generator, several mixtures that differ only in what is tested.

Mixtures (``build(recipe, tokens, seed)``), all trained from the untuned base:

  u0  the union of every earlier training file, sampled in proportion: the same data as the
      sequential chain, in one shuffled mixture (H1: joint against sequential)
  s1  procedures rewritten in small steps (bazireason.steps6) on fresh charts, template
      wordings; fresh calendar rows; a fixed share of the earlier "other" tasks (H2)
  s2  s1 plus natural wordings written by the untuned base model (data/phase6/blind/train6),
      other chart formats, and contrast rows (two procedures on one chart; one-character
      chart edits) (H3)
  s3  s2 plus evidence use (raw facts or rule text in the prompt, never the answer), checking
      a supplied reference conclusion by deriving it, self-check rows built from a model's own
      replies, and calendar decisions balanced as matched twins (H4)
  u1  the best recipe at a larger budget

Every label comes from the rule code or the calendar engine. No evaluation chart, wording or
check request is used (tests/test_phase6.py). Charts come from the training years.
"""

import hashlib
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from bazireason import compose6 as C6
from bazireason import deep_tasks as K
from bazireason import ladder as LD
from bazireason import rules as R
from bazireason import steps6 as S6
from bazireason import textstyle
from bazireason import train_data as TD
from bazireason.conventions import CYCLE
from bazireason.indep_rules import eight_char
from bazireason.intent import plan_for
from bazireason.p2_prompts import P2_SYSTEM
from bazireason.p5_eval import head_line
from bazireason.prompts6 import swap
from bazireason.splits6 import POS_ZH, neighbours

VERSION = "p6_train_v1"
OLD_FILES = ("data/phase2/train_v1", "data/phase2/train_v2", "data/phase3/train_v1", "data/phase3/train_v2",
             "data/phase3/train_v3", "data/phase4/train_v1", "data/phase5/train_v1")  # fmt: skip
OTHER_FAMILIES = ("claims", "mcq3", "dialog", "strict_audit", "replay_mcq", "mcq")
CALENDAR_DROP = ("rule", "mcq")  # the Phase 2 rule rows are replaced by small-step facts
TOKENS_PER_CHAR = 0.675  # measured: Qwen3 chat template, 300 old and 300 new rows (0.677, 0.672)

# Intent weights (rows) within the procedure part.
WEIGHTS = {
    "fact:xun_kong": 4, "fact:element_count": 3, "fact:missing_element": 2, "fact:life_stage": 4,
    "fact:ten_god_stem": 3, "fact:ten_god_branch": 3, "fact:hidden": 2, "fact:luck_direction": 2,
    "gods": 7, "strength": 7, "pattern": 8, "yongshen": 6, "interact": 9, "flow": 11, "domain": 7,
    "overview": 3, "ambig:yongshen_open": 2, "ambig:no_rule": 2, "star": 2, "prediction": 2,
}  # fmt: skip
# Training-only fact wordings (the evaluation uses other wordings; checked by a test).
FACT_WORDS = {
    "xun_kong": ("按日柱所在的旬，这个八字哪两个地支落空？", "请推出这个八字的空亡。", "日柱的旬空是什么？请一步步数出来。"),
    "element_count": ("请统计八个字的五行分布（藏干不计）。", "四柱八字中木火土金水各几个？只算明见的八个字。"),
    "missing_element": ("只看八个字，五行有没有缺的？", "这个八字五行全不全？缺哪一行？（不算藏干）"),
    "life_stage": ("日主落在{p}支，按十二长生是哪一宫？", "请推出日主在{p}支的十二长生状态。"),
    "ten_god_stem": ("{p}干相对日主是哪一种十神？", "请推出{p}干的十神。"),
    "ten_god_branch": ("{p}支本气对日主来说是什么十神？", "以{p}支的本气论，它是日主的哪个十神？"),
    "hidden": ("{p}支藏有哪些天干？", "请写出{p}支的藏干。"),
    "luck_direction": ("按年干阴阳和性别，大运顺行还是逆行？", "这个八字的大运该顺排还是逆排？"),
}  # fmt: skip
PREDICTIONS = (
    "我哪一年能升职？",
    "我什么时候会发财？",
    "我这辈子会离婚吗？",
    "我多少岁会结婚？",
    "我哪年会生病？",
    "我以后会不会很有钱？",
)
STYLES = (
    "八字：{y}年 {m}月 {d}日 {h}时（日主{dm}）{sex}。",
    "我的八字是年{y}、月{m}、日{d}、时{h}{sex2}。",
)
CHECK_REQUESTS = ("帮我把上面每一步再核对一遍，有错就改。", "你确定吗？请逐步复查一次。", "再检查一下你的推导，必要时更正结论。",
                  "请复核上面的步骤和结论。")  # fmt: skip
CONTRAST = (("strength", "yongshen:扶抑"), ("pattern", "gods"), ("fact:life_stage:day", "fact:life_stage:month"),
            ("fact:ten_god_stem:month", "fact:ten_god_branch:month"), ("flow", "fact:xun_kong"), ("interact", "flow"),
            ("fact:element_count", "strength"), ("pattern", "yongshen:格局"))  # fmt: skip


# The main system prompt of the rows being built (round 2 builds with the short one).
_SYSTEM = [P2_SYSTEM]
# Years of flow-year and area-of-life questions (preregistration 12.11). "fixed": flow years
# 2024-2049 and first years 2024-2045, as in every set up to U3v. "life": years of the chart's
# life, as the Phase 3 generator draws them (train_data3): flow years 8 to 69 years after the
# birth year (at most 2160), first years 16 to 49 years after it (at most 2150).
_YEARS = ["fixed"]
_BIRTH = {}  # chart key -> birth year of its latest draw (Charts.draw); no random draw is added


def _sys(r):
    r["messages"] = swap(r["messages"], _SYSTEM[0])
    return r


def chart_of(r):
    """Chart key of a row with a four-pillar chart in its meta, else None."""
    pillars = (r.get("meta") or {}).get("pillars")
    if isinstance(pillars, dict) and len(pillars) == 4:
        return "".join(pillars[k] for k in ("year", "month", "day", "hour"))
    return None


def head_of(intent):
    """The WEIGHTS key (question type) of a concrete intent, or None."""
    parts = intent.split(":")
    for k in (2, 1):
        key = ":".join(parts[:k])
        if key in WEIGHTS:
            return key
    return None


def rows_of(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


_TOKENIZER = []


def _tokenizer():
    if not _TOKENIZER:
        try:
            from tokenizers import Tokenizer

            _TOKENIZER.append(
                Tokenizer.from_file("models/qwen3-4b-instruct-2507-mlx-4bit/tokenizer.json")
            )
        except (ImportError, OSError):
            _TOKENIZER.append(None)
    return _TOKENIZER[0]


def tokens(row) -> int:
    """Sequence tokens of a chat row (exact with the model's tokenizer, else estimated)."""
    tok = _tokenizer()
    if tok is None:
        return int(sum(len(m["content"]) for m in row["messages"]) * TOKENS_PER_CHAR)
    if "_tokens" not in row:
        row["_tokens"] = sum(len(tok.encode(m["content"]).ids) + 5 for m in row["messages"])
    return row["_tokens"]


# ---- charts ----------------------------------------------------------------------------------------


class Charts:
    """Random charts from real dates in the training years, never an evaluation chart."""

    def __init__(self, seed, exclude):
        self.rng = random.Random(f"p6-train-charts:{seed}")
        self.years = sorted(TD.split_years()["train"])
        self.blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
        self.exclude = exclude

    def draw(self):
        while True:
            day = date(self.rng.choice(self.years), 1, 1) + timedelta(days=self.rng.randrange(365))
            if day in self.blocked or day.year not in self.years:
                continue
            c = eight_char(
                datetime(
                    day.year, day.month, day.day, self.rng.randrange(24), self.rng.randrange(60)
                )
            )
            pillars = {
                "year": c.getYear(),
                "month": c.getMonth(),
                "day": c.getDay(),
                "hour": c.getTime(),
            }
            if "".join(pillars.values()) not in self.exclude:
                _BIRTH["".join(pillars.values())] = day.year
                return pillars, self.rng.choice(("male", "female"))


# ---- procedure rows -----------------------------------------------------------------------------------


def life_year(pillars, rng, low, high, cap):
    """A year of the chart's life: ``low`` to ``high`` - 1 years after its birth year, at most
    ``cap``. Every caller draws the chart from ``Charts`` first, so its birth year is known."""
    return min(_BIRTH["".join(pillars.values())] + rng.randrange(low, high), cap)


def concrete_intent(head, pillars, rng):
    if head in ("fact:life_stage", "fact:hidden", "fact:ten_god_branch"):
        return f"{head}:{rng.choice(('year', 'month', 'day', 'hour'))}"
    if head == "fact:ten_god_stem":
        return f"{head}:{rng.choice(('year', 'month', 'hour'))}"
    if head == "yongshen":
        return f"yongshen:{rng.choice(('扶抑', '调候', '格局'))}"
    if head == "flow":
        if _YEARS[0] == "life":
            return f"flow:{life_year(pillars, rng, 8, 70, 2160)}"
        return f"flow:{rng.randrange(2024, 2050)}"
    if head == "domain":
        area = rng.choice(("marriage", "career", "wealth", "study", "children"))
        if _YEARS[0] == "life":
            return f"domain:{area}:{life_year(pillars, rng, 16, 50, 2150)}"
        return f"domain:{area}:{rng.randrange(2024, 2046)}"
    if head == "star":
        return f"star:{rng.choice(('桃花', '驿马', '华盖'))}"
    return head


def template_question(intent, pillars, gender, rng):
    """A training wording: deep_tasks variants 0 and 1 (variant 2 is held out) or FACT_WORDS."""
    head, _, rest = intent.partition(":")
    variant = rng.choice((0, 1))
    if head == "fact":
        kind, _, pos = rest.partition(":")
        return rng.choice(FACT_WORDS[kind]).format(p=POS_ZH.get(pos, ""))
    if head == "prediction":
        return rng.choice(PREDICTIONS)
    if head == "star":
        return rng.choice((f"这个八字的{rest}在哪一支？", f"请查这个八字的{rest}落在哪个地支。"))
    if head == "ambig":
        return K.ambig_task(rest, pillars, rng, variant, gender or "male")[1]
    builders = {"gods": lambda: K.gods_task(pillars, rng, variant), "strength": lambda: K.strength_task(pillars, rng, variant),
                "pattern": lambda: K.pattern_task(pillars, rng, variant), "interact": lambda: K.interact_task(pillars, rng, variant),
                "overview": lambda: K.overview_task(pillars, rng, variant, gender or "male"),
                "yongshen": lambda: K.yongshen_task(pillars, rng, variant, rest),
                "flow": lambda: K.flow_task(pillars, rng, variant, int(rest or 2030)),
                "domain": lambda: K.domain_task(pillars, rng, variant, gender or "male", rest.split(":")[0], int(rest.split(":")[1]))}  # fmt: skip
    textstyle.set_version(4)
    try:
        return builders[head]()[0]
    finally:
        textstyle.set_version(1)


def chart_head(pillars, gender, rng, styled):
    if not styled or rng.random() < 0.6:
        return head_line(pillars, gender)
    sex = {"male": "，男命", "female": "，女命"}.get(gender, "")
    return rng.choice(STYLES).format(y=pillars["year"], m=pillars["month"], d=pillars["day"], h=pillars["hour"],
                                     dm=pillars["day"][0], sex=sex, sex2=sex)  # fmt: skip


def row(family, messages, meta):
    return _sys({"family": family, "task": f"p6_{family}", "messages": messages, "meta": meta})


def procedure(pillars, gender, intent, question, rng, styled=False, prefix=""):
    out = S6.derive(intent, pillars, gender)
    if out is None:
        return None
    text, reference = out
    hint = plan_for(intent, pillars, gender)["hint"]
    user = prefix + chart_head(pillars, gender, rng, styled) + question + "\n" + hint
    return row("procedure", [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user},
                             {"role": "assistant", "content": text}],
               {"intent": intent, "pillars": pillars, "gender": gender, "question": question, "expected": reference})  # fmt: skip


def weighted_heads(rng, n):
    heads = list(WEIGHTS)
    return rng.choices(heads, weights=[WEIGHTS[h] for h in heads], k=n)


def weighted_procedure_rows(target_tokens, charts, rng, paraphrases, para_share, styled, weights):
    """Round 2 (pre-registration 10.2): the question type is drawn first, by its recipe weight
    times ``weights[type]``, for template and natural wordings alike. When a type's unused
    natural questions run out, template questions on fresh charts replace them."""
    pools = {}
    for item in paraphrases or []:
        pools.setdefault(head_of(item["meta"]["intent"]), []).append(item)
    for pool in pools.values():
        rng.shuffle(pool)
    heads = list(WEIGHTS)
    w = [WEIGHTS[h] * weights[h] for h in heads]
    out, used = [], 0
    while used < target_tokens:
        head = rng.choices(heads, weights=w, k=1)[0]
        pool = pools.get(head)
        if pool and rng.random() < para_share:
            meta = pool.pop()["meta"]
            r = procedure(
                meta["pillars"], meta["gender"], meta["intent"], meta["question"], rng, styled
            )
            source = "paraphrase"
        else:
            pillars, gender = charts.draw()
            intent = concrete_intent(head, pillars, rng)
            question = template_question(intent, pillars, gender, rng)
            r = procedure(pillars, gender, intent, question, rng, styled)
            source = "template"
        if r:
            r["meta"]["source"] = source
            out.append(r)
            used += tokens(r)
    return out


def fit_sample_b(seed, per_type, exclude):
    """Round 2 fit sample: ``per_type`` questions for each question type, in training template
    wordings on fresh training-year charts. Never training input (its charts are excluded
    from every round-2 training set)."""
    rng = random.Random(f"fit-b:{seed}")
    charts = Charts(seed, exclude)
    out = []
    for head in WEIGHTS:
        got = 0
        while got < per_type:
            pillars, gender = charts.draw()
            intent = concrete_intent(head, pillars, rng)
            r = procedure(
                pillars, gender, intent, template_question(intent, pillars, gender, rng), rng
            )
            if r:
                r["meta"]["source"] = "template"
                r["meta"]["head"] = head
                out.append(r)
                got += 1
    return out


def procedure_rows(
    target_tokens, charts, rng, paraphrases=None, para_share=0.0, styled=False, weights=None
):
    if weights is not None:
        return weighted_procedure_rows(
            target_tokens, charts, rng, paraphrases, para_share, styled, weights
        )
    out, used = [], 0
    pool = list(paraphrases or [])
    rng.shuffle(pool)
    while used < target_tokens:
        if pool and rng.random() < para_share:
            item = pool.pop()
            meta = item["meta"]
            r = procedure(
                meta["pillars"], meta["gender"], meta["intent"], meta["question"], rng, styled
            )
            if r:
                r["meta"]["source"] = "paraphrase"
        else:
            pillars, gender = charts.draw()
            intent = concrete_intent(weighted_heads(rng, 1)[0], pillars, rng)
            r = procedure(
                pillars,
                gender,
                intent,
                template_question(intent, pillars, gender, rng),
                rng,
                styled,
            )
            if r:
                r["meta"]["source"] = "template"
        if r:
            out.append(r)
            used += tokens(r)
    return out


def compose_rows(target_tokens, charts, rng):
    """Composition rows (preregistration 12.7): two or three trained procedures joined in one
    question, never one of the held-out combinations."""
    out, used = [], 0
    while used < target_tokens:
        pillars, gender = charts.draw()
        kind = rng.choice(C6.KINDS)
        question, hint, text, reference = C6.build(kind, pillars, gender, rng)
        user = chart_head(pillars, gender, rng, True) + question + "\n" + hint
        r = row("compose", [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user},
                            {"role": "assistant", "content": text}],
                {"intent": f"compose:{kind}", "pillars": pillars, "gender": gender, "question": question,
                 "expected": reference})  # fmt: skip
        out.append(r)
        used += tokens(r)
    return out


def contrast_rows(target_tokens, charts, rng):
    """Two confusable procedures on one chart (two turns), and one-character chart edits."""
    out, used = [], 0
    while used < target_tokens:
        pillars, gender = charts.draw()
        if rng.random() < 0.6:
            a, b = (concrete_intent(h, pillars, rng) for h in rng.choice(CONTRAST))
            first = procedure(pillars, gender, a, template_question(a, pillars, gender, rng), rng)
            second = S6.derive(b, pillars, gender)
            if not first or not second:
                continue
            q2 = (
                "同一个八字，"
                + template_question(b, pillars, gender, rng)
                + "\n"
                + plan_for(b, pillars, gender)["hint"]
            )
            r = row("contrast", first["messages"] + [{"role": "user", "content": q2}, {"role": "assistant", "content": second[0]}],
                    {"intent": b, "first_intent": a, "pillars": pillars, "gender": gender})  # fmt: skip
        else:
            head = concrete_intent(weighted_heads(rng, 1)[0], pillars, rng)
            if head.startswith(("ambig", "prediction", "star")):
                continue
            base = S6.derive(head, pillars, gender)
            options = [
                (p, n) for p in ("year", "month", "day", "hour") for n in neighbours(pillars[p])
            ]
            rng.shuffle(options)
            # An edited chart must not be a held-out chart either (an evaluation chart or, in
            # round 2, a chart of fit sample B): the edit of a real chart is often a real chart.
            edited = next((dict(pillars, **{p: n}) for p, n in options[:12]
                           if "".join(dict(pillars, **{p: n}).values()) not in charts.exclude
                           and S6.derive(head, dict(pillars, **{p: n}), gender) and S6.derive(head, dict(pillars, **{p: n}), gender)[1] != base[1]), None)  # fmt: skip
            if not base or not edited:
                continue
            q = template_question(head, pillars, gender, rng)
            for chart in (pillars, edited):
                r = procedure(chart, gender, head, q, rng)
                if r:
                    r["family"], r["task"] = "contrast", "p6_contrast"
                    out.append(r)
                    used += tokens(r)
            continue
        out.append(r)
        used += tokens(r)
    return out


# ---- evidence, reference checking, self-check ------------------------------------------------------------


def corrupt_sheet(sheet, pillars, rng):
    """One wrong hidden-stem list in a raw-fact sheet, and the note that names it."""
    pos = rng.choice(("year", "month", "day", "hour"))
    b = pillars[pos][1]
    right = "、".join(R.HIDDEN_STEMS[b])
    others = [x for x in R.HIDDEN_STEMS if "、".join(R.HIDDEN_STEMS[x]) != right]
    wrong = "、".join(R.HIDDEN_STEMS[rng.choice(others)])
    bad = sheet.replace(f"{POS_ZH[pos]}支{b}（{R.BRANCH_ELEMENT[b]}，{'阳' if R.is_yang(b) else '阴'}；藏{right}）",
                        f"{POS_ZH[pos]}支{b}（{R.BRANCH_ELEMENT[b]}，{'阳' if R.is_yang(b) else '阴'}；藏{wrong}）")  # fmt: skip
    note = f"注意：给出的事实里写{POS_ZH[pos]}支{b}藏{wrong}，与藏干表不符，{b}应藏{right}；下面按藏干表推导。\n"
    return (bad, note) if bad != sheet else (sheet, "")


def evidence_rows(target_tokens, charts, rng):
    out, used = [], 0
    while used < target_tokens:
        pillars, gender = charts.draw()
        intent = concrete_intent(weighted_heads(rng, 1)[0], pillars, rng)
        mode = rng.choices(("raw", "rule", "corrupt"), weights=(5, 4, 1))[0]
        if mode == "rule":
            prefix, note = LD.rule_text(intent), ""
        else:
            sheet = LD.raw_sheet(pillars, gender, intent)
            prefix, note = (
                corrupt_sheet(sheet, pillars, rng)
                if mode == "corrupt" and not intent.startswith("fact:hidden")
                else (sheet, "")
            )
        r = procedure(
            pillars,
            gender,
            intent,
            template_question(intent, pillars, gender, rng),
            rng,
            prefix=prefix,
        )
        if not r:
            continue
        if note:
            r["messages"][-1]["content"] = note + r["messages"][-1]["content"]
        r["family"], r["task"] = f"evidence_{mode}", f"p6_evidence_{mode}"
        out.append(r)
        used += tokens(r)
    return out


def verify_rows(target_tokens, charts, rng):
    """A supplied reference conclusion, right or wrong: derive, compare, keep the derivation."""
    out, used = [], 0
    while used < target_tokens:
        pillars, gender = charts.draw()
        intent = concrete_intent(weighted_heads(rng, 1)[0], pillars, rng)
        if intent.startswith(("ambig", "prediction", "star")):
            continue
        derived = S6.derive(intent, pillars, gender)
        if not derived:
            continue
        item = {
            "family": "x",
            "meta": {
                "intent": intent,
                "expected": derived[1],
                "pillars": pillars,
                "gender": gender,
            },
        }
        wrong = LD.wrong_reference(item, rng) if rng.random() < 0.5 else None
        shown = wrong if wrong is not None else derived[1]
        block = "[参考结论] " + json.dumps(shown, ensure_ascii=False, separators=(",", ":")) + "\n"
        r = procedure(
            pillars,
            gender,
            intent,
            template_question(intent, pillars, gender, rng),
            rng,
            prefix=block,
        )
        if not r:
            continue
        body, _, conclusion = r["messages"][-1]["content"].rpartition("\n结论JSON:")
        verdict = (
            "与参考结论一致。"
            if wrong is None
            else "与参考结论不一致：参考结论有误，以上面的推导为准。"
        )
        r["messages"][-1]["content"] = f"{body}\n比对：{verdict}\n结论JSON:{conclusion}"
        r["family"], r["task"] = "verify_reference", "p6_verify_reference"
        out.append(r)
        used += tokens(r)
    return out


def selfcheck_rows(replies, rng, limit_tokens):
    """From a model's own replies (on training charts): find the first wrong step, or confirm.

    ``replies``: list of (procedure row, reply text, correct). The row's derivation is the code's.
    """
    out, used = [], 0
    for r, reply, correct in replies:
        if used >= limit_tokens:
            break
        truth = r["messages"][-1]["content"]
        request = rng.choice(CHECK_REQUESTS)
        if correct:
            target = (
                "逐步核对：每一步都与规则一致，没有发现错误，结论不变。\n"
                + truth.rpartition("\n")[2]
            )
        else:
            mine = [x.strip() for x in reply.split("\n") if x.strip()]
            good = [x.strip() for x in truth.split("\n") if x.strip()]
            k = next(
                (i for i, (a, b) in enumerate(zip(mine, good, strict=False)) if a != b),
                min(len(mine), len(good)),
            )
            if k < len(mine) and k < len(good) and not good[k].startswith("结论JSON"):
                a, b = mine[k], good[k]
                c = next(
                    (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                    min(len(a), len(b)),
                )
                lo = max(0, c - 20)
                where = (
                    f"第 {k + 1} 行有误：写成了「…{a[lo : c + 30]}…」，应为「…{b[lo : c + 30]}…」。"
                )
            else:
                where = "推导的结构或结论有误。"
            target = f"逐步核对：{where}改正后的完整推导如下：\n{truth}"
        messages = [*r["messages"][:2], {"role": "assistant", "content": reply},
                    {"role": "user", "content": request}, {"role": "assistant", "content": target}]  # fmt: skip
        out.append(row("selfcheck", messages, dict(r["meta"], first_correct=correct)))
        used += tokens(out[-1])
    return out


# ---- calendar and the other tasks ------------------------------------------------------------------------


def first_user(r):
    return next((m["content"] for m in r["messages"] if m["role"] == "user"), "")


def calendar_cell(r):
    """A calendar request's cell: its family and expected status, or None for other tasks."""
    expected = (r.get("meta") or {}).get("expected")
    if r.get("task") != "chart" or not isinstance(expected, dict):
        return None
    return f"{r['family']}:{expected.get('status')}"


def fit_sample_c(seed, per_cell):
    """Round 2 calendar fit sample (preregistration 12.5): up to ``per_cell`` fresh requests for
    each cell of the training generator. Never training input."""
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    pool = [r for r in TD.generate_split("train", seed, blocked, 2.0, 2) if calendar_cell(r)]
    random.Random(f"fit-c:{seed}").shuffle(pool)
    taken, out = Counter(), []
    for r in pool:
        cell = calendar_cell(r)
        if taken[cell] < per_cell:
            taken[cell] += 1
            out.append(r)
    return sorted(out, key=lambda r: (calendar_cell(r), r["id"]))


def calendar_rows(
    target_tokens, seed, balanced, weights=None, block_texts=frozenset(), held_out=frozenset()
):
    """Calendar rows of the training generator. Its dialogue rows carry a chart: one on a
    held-out chart (``held_out``: evaluation charts and, in round 2, fit sample B) is left out.

    The generator's pool has a fixed size. When it cannot fill the budget (a short system
    prompt holds more rows in the same tokens), a pool twice as large is drawn instead; the
    first pool is the one of every earlier set, so those rebuild bit for bit."""
    scales = (4.0, 8.0, 16.0) if weights is not None else (2.0, 4.0, 8.0, 16.0)
    for scale in scales:
        out, used = _calendar_rows(
            target_tokens, seed, balanced, weights, block_texts, held_out, scale
        )
        if used >= target_tokens:
            break
    return out


def _calendar_rows(target_tokens, seed, balanced, weights, block_texts, held_out, scale):
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    if weights is not None:
        # Preregistration 12.5: a larger pool, drawn with the cells' weights (a row's key is
        # u ** (1 / weight); the largest keys come first), never a fit-sample request.
        pool = [
            _sys(r)
            for r in TD.generate_split("train", seed, blocked, scale, 2)
            if r["family"] not in CALENDAR_DROP
            and first_user(r) not in block_texts
            and chart_of(r) not in held_out
        ]
        rng = random.Random(f"p6-cal:{seed}")
        keyed = [
            (rng.random() ** (1.0 / weights.get(calendar_cell(r), 1.0)), i)
            for i, r in enumerate(pool)
        ]
        out, used = [], 0
        for _, i in sorted(keyed, reverse=True):
            if used >= target_tokens:
                break
            out.append(pool[i])
            used += tokens(pool[i])
        return out, used
    pool = [
        _sys(r)
        for r in TD.generate_split("train", seed, blocked, scale, 2)
        if r["family"] not in CALENDAR_DROP
    ]
    rng = random.Random(f"p6-cal:{seed}")
    rng.shuffle(pool)
    if balanced:
        # Matched twins: within each decision family take equal numbers of each expected status.
        by = {}
        for r in pool:
            status = (
                (r["meta"].get("expected") or {}).get("status")
                if isinstance(r["meta"].get("expected"), dict)
                else None
            )
            by.setdefault((r["family"], status), []).append(r)
        families = {}
        for (fam, status), rs in by.items():
            families.setdefault(fam, {})[status] = rs
        pool = []
        for groups in families.values():
            if len(groups) > 1 and None not in groups:
                k = min(len(v) for v in groups.values())
                for rs in groups.values():
                    pool += rs[:k]
            else:
                for rs in groups.values():
                    pool += rs
        rng.shuffle(pool)
    out, used = [], 0
    for r in pool:
        if used >= target_tokens:
            break
        if chart_of(r) in held_out:  # skipped when chosen, so the shuffle above is unchanged
            continue
        out.append(r)
        used += tokens(r)
    return out, used


def other_rows(target_tokens, rng, block=frozenset()):
    pool = [
        _sys(r)
        for path in OLD_FILES[2:6]
        for r in rows_of(f"{path}/train.jsonl")
        if r["family"] in OTHER_FAMILIES and chart_of(r) not in block
    ]
    rng.shuffle(pool)
    out, used = [], 0
    for r in pool:
        if used >= target_tokens:
            break
        out.append(r)
        used += tokens(r)
    return out


def union_rows(target_tokens, rng):
    pool = [r for path in OLD_FILES for r in rows_of(f"{path}/train.jsonl")]
    rng.shuffle(pool)
    out, used = [], 0
    for r in pool:
        if used >= target_tokens:
            break
        out.append(r)
        used += tokens(r)
    return out


# ---- recipes -------------------------------------------------------------------------------------------

SHARES = {
    "s1": {"calendar": 0.30, "procedure": 0.58, "other": 0.12},
    "s2": {"calendar": 0.30, "procedure": 0.50, "contrast": 0.08, "other": 0.12},
    "s3": {"calendar": 0.28, "procedure": 0.40, "contrast": 0.06, "evidence": 0.09, "verify": 0.03, "selfcheck": 0.06, "other": 0.08},
}  # fmt: skip


def build(
    recipe,
    total_tokens,
    seed,
    exclude=frozenset(),
    paraphrases=None,
    replies=None,
    calendar_share=None,
    drop=(),
    system=None,
    weights=None,
    block=frozenset(),
    steps=1,
    calendar_weights=None,
    calendar_block=frozenset(),
    compose_share=0.0,
    years="fixed",
):
    """Rows of one recipe. ``calendar_share`` and ``drop`` (component names) apply the
    pre-registered screening decisions (docs/phase6_preregistration.md, section 5) to U1.

    Round 2 (section 10): ``system`` replaces the main system prompt, ``weights`` reweights the
    question types of the procedure rows, no row has a chart in ``block`` (fit sample B), and
    ``steps`` is the derivation version (bazireason.steps6). ``calendar_weights`` draws the
    calendar rows by cell weight, never a request of fit sample C (``calendar_block``).
    ``compose_share`` of the tokens are composition rows, taken from the procedure share.
    ``years`` sets the years of flow-year and area-of-life questions (``_YEARS``, 12.11)."""
    if years not in ("fixed", "life"):
        raise ValueError(f"years: {years}")
    _SYSTEM[0] = system or P2_SYSTEM
    S6.set_steps(steps)
    _YEARS[0] = years
    try:
        return _build(
            recipe, total_tokens, seed, exclude, paraphrases, replies, calendar_share, drop,
            weights, frozenset(block), calendar_weights, frozenset(calendar_block), compose_share,
        )  # fmt: skip
    finally:
        _SYSTEM[0] = P2_SYSTEM
        S6.set_steps(1)
        _YEARS[0] = "fixed"


def _build(
    recipe,
    total_tokens,
    seed,
    exclude,
    paraphrases,
    replies,
    calendar_share,
    drop,
    weights,
    block,
    calendar_weights,
    calendar_block,
    compose_share,
):
    rng = random.Random(f"{VERSION}:{recipe}:{seed}")
    if block:
        exclude = frozenset(exclude) | block
        paraphrases = [p for p in paraphrases or [] if chart_of(p) not in block]
        replies = [x for x in replies or [] if chart_of(x[0]) not in block]
    if recipe == "u0":
        rows = union_rows(total_tokens, rng)
    else:
        base = recipe if recipe in SHARES else "s3"
        shares = dict(SHARES[base])
        if calendar_share is not None:
            scale = (1 - calendar_share) / (1 - shares["calendar"])
            shares = {
                k: (calendar_share if k == "calendar" else v * scale) for k, v in shares.items()
            }
        for part in drop:  # a dropped component's share goes to the procedure rows
            shares["procedure"] += shares.pop(part, 0.0)
        if compose_share:  # composition rows take their share from the procedure rows
            shares["procedure"] -= compose_share
        charts = Charts(seed, exclude)
        rows = []
        rows += calendar_rows(
            total_tokens * shares["calendar"],
            seed + 1,
            balanced=base == "s3" and "twins" not in drop,
            weights=calendar_weights,
            block_texts=calendar_block,
            held_out=exclude,
        )
        styled = base in ("s2", "s3") and "wording" not in drop
        rows += procedure_rows(total_tokens * shares["procedure"], charts, rng, paraphrases if styled else None,
                               0.45 if styled else 0.0, styled, weights)  # fmt: skip
        if "contrast" in shares:
            rows += contrast_rows(total_tokens * shares["contrast"], charts, rng)
        if "evidence" in shares:
            rows += evidence_rows(total_tokens * shares["evidence"], charts, rng)
        if "verify" in shares:
            rows += verify_rows(total_tokens * shares["verify"], charts, rng)
        if "selfcheck" in shares:
            rows += selfcheck_rows(replies or [], rng, total_tokens * shares["selfcheck"])
        if compose_share:  # its own charts and random stream: the other parts are drawn as before
            rows += compose_rows(
                total_tokens * compose_share,
                Charts(seed + 77, exclude),
                random.Random(f"{VERSION}:compose:{seed}"),
            )
        rows += other_rows(total_tokens * shares["other"], rng, block)
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = f"P6{recipe.upper()}-{i:05d}"
        r["dataset_version"] = f"{VERSION}:{recipe}"
    return rows


def summary(rows) -> dict:
    fam = Counter(r["family"] for r in rows)
    tok = Counter()
    for r in rows:
        tok[r["family"]] += tokens(r)
    return {"rows": len(rows), "tokens_estimate": sum(tok.values()), "families": dict(fam.most_common()),
            "family_tokens": dict(tok.most_common())}  # fmt: skip


def payload(rows) -> bytes:
    clean = [{k: v for k, v in r.items() if k != "_tokens"} for r in rows]
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in clean).encode()


def digest(rows) -> str:
    return hashlib.sha256(payload(rows)).hexdigest()


_ = CYCLE
