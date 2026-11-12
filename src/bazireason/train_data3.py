"""Phase 3 fine-tuning dataset (``p3_train_v1``): rule tables and multi-step reasoning.

Motivated by the Phase 2 error analysis (``docs/phase3_progress_report.md``):

* Rule answers failed on table recall, not on form, so a large ``drill`` family
  recites one table row per question and the chart-level families write out the
  rows they use.
* Strict one-pass tool answers were copied even when internally inconsistent,
  so ``strict_audit`` pairs clean and corrupted tool answers in the benchmark's
  own JSON-only format. Only errors that the supplied chart itself exposes are
  used; a wrong side of a solar term cannot be seen without the instant and is
  never a training target.
* Questions without a unique answer are paired with near-identical questions
  that have one (``ambig``), so the model cannot learn to always abstain.

Leakage guards are those of Phase 2 and are enforced by the same ``Pool``:
disjoint year blocks per split, no date within seven days of an evaluation
case or one day of a contest birth date, and test-only question wordings. No
contest question, option, answer or explanation is read. A share of the rows
replays the Phase 2 calendar families on new samples so the earlier skills are
rehearsed rather than overwritten.
"""

import copy
import json
import random
from datetime import datetime, timedelta

from bazireason import deep_rules as D
from bazireason import deep_tasks as K
from bazireason import rules as R
from bazireason import textstyle
from bazireason import train_data as TD
from bazireason import train_tasks as T
from bazireason import worked as W
from bazireason.calendar_engine import UTC8, compute_chart, jie_around
from bazireason.charttools import chart_tool_text, verified_chart
from bazireason.controls import corrupt_context
from bazireason.conventions import BRANCHES, CYCLE
from bazireason.p2_prompts import MCQ_SYSTEM, P2_SYSTEM, chat_messages, final_line, mcq_user
from bazireason.prompts import CONTEXT_LABELS, SYSTEM
from bazireason.schema import model_view

DATASET_VERSION = "p3_train_v1"
SEED = 2026111201
SPLITS = TD.SPLITS
# (train, valid, test) rows per family.
SIZES = {
    "drill": (1500, 60, 150),
    "gods": (350, 14, 40),
    "strength": (220, 8, 30),
    "pattern": (220, 8, 30),
    "yongshen": (240, 9, 30),
    "interact": (400, 16, 40),
    "flow": (500, 20, 50),
    "domain": (300, 12, 36),
    "claims": (450, 18, 48),
    "ambig": (450, 18, 54),
    "mcq3": (500, 20, 63),
    "dialog": (260, 10, 30),
    "overview": (160, 8, 20),
    "strict_audit": (700, 30, 0),
    "strict_shift": (0, 0, 180),
}
REPLAY_SCALE = (0.2, 0.25, 0.0)
SIZES_V2 = {
    # Second Phase 3 round: fresh samples in the explicit wording, weighted by the first
    # round's errors on the development split (Ten Gods inside chart tasks, pair relations,
    # multiple choice over years, one-pass tool audit).
    "drill": (760, 30, 0),
    "gods": (260, 10, 0),
    "strength": (70, 4, 0),
    "pattern": (110, 5, 0),
    "yongshen": (120, 6, 0),
    "interact": (300, 10, 0),
    "flow": (300, 10, 0),
    "domain": (220, 8, 0),
    "claims": (200, 8, 0),
    "ambig": (130, 9, 0),
    "mcq3": (400, 14, 0),
    "dialog": (120, 6, 0),
    "overview": (120, 5, 0),
    "strict_audit": (600, 20, 0),
    "strict_shift": (0, 0, 0),
}
SIZES_V3 = {
    # Third Phase 3 round (corrective): a smaller rehearsal of the Phase 3 families in the
    # explicit wording without book titles; most rows replay the Phase 2 families.
    "drill": (300, 20, 0),
    "gods": (120, 5, 0),
    "strength": (50, 3, 0),
    "pattern": (60, 3, 0),
    "yongshen": (60, 3, 0),
    "interact": (140, 5, 0),
    "flow": (140, 5, 0),
    "domain": (90, 4, 0),
    "claims": (100, 4, 0),
    "ambig": (90, 9, 0),
    "mcq3": (160, 6, 0),
    "dialog": (70, 3, 0),
    "overview": (50, 2, 0),
    "strict_audit": (200, 8, 0),
    "strict_shift": (0, 0, 0),
}
VERSIONS = {
    1: ("p3_train_v1", SIZES, REPLAY_SCALE),
    2: ("p3_train_v2", SIZES_V2, (0.12, 0.15, 0.0)),
    3: ("p3_train_v3", SIZES_V3, (0.5, 0.5, 0.0)),
}
# Wording of the answers: 2 is the first round's, 3 the explicit second-round wording,
# 4 the explicit wording without book titles.
TEXTSTYLE = {1: 2, 2: 3, 3: 4}
# Phase 2 rule tasks whose replayed answers get the table they need written first (round 3).
GOD_TABLE_TASKS = ("ten_god", "hidden", "flow_year", "pattern", "relatives", "strength")


def explicit_phase2_rule(row):
    """Prepend the needed table to a replayed Phase 2 rule answer; the rest is unchanged.

    After round 2 the adapter used its explicit procedure only under Phase 3 wording and
    still guessed Ten Gods and relations under the Phase 2 rule prompts.
    """
    kind = row["task"].removeprefix("rule_")
    pillars = row["meta"]["pillars"]
    if kind in GOD_TABLE_TASKS:
        lead = K.god_table(pillars["day"][0])
    elif kind == "relations":
        lead = K.relation_tables(pillars)
    elif kind == "xun_kong":
        lead = K.cycle_index_text(pillars["day"])
    else:
        return row
    answer = row["messages"][-1]
    row["messages"][-1] = {**answer, "content": lead + "\n" + answer["content"]}
    return row


class Pool3(TD.Pool):
    """The Phase 2 sampler with its own random stream."""

    def __init__(self, split, seed, blocked, name=DATASET_VERSION):
        super().__init__(split, seed, blocked, 2)
        self.rng = random.Random(f"{name}:{seed}:{split}")


class Chart:
    """One sampled chart with the two ways it can be shown to the model."""

    def __init__(self, pool, hours=range(24)):
        self.wall, self.pillars, self.sect = TD._random_chart(pool, hours)
        self.gender = pool.rng.choice(("male", "female"))
        self._tool = None

    def bare(self, gender=True) -> str:
        line = T.chart_line(self.pillars)
        return line + (f"，{T.GENDER_ZH[self.gender]}命。" if gender else "。")

    def tool(self):
        """The verified tool block, or None when the engine does not reproduce the chart."""
        if self._tool is None:
            inputs = {"datetime": TD._iso(W.shanghai_wall(self.wall)), "timezone": TD.SHANGHAI}
            chart = verified_chart(inputs, TD.convention(self.sect), self.gender)
            self._tool = (
                chart if chart["status"] == "ok" and chart["pillars"] == self.pillars else {}
            )
        return self._tool or None

    def header(self, pool, tool_share=0.3, gender=True):
        """(header text, mode). The tool block always states the gender."""
        if pool.rng.random() < tool_share and self.tool():
            return chart_tool_text(self.tool(), flow_years=False) + "\n", "chart"
        return self.bare(gender), "none"


def _row(family, kind, mode, group, messages, answer, meta):
    return TD._row(family, f"p3_{family}_{kind}", mode, group, messages, answer, meta)


def _chat(header, question, hint):
    return chat_messages(header + question + "\n" + K.HINTS[hint])


def _meta(chart, expected, variant, kind, **extra):
    return {
        "pillars": chart.pillars,
        "gender": chart.gender,
        "expected": expected,
        "template": variant,
        "kind": kind,
        **extra,
    }


# ---- Families ---------------------------------------------------------------------------------


def drill(pool, count):
    rows = []
    while len(rows) < count:
        kinds = K.DRILLS_V3 if K.explicit() else K.DRILLS
        kind = kinds[len(rows) % len(kinds)]
        variant = pool.variant()
        question, text, response = K.drill(kind, pool.rng, variant)
        rows.append(
            _row(
                "drill",
                kind,
                "none",
                f"drill-{len(rows)}",
                chat_messages(question + "\n" + K.HINTS["answer"]),
                (text + "\n" if text else "") + final_line(response),
                {"expected": response, "template": variant, "kind": kind},
            )
        )
    return rows


def _simple(family, hint, build, tool_share=0.3):
    def builder(pool, count):
        rows = []
        while len(rows) < count:
            chart, variant = Chart(pool), pool.variant()
            header, mode = chart.header(pool, tool_share)
            question, text, response = build(chart, pool.rng, variant)
            kind = response.get("school", family) if family == "yongshen" else family
            rows.append(
                _row(
                    family,
                    kind,
                    mode,
                    f"{family}-{len(rows)}",
                    _chat(header, question, hint),
                    text + "\n" + final_line(response),
                    _meta(chart, response, variant, kind),
                )
            )
        return rows

    return builder


gods = _simple("gods", "gods", lambda c, rng, v: K.gods_task(c.pillars, rng, v))
strength = _simple("strength", "strength", lambda c, rng, v: K.strength_task(c.pillars, rng, v))
pattern = _simple("pattern", "pattern", lambda c, rng, v: K.pattern_task(c.pillars, rng, v))
yongshen = _simple("yongshen", "yongshen", lambda c, rng, v: K.yongshen_task(c.pillars, rng, v))
interact = _simple(
    "interact", "interact", lambda c, rng, v: K.interact_task(c.pillars, rng, v), 0.0
)


def flow(pool, count):
    rows = []
    while len(rows) < count:
        chart, variant = Chart(pool), pool.variant()
        year = min(chart.wall.year + pool.rng.randrange(8, 70), 2160)
        header, mode = chart.header(pool, 0.4)
        cycles = chart.tool()["luck"]["cycles"] if mode == "chart" else None
        # Every sixth row is steered to a year with a special configuration, which is rare.
        if len(rows) % 6 == 0:
            for candidate in range(chart.wall.year + 8, chart.wall.year + 68):
                luck = D.luck_in_year(cycles, candidate) if cycles else None
                if D.flow_analysis(chart.pillars, R.year_pillar(candidate), luck)["special"]:
                    year = candidate
                    break
        question, text, response = K.flow_task(chart.pillars, pool.rng, variant, year, cycles)
        rows.append(
            _row(
                "flow",
                "flow",
                mode,
                f"flow-{len(rows)}",
                _chat(header, question, "flow"),
                text + "\n" + final_line(response),
                _meta(chart, response, variant, "flow", year=year),
            )
        )
    return rows


def domain(pool, count):
    rows, names = [], tuple(D.DOMAINS)
    while len(rows) < count:
        chart, variant = Chart(pool), pool.variant()
        name = names[len(rows) % len(names)]
        first_year = min(chart.wall.year + pool.rng.randrange(16, 50), 2150)
        header, mode = chart.header(pool, 0.3)
        question, text, response = K.domain_task(
            chart.pillars, pool.rng, variant, chart.gender, name, first_year
        )
        rows.append(
            _row(
                "domain",
                name,
                mode,
                f"domain-{len(rows)}",
                _chat(header, question, "domain"),
                text + "\n" + final_line(response),
                _meta(chart, response, variant, name, year=first_year),
            )
        )
    return rows


def claims(pool, count):
    rows = []
    while len(rows) < count:
        chart, variant = Chart(pool), pool.variant()
        header, mode = chart.header(pool, 0.45)
        question, text, response = K.claims_task(chart.pillars, pool.rng, variant, chart.gender)
        if mode == "chart":
            question = "以上是核验过的排盘。" + question
        rows.append(
            _row(
                "claims",
                "claims",
                mode,
                f"claims-{len(rows)}",
                _chat(header + ("\n" if mode == "none" else ""), question, "claims"),
                text + "\n" + final_line(response),
                _meta(chart, response, variant, "claims"),
            )
        )
    return rows


def ambig(pool, count):
    rows = []
    while len(rows) < count:
        kind = K.AMBIG_KINDS[len(rows) % len(K.AMBIG_KINDS)]
        chart, variant = Chart(pool), pool.variant()
        shape, question, text, response = K.ambig_task(
            kind, chart.pillars, pool.rng, variant, chart.gender
        )
        header = {
            "full": chart.bare(),
            "three": K.three_pillar_line(chart.pillars) + "。",
            "nogender": chart.bare(gender=False),
        }[shape]
        rows.append(
            _row(
                "ambig",
                kind,
                "none",
                f"ambig-{len(rows)}",
                _chat(header, question, "ambig"),
                text + "\n" + final_line(response),
                _meta(chart, response, variant, kind),
            )
        )
    return rows


def mcq3(pool, count):
    rows = []
    while len(rows) < count:
        kind = K.MCQ_KINDS[len(rows) % len(K.MCQ_KINDS)]
        chart = Chart(pool)
        year = min(chart.wall.year + pool.rng.randrange(15, 45), 2120)
        item = K.mcq_task(kind, chart.pillars, pool.rng, chart.gender, year)
        if item is None:
            continue
        inputs = {"datetime": TD._iso(W.shanghai_wall(chart.wall)), "timezone": TD.SHANGHAI}
        profile = (
            f"性别：{T.GENDER_ZH[chart.gender]}；出生（公历）：{inputs['datetime'].replace('T', ' ')}；"
            f"时区：{TD.SHANGHAI}"
        )
        if pool.rng.random() < 0.5 and chart.tool():
            block, mode = chart_tool_text(chart.tool(), flow_years=False), "chart"
        else:
            block, mode = f"[四柱] {T.pillars_text(chart.pillars)}", "none"
        user = mcq_user(profile, item["question"], item["options"], block)
        rows.append(
            _row(
                "mcq3",
                kind,
                mode,
                f"mcq3-{len(rows)}",
                chat_messages(user, MCQ_SYSTEM),
                f"{item['reason']}\n答案：{item['answer']}",
                _meta(chart, {"answer": item["answer"]}, 0, kind, year=year),
            )
        )
    return rows


def overview(pool, count):
    rows = []
    while len(rows) < count:
        chart, variant = Chart(pool), pool.variant()
        if not chart.tool():
            continue
        question, text, response = K.overview_task(chart.pillars, pool.rng, variant, chart.gender)
        rows.append(
            _row(
                "overview",
                "overview",
                "chart",
                f"overview-{len(rows)}",
                _chat(chart_tool_text(chart.tool(), flow_years=False) + "\n", question, "overview"),
                text + "\n" + final_line(response),
                _meta(chart, response, variant, "overview"),
            )
        )
    return rows


# ---- Multi-turn conversations -------------------------------------------------------------------

FIRST_TURNS = ("gods", "strength", "pattern", "interact")
SECOND_TURNS = ("flow", "yongshen", "domain", "claims")
THIRD_TURNS = ("pushback", "self_correct", "decline", "followup")


def _turn(kind, chart, pool, variant):
    """(question, hint, text, response) for one content turn on a chart."""
    rng = pool.rng
    if kind == "gods":
        q, t, r = K.gods_task(chart.pillars, rng, variant)
        return q, "gods", t, r
    if kind == "strength":
        q, t, r = K.strength_task(chart.pillars, rng, variant)
        return q, "strength", t, r
    if kind == "pattern":
        q, t, r = K.pattern_task(chart.pillars, rng, variant)
        return q, "pattern", t, r
    if kind == "interact":
        q, t, r = K.interact_task(chart.pillars, rng, variant)
        return q, "interact", t, r
    if kind == "flow":
        year = min(chart.wall.year + rng.randrange(8, 70), 2160)
        q, t, r = K.flow_task(chart.pillars, rng, variant, year)
        return q, "flow", t, r
    if kind == "yongshen":
        q, t, r = K.yongshen_task(chart.pillars, rng, variant)
        return q, "yongshen", t, r
    if kind == "domain":
        name = rng.choice(tuple(D.DOMAINS))
        first = min(chart.wall.year + rng.randrange(16, 50), 2150)
        q, t, r = K.domain_task(chart.pillars, rng, variant, chart.gender, name, first)
        return q, "domain", t, r
    q, t, r = K.claims_task(chart.pillars, rng, variant, chart.gender, 3)
    return q, "claims", t, r


def dialog(pool, count):
    """Two- and three-turn conversations on one chart; the last assistant turn is the target."""
    rows = []
    while len(rows) < count:
        chart, variant = Chart(pool), pool.variant()
        rng = pool.rng
        q1, h1, t1, r1 = _turn(rng.choice(FIRST_TURNS), chart, pool, variant)
        history = [
            {"role": "user", "content": chart.bare() + q1 + "\n" + K.HINTS[h1]},
            {"role": "assistant", "content": t1 + "\n" + final_line(r1)},
        ]
        scenario = THIRD_TURNS[len(rows) % len(THIRD_TURNS)]
        day = chart.pillars["day"][0]
        name = rng.choice(D.OTHER)
        stem = chart.pillars[name][0]
        right = R.ten_god(day, stem)
        wrong = rng.choice([g for g in R.TEN_GODS if g != right])
        where = f"{K.POS[name]}干{stem}"
        if scenario == "followup":
            q2, h2, t2, r2 = _turn(rng.choice(SECOND_TURNS), chart, pool, variant)
            user = "还是这个八字。" + q2 + "\n" + K.HINTS[h2]
            answer, expected = t2 + "\n" + final_line(r2), r2
        elif scenario == "pushback":
            user = rng.choice(
                (
                    f"不对吧，我看到别的地方说{where}是{wrong}。",
                    f"有位老师说{where}应该是{wrong}，你是不是弄错了？",
                    f"可是软件上显示{where}为{wrong}，请改过来。",
                )
            )
            user += "\n" + K.HINTS["followup"]
            expected = {"revised": False, "answer": right}
            answer = (
                f"按规则重新核对：{R.ten_god_reason(day, stem)}。所以{where}是{right}，不是{wrong}，"
                "前面的结论不变；如果对方用的是别的日主或别的排盘，可以把依据发来再比对。\n"
                + final_line(expected)
            )
        elif scenario == "self_correct":
            # The earlier assistant turn is shown with one wrong Ten God, then questioned.
            history[1] = {
                "role": "assistant",
                "content": f"按规则推算，{where}是日主的{wrong}。\n"
                + final_line({"answer": wrong}),
            }
            history[0] = {
                "role": "user",
                "content": chart.bare() + f"{where}是日主的什么十神？\n" + K.HINTS["answer"],
            }
            user = rng.choice(
                ("这个结果好像不对，请按规则重新核对。", "请再检查一遍你刚才的结论。")
            )
            user += "\n" + K.HINTS["followup"]
            expected = {"revised": True, "answer": right}
            answer = (
                f"重新核对：{R.ten_god_reason(day, stem)}。上一条写成{wrong}是错的，应为{right}。\n"
                + final_line(expected)
            )
        else:
            topic = rng.randrange(len(T.POLICY_TOPICS))
            question, text, expected, _ = T.policy_answer(
                chart.pillars, chart.gender, topic, rng.randrange(3), chart.wall.year + 25
            )
            user = question.split("命。", 1)[1] + "\n" + TD.OUTPUT_HINTS["policy"]
            answer = text + "\n" + final_line(expected)
        rows.append(
            _row(
                "dialog",
                scenario,
                "none",
                f"dialog-{len(rows)}",
                chat_messages(user, P2_SYSTEM, history),
                answer,
                _meta(chart, expected, variant, scenario, wrong_claim=wrong),
            )
        )
    return rows


# ---- Strict one-pass audit of a supplied tool answer -------------------------------------------

AUDIT_TRAIN = (
    ("clean", 45),
    ("hour_stem", 16),
    ("day_rollover", 12),
    ("month_stem", 8),
    ("year_shift", 5),
    ("hour_branch", 5),
    ("default_guess", 6),
    ("clean_open", 3),
)
# Second round: more of the errors that a table lookup exposes, which the first round missed.
AUDIT_TRAIN_V2 = (
    ("clean", 42),
    ("hour_stem", 22),
    ("month_stem", 14),
    ("day_rollover", 8),
    ("year_shift", 5),
    ("hour_branch", 4),
    ("default_guess", 3),
    ("clean_open", 2),
)
AUDIT_SHIFT = (
    ("clean", 30),
    ("hour_stem", 20),
    ("day_rollover", 20),
    ("month_stem", 15),
    ("year_shift", 10),
    ("hour_branch", 10),
    ("default_guess", 15),
    # The two below reverse the association between the hour of day and the kind of error.
    ("day_shift_midday", 30),
    ("hour_stem_night", 30),
)
NIGHT_HOURS = (21, 22, 23, 0, 1)
AUDIT_ZONES = (TD.SHANGHAI,) * 6 + ("Asia/Hong_Kong", "Asia/Taipei", "Asia/Singapore", "UTC")


def strict_messages(case, supplied):
    """The benchmark's tool_context prompt with an arbitrary supplied answer."""
    public = {"input": case["input"], "convention": case["convention"]}
    prompt = "Compute the calendar result under the specified profile.\n"
    prompt += json.dumps(public, ensure_ascii=False, sort_keys=True)
    prompt += f"\n{CONTEXT_LABELS['tool_context']}:\n" + json.dumps(
        supplied, ensure_ascii=False, sort_keys=True
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def _shift(pillar, steps):
    return CYCLE[(CYCLE.index(pillar) + steps) % 60]


def _audit_case(pool, kind):
    """(case, truth, supplied, changed fields) for one audit row, or None to resample."""
    rng = pool.rng
    night = kind in ("day_rollover", "hour_stem_night")
    hours = NIGHT_HOURS if night else range(2, 21) if kind != "clean" else range(24)
    if kind in ("default_guess", "clean_open"):
        hours = (23,) if kind == "default_guess" and rng.random() < 0.6 else range(2, 21)
    wall = pool.wall(hours)
    reference = wall.replace(tzinfo=UTC8)
    (prev_t, _), (next_t, _) = jie_around(reference)
    near_jie = min(reference - prev_t, next_t - reference) <= timedelta(days=2)
    if near_jie and kind != "clean":
        return None
    zone = rng.choice(AUDIT_ZONES)
    inputs = {"datetime": TD._iso(TD._wall_in_zone(wall, zone)), "timezone": zone}
    conv = TD.convention(rng.choice((1, 2)))
    if kind == "default_guess":
        style = "sect" if wall.hour == 23 else rng.choice(("date", "zone"))
        if style == "sect":
            conv = TD.convention(None)
        elif style == "date":
            inputs["datetime"] = inputs["datetime"][:10]
        else:
            inputs.pop("timezone")
    if kind == "clean_open":
        conv = TD.convention(None)  # no sect, but the hour makes it irrelevant
    case = {"input": inputs, "convention": conv}
    context = compute_chart(inputs, conv)
    truth = model_view(context)
    pool.reserve(wall)
    if kind == "default_guess":
        if context["status"] != "ambiguous":
            return None
        supplied, corruption = corrupt_context(case, context)
        if corruption is None:
            return None
        return case, truth, supplied, dict(corruption["fields"])
    if context["status"] != "ok":
        return None
    supplied = copy.deepcopy(truth)
    pillars = supplied["pillars"]
    if kind in ("clean", "clean_open"):
        return case, truth, supplied, {}
    if kind in ("hour_stem", "hour_stem_night"):
        steps = rng.choice((12, 12, 12, 24, 36, 48)) if kind == "hour_stem" else 12
        changed = {"hour": _shift(pillars["hour"], steps)}
    elif kind == "month_stem":
        changed = {"month": _shift(pillars["month"], rng.choice((12, 24, 36, 48)))}
    elif kind == "year_shift":
        if not 3 <= reference.month <= 12:
            return None
        changed = {"year": _shift(pillars["year"], rng.choice((-1, 1)))}
    elif kind == "hour_branch":
        other = BRANCHES[(BRANCHES.index(pillars["hour"][1]) + rng.choice((-1, 1))) % 12]
        changed = {"hour": R.hour_pillar_for(pillars["day"][0], other)}
    elif kind == "day_rollover":
        supplied, corruption = corrupt_context(case, context)
        if corruption is None or corruption["type"] != "day_rollover":
            return None
        return case, truth, supplied, dict(corruption["fields"])
    elif kind == "day_shift_midday":
        changed = {"day": _shift(pillars["day"], rng.choice((-1, 1)))}
    else:
        raise ValueError(kind)
    pillars.update(changed)
    return case, truth, supplied, changed


def _audit(pool, count, weights, family):
    rows, kinds = [], [kind for kind, weight in weights for _ in range(weight)]
    while len(rows) < count:
        kind = (
            kinds[len(rows) * 7 % len(kinds)]
            if family == "strict_shift"
            else pool.rng.choice(kinds)
        )
        made = _audit_case(pool, kind)
        if made is None:
            continue
        case, truth, supplied, changed = made
        rows.append(
            _row(
                family,
                kind,
                "tool_context" if not changed else "tool_corrupted",
                f"{family}-{len(rows)}",
                strict_messages(case, supplied),
                json.dumps(truth, ensure_ascii=False),
                {
                    "inputs": case["input"],
                    "convention": case["convention"],
                    "expected": truth,
                    "supplied": supplied,
                    "changed": changed,
                    "kind": kind,
                },
            )
        )
    return rows


def strict_audit(pool, count):
    return _audit(pool, count, AUDIT_TRAIN_V2 if K.explicit() else AUDIT_TRAIN, "strict_audit")


def strict_shift(pool, count):
    return _audit(pool, count, AUDIT_SHIFT, "strict_shift")


# ---- Assembly -----------------------------------------------------------------------------------

FAMILIES = (
    ("drill", drill),
    ("gods", gods),
    ("strength", strength),
    ("pattern", pattern),
    ("yongshen", yongshen),
    ("interact", interact),
    ("flow", flow),
    ("domain", domain),
    ("claims", claims),
    ("ambig", ambig),
    ("mcq3", mcq3),
    ("dialog", dialog),
    ("overview", overview),
    ("strict_audit", strict_audit),
    ("strict_shift", strict_shift),
)


def generate_split(split, seed=SEED, blocked=None, scale=1.0, version=1):
    """Rows of one split: the Phase 3 families, then a replay of the Phase 2 families."""
    if blocked is None:
        blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    name, sizes, replay = VERSIONS[version]
    pool = Pool3(split, seed + version - 1, blocked, name)
    column = SPLITS.index(split)
    rows = []
    textstyle.set_version(TEXTSTYLE[version])
    try:
        for family, builder in FAMILIES:
            target = round(sizes[family][column] * scale)
            if target:
                rows.extend(builder(pool, target))
    finally:
        textstyle.set_version(1)
    rows = [row for row in rows if TD._dates_allowed(row, pool)]
    if replay[column]:
        old = TD.generate_split(split, seed + 7 * version, blocked, replay[column] * scale, 2)
        for row in old:
            if version >= 3 and row["family"] == "rule":
                explicit_phase2_rule(row)
            row["family"] = "replay_" + row["family"]
            row.pop("id", None)
        rows.extend(old)
    prefix = "P3" if version == 1 else f"P3v{version}"
    for index, row in enumerate(rows, 1):
        row["id"] = f"{prefix}-{split}-{index:05d}"
        row["split"] = split
        row["dataset_version"] = name
        row["seed"] = seed
    return rows


dataset_bytes = TD.dataset_bytes
sha256_bytes = TD.sha256_bytes
_ = datetime
