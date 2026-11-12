"""Phase 4 continual fine-tuning data (approach A), built from the Phase 4 error analysis.

``p4_train_v1`` continues from the Phase 3 round-3 adapter. Its composition answers the
measured failures (results/phase4/error_analysis.json), not a wish for more rows:

* **Calendar forgetting.** About 60% of the rows replay the Phase 2 families with verified
  worked answers. Families where round 3 fell below Phase 2 get more rows: time zones
  that need the tool (2/28 vs 18/28), solar-term boundaries, verification,
  counterfactuals.
* **Wrong abstention.** Boundary rows pair a birth in the same minute as a solar term
  (ambiguous) with births minutes or hours away (answerable), and zone rows pair a
  supplied offset with a missing one.
* **Procedure confusion.** Every rule answer starts with a one-line header naming the
  procedure and its scope (五行计数 counts the eight visible characters; 强弱计数 counts
  得令 / 得地 / 得势). 9 of 14 wrong five-element answers had run the strength procedure.
* **Wording transfer.** A third of rule questions get a surface rewrite: another way of
  writing the pillars, a lead-in phrase, 日元 for 日主. The held-out test wordings and the
  independent set's templates are not used.
* **Citations.** No answer names a book (wording version 4).
"""

import random
import re

from bazireason import textstyle
from bazireason import train_data as TD
from bazireason import train_data3 as P3

VERSION = "p4_train_v1"
SEED = 2026111421
# Phase 2 families (train, valid) rows, drawn from a triple-size pool of the Phase 2 generator.
P2_QUOTAS = {
    "chart_plain": (120, 6), "chart_facts": (60, 3), "jie_boundary": (220, 10),
    "jie_window": (120, 6), "zi_transition": (160, 8), "tz_known": (100, 5),
    "tz_needs_tool": (120, 6), "tz_facts": (40, 2), "dst_fold": (50, 3), "dst_gap": (15, 1),
    "shanghai_dst": (20, 1), "missing": (60, 3), "invalid": (60, 3), "verify": (220, 10),
    "internal": (100, 5), "rule": (300, 14), "counterfactual": (160, 8), "policy": (40, 2),
    "mcq": (80, 4), "multi_turn": (80, 4), "v3_strict": (60, 3), "v4_reasoned": (60, 3),
}  # fmt: skip
# Within the Phase 2 rule family: the confused pair gets more rows.
RULE_SHARE = {"rule_elements": 50, "rule_strength": 40}
# Phase 3 families (train, valid, test) in the explicit wording without book titles.
P3_SIZES = {
    "drill": (250, 12, 0), "gods": (110, 5, 0), "strength": (70, 3, 0), "pattern": (70, 3, 0),
    "yongshen": (60, 3, 0), "interact": (130, 6, 0), "flow": (130, 6, 0), "domain": (70, 3, 0),
    "claims": (90, 4, 0), "ambig": (100, 6, 0), "mcq3": (140, 6, 0), "dialog": (80, 4, 0),
    "overview": (50, 2, 0), "strict_audit": (120, 6, 0), "strict_shift": (0, 0, 0),
}  # fmt: skip
PROCEDURES = {
    "elements": "五行计数：只数四柱明见的八个干支，不看藏干，也不看月令。",
    "strength": "强弱计数：得令、得地、得势三项，占两项为偏强。",
    "pattern": "月令取格：看月支藏干在年、月、时干的透出。",
    "ten_god": "十神：先写日主的十神表，再逐个查。",
    "hidden": "藏干表：只列地支所藏的天干。",
    "seasonal": "旺相休囚死：比较日主五行与月令五行。",
    "life_stage": "十二长生：阳干顺行，阴干逆行。",
    "xun_kong": "旬空：由日柱在六十甲子中的序数定旬，再查空亡。",
    "relations": "地支关系：逐对查合、冲、刑、害、破。",
    "flow": "流年：先求流年干支，再与原局逐项比对。",
    "luck": "大运：年干阴阳与性别定顺逆。",
    "relatives": "六亲与宫位：传统对应，不作预测。",
    "yongshen": "用神：只按题目指定的一派。",
}
TASK_PROCEDURE = {
    "rule_elements": "elements", "rule_strength": "strength", "p3_strength_strength": "strength",
    "rule_pattern": "pattern", "p3_pattern_pattern": "pattern", "rule_ten_god": "ten_god",
    "p3_gods_gods": "ten_god", "p3_drill_ten_god": "ten_god", "p3_drill_god_table": "ten_god",
    "rule_hidden": "hidden", "p3_drill_hidden": "hidden", "rule_seasonal": "seasonal",
    "p3_drill_season": "seasonal", "rule_life_stage": "life_stage", "p3_drill_stage": "life_stage",
    "rule_xun_kong": "xun_kong", "rule_relations": "relations", "p3_interact_interact": "relations",
    "p3_drill_branch_pair": "relations", "p3_drill_branch_card": "relations",
    "rule_flow_year": "flow", "p3_flow_flow": "flow", "rule_luck": "luck",
    "p3_drill_luck_dir": "luck", "rule_relatives": "relatives",
    "p3_yongshen_扶抑": "yongshen", "p3_yongshen_调候": "yongshen", "p3_yongshen_格局": "yongshen",
}  # fmt: skip
PILLAR_LINE = re.compile(r"年柱(\S{2})\s*月柱(\S{2})\s*日柱(\S{2})\s*时柱(\S{2})")
PILLAR_STYLES = ("{0}年 {1}月 {2}日 {3}时", "年{0}、月{1}、日{2}、时{3}")
LEADS = ("请教一个问题：", "想核对一下：", "帮我看看这个：")
AUGMENT_SHARE = 0.35


def procedure_header(row):
    """Prepend the procedure line to a rule answer (no change for other tasks)."""
    key = TASK_PROCEDURE.get(row["task"])
    if key:
        answer = row["messages"][-1]
        row["messages"][-1] = {
            **answer,
            "content": f"【规则】{PROCEDURES[key]}\n{answer['content']}",
        }
    return row


def augment(row, rng):
    """Surface rewrite of the question of a rule row; the answer is unchanged."""
    if row["task"] not in TASK_PROCEDURE or rng.random() >= AUGMENT_SHARE:
        return row
    index = max(i for i, m in enumerate(row["messages"]) if m["role"] == "user")
    text = row["messages"][index]["content"]
    body, sep, hint = text.partition("\n结论JSON")
    style = rng.choice(PILLAR_STYLES)
    body = PILLAR_LINE.sub(lambda m: style.format(*m.groups()), body, count=1)
    body = body.replace("日主", "日元") if rng.random() < 0.5 else body
    body = rng.choice(LEADS) + body if rng.random() < 0.5 else body
    row["messages"][index] = {**row["messages"][index], "content": body + sep + hint}
    row["augmented"] = True
    return row


def phase2_part(split, seed, blocked, rng):
    column = TD.SPLITS.index(split)
    pool = TD.generate_split(split, seed, blocked, 3.0 if split == "train" else 4.0, 2)
    by_family = {}
    for row in pool:
        by_family.setdefault(row["family"], []).append(row)
    rows = []
    for family, sizes in P2_QUOTAS.items():
        want = sizes[column]
        candidates = by_family.get(family, [])
        if family == "rule":
            picked = []
            for task, share in RULE_SHARE.items():
                n = round(want * share / 300)
                picked += rng.sample([r for r in candidates if r["task"] == task], n)
            rest = [r for r in candidates if r["task"] not in RULE_SHARE]
            picked += rng.sample(rest, want - len(picked))
            for row in picked:
                P3.explicit_phase2_rule(row)
        else:
            picked = rng.sample(candidates, min(want, len(candidates)))
        for row in picked:
            row["family"] = "replay_" + row["family"]
            row.pop("id", None)
        rows += picked
    return rows


def generate_split(split, seed=SEED, blocked=None):
    if blocked is None:
        blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    rng = random.Random(f"{VERSION}:{seed}:{split}")
    column = TD.SPLITS.index(split)
    pool = P3.Pool3(split, seed, blocked, VERSION)
    rows = []
    textstyle.set_version(4)
    try:
        for family, builder in P3.FAMILIES:
            target = P3_SIZES[family][column]
            if target:
                rows.extend(builder(pool, target))
    finally:
        textstyle.set_version(1)
    rows = [row for row in rows if TD._dates_allowed(row, pool)]
    rows += phase2_part(split, seed + 11, blocked, rng)
    rows = [augment(procedure_header(row), rng) for row in rows]
    rng.shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["id"] = f"P4-{split}-{index:05d}"
        row["split"] = split
        row["dataset_version"] = VERSION
        row["seed"] = seed
    return rows


dataset_bytes = TD.dataset_bytes
sha256_bytes = TD.sha256_bytes
