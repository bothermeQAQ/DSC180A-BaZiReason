"""Phase 5 items and scoring: one format for template rule rows, independent questions and
blind-wording questions, with the oracle intent of every item.

An item is an ordinary chat row (system + user turn ending with a JSON hint) whose ``meta``
also carries ``pillars``, ``gender``, ``question`` (the user's bare question), ``intent``
(the oracle intent, see ``bazireason.intent``) and ``expected``.

Scoring reads only the final JSON line, as in Phases 2-4. The "fact" family uses a
content-lenient comparison fixed before any Phase 5 run: a list answer may also be given
as a string of its characters ("寅卯" for ["寅", "卯"]); the Phase 4 scorer rejected that
form, which is reported as a scoring artefact in results/phase5/addenda/.
"""

import re

from bazireason import deep_tasks as K
from bazireason.intent import FACT_HINTS
from bazireason.p2_eval import score_policy
from bazireason.p2_prompts import P2_SYSTEM, extract_final_json
from bazireason.p3_eval import score3

GROUNDED_FAMILIES = ("gods", "strength", "pattern", "yongshen", "interact", "flow", "domain", "ambig", "overview")  # fmt: skip
# Ambiguity kinds whose oracle plan reproduces the row label exactly (the others use a question
# variant chosen at random during generation, or a three-pillar chart).
AMBIG_KINDS = ("yongshen_open", "star_basis", "no_rule")
FOUR = re.compile(r"([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])")
STARS = ("桃花", "驿马", "华盖")
LIST_KINDS = ("hidden", "xun_kong", "missing_element")


def head_line(pillars: dict, gender) -> str:
    sex = {"male": "，男命", "female": "，女命"}.get(gender, "")
    return (
        f"八字：年柱{pillars['year']} 月柱{pillars['month']} 日柱{pillars['day']} "
        f"时柱{pillars['hour']}（日主{pillars['day'][0]}）{sex}。"
    )


def make_item(item_id, split, source, pillars, gender, question, intent, plan, group):
    """A row in the common format; ``plan`` comes from ``intent.plan_for`` (the oracle)."""
    user = head_line(pillars, gender) + question + "\n" + plan["hint"]
    return {
        "id": item_id,
        "family": plan["family"],
        "task": f"p5_{plan['family']}_{plan['kind']}",
        "split": split,
        "mode": "none",
        "group_id": group,
        "messages": [{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": user}],
        "meta": {
            "source": source,
            "pillars": pillars,
            "gender": gender,
            "question": question,
            "intent": intent,
            "kind": plan["kind"],
            "expected": plan["reference"],
        },
    }


# ---- oracle intent and bare question of earlier rows --------------------------------------------
def template_intent(row):
    """The oracle intent of a Phase 3 template row, or None if it is not a grounded family."""
    family, meta = row["family"], row["meta"]
    kind = meta.get("kind")
    if family not in GROUNDED_FAMILIES or "pillars" not in meta or len(meta["pillars"]) < 4:
        return None
    if family == "ambig":
        if kind not in AMBIG_KINDS:
            return None
        if kind == "star_basis":
            user = (
                row["messages"][-2]["content"]
                if row["messages"][-1]["role"] == "assistant"
                else row["messages"][-1]["content"]
            )
            star = next((s for s in STARS if s in user), None)
            return f"star:{star}" if star else None
        return f"ambig:{kind}"
    if family == "yongshen":
        return f"yongshen:{kind}"
    if family == "flow":
        return f"flow:{meta['year']}"
    if family == "domain":
        return f"domain:{kind}:{meta['year']}"
    return family


def indep_item(row):
    """A Phase 4 independent row in the common format (labels stay lunar-python's)."""
    user = row["messages"][-1]["content"]
    found = FOUR.findall(user.split("\n")[0])
    pillars = dict(zip(("year", "month", "day", "hour"), found[:4], strict=True))
    if row["meta"]["kind"] == "xun_kong" and "日柱为" in user:
        pillars = dict(
            zip(
                ("year", "month", "day", "hour"),
                FOUR.findall(user.split("全盘")[1])[:4],
                strict=True,
            )
        )
    gender = "male" if "男" in user else ("female" if "女" in user else None)
    kind = row["meta"]["kind"]
    position = row["meta"].get("pillar")
    intent = f"fact:{kind}" + (
        f":{'hour' if position == 'time' else position}"
        if kind in ("ten_god_stem", "ten_god_branch", "hidden", "life_stage")
        else ""
    )
    meta = dict(row["meta"], pillars=pillars, gender=gender, intent=intent,
                question=user.split("\n")[0], expected=row["meta"]["expected"])  # fmt: skip
    return dict(row, family="fact", meta=meta)


def bare_question(row) -> str:
    """The user's question without the chart line and the JSON hint."""
    meta = row["meta"]
    if "question" in meta:
        return meta["question"]
    user = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")
    if row["messages"][-1]["role"] == "assistant":
        user = row["messages"][-2]["content"]
    body = user.rsplit("\n", 1)[0]
    if body.startswith("[排盘工具"):
        return body.split("\n")[-1]
    return body.split("。", 1)[1] if body.startswith("八字：") and "。" in body else body


# ---- scoring -------------------------------------------------------------------------------------
def _norm_fact(kind, value):
    if kind in LIST_KINDS:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            value = "".join(value)  # ["申酉"] and ["申", "酉"] and "申酉" are the same answer
        if isinstance(value, str):
            value = re.findall(r"[甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥木火土金水]", value)
        return sorted(value) if isinstance(value, list) else None
    if kind == "luck_direction" and isinstance(value, str):
        value = (
            value.strip()
            .replace("顺行", "顺")
            .replace("逆行", "逆")
            .replace("forward", "顺")
            .replace("backward", "逆")
        )
        return value[:1]
    if kind == "element_count" and isinstance(value, dict):
        try:
            return {k: int(value.get(k, -1)) for k in "木火土金水"}
        except (TypeError, ValueError):
            return None
    return value.strip() if isinstance(value, str) else value


def score5(row, text):
    family, meta = row["family"], row["meta"]
    base = {
        "id": row["id"],
        "family": family,
        "intent": meta.get("intent"),
        "source": meta.get("source", "template"),
        "group": f"{family}:{row.get('group_id', row['id'])}",
    }
    answer = extract_final_json(text or "")
    if family == "fact":
        kind = meta["intent"].split(":")[1]
        got = answer.get("answer") if isinstance(answer, dict) else None
        want = meta["expected"]["answer"]
        ok = got is not None and _norm_fact(kind, got) == _norm_fact(kind, want)
        return dict(base, kind=kind, parsed=isinstance(answer, dict), correct=bool(ok))
    if family == "policy":
        return dict(base, kind="prediction", **score_policy(meta["expected"], answer, text))
    if family in GROUNDED_FAMILIES or family in ("claims", "mcq3", "dialog"):
        scored = score3(row, text)
        return dict(base, kind=scored.get("kind"), parsed=scored.get("parsed"), correct=scored["correct"],
                    unseen_wording=scored.get("unseen_wording"))  # fmt: skip
    raise ValueError(family)


HINT_OF_FAMILY = dict(K.HINTS, fact=None, policy=None)
__all__ = [
    "FACT_HINTS",
    "make_item",
    "template_intent",
    "indep_item",
    "bare_question",
    "score5",
    "head_line",
]
