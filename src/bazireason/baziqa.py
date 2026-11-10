"""BaziQA contest benchmark: loading, verified-chart prompts, parsing and scoring.

BaziQA (Chen and Liu, arXiv:2602.12889; github.com/ChenJiangxi/BaziQA, MIT for the
transcription) holds the questions of the Hong Kong "Global Fortune Teller
Contest". This module only *evaluates* on it. No question, option, answer key
or source page is used for training, validation or prompt tuning: the training
generator reads birth dates alone, to keep those dates out of its samples.

Answering a contest question correctly is not evidence that destiny can be
predicted. The keys record what a contest organiser published about a person;
the benchmark measures agreement with those keys.

Chart convention for the "with tools" conditions
------------------------------------------------
Charts come from this project's engine under its declared profile: the civil
instant is normalised to fixed UTC+08:00 with IANA rules, the year changes at
立春 and the month at the Jie instants. ``sect=1`` (day pillar advances at 23:00)
is used because the contest setters do so. A record without a time zone cannot
be normalised; the tool then shows a chart under the explicit, labelled
assumption that the written clock time is UTC+08:00. A record without a time of
day gets no chart.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from bazireason import rules
from bazireason.calendar_engine import tzdata_keys
from bazireason.charttools import chart_tool_text, pillars_text, verified_chart
from bazireason.conventions import CYCLE, PILLARS, Convention
from bazireason.metrics import ratio
from bazireason.p2_prompts import (
    MCQ_SYSTEM,
    OUTPUT_HINTS,
    P2_SYSTEM,
    SELF_CHART_LABEL,
    chat_messages,
    mcq_user,
)

PINNED_COMMIT = "3bad3deaa5aa3a191c9c68b4d8b17f46decff45e"  # data identical to tag v2.3.0
CONTEST_SECT = 1
ASSUMED_ZONE = "Etc/GMT-8"  # fixed UTC+08:00 without daylight saving
CONDITIONS = ("base_no_tools", "base_tools", "tuned_no_tools", "tuned_tools")
# Champion results as listed in the BaziQA README (correct answers out of all questions).
HUMAN_CHAMPIONS = {
    2010: (6, 9),
    2011: (8, 15),
    2012: (8, 15),
    2015: (15, 35),
    2016: (23, 35),
    2018: (19, 40),
    2019: (20, 40),
    2020: (17, 40),
    2022: (19, 40),
    2023: (21, 40),
    2024: (26, 40),
    2026: (31, 40),
}
# BaziQA paper, v1.0.0 data, 2021-2025, multi-turn with the authors' charts and prompts.
PUBLISHED_BASELINES = {
    "DeepSeek-Chat-V3 (multi-turn)": 0.367,
    "DeepSeek-R1 (multi-turn)": 0.341,
    "GPT-5.1-Chat (multi-turn)": 0.325,
    "Gemini-2.5-Flash (multi-turn)": 0.324,
    "Gemini-3-Pro (multi-turn)": 0.321,
}
# Fixed keyword rules, applied in this order to the question stem only. They were
# written before any model output on the benchmark was seen.
CATEGORY_RULES = (
    ("健康灾厄", r"病|健康|手术|受伤|伤|癌|意外|去世|过世|身故|死|亡|灾|车祸|寿|开刀|残"),
    ("感情婚姻", r"婚|配偶|丈夫|妻|老公|老婆|感情|恋|伴侣|桃花|同性|对象|男友|女友|姻缘"),
    ("六亲家庭", r"父|母|兄|弟|姐|妹|子女|儿子|女儿|孩子|六亲|家境|出身|祖|家庭|生育|小孩"),
    ("学业", r"学历|学业|读书|考试|升学|大学|留学|毕业|博士|硕士|成绩"),
    ("财富", r"财|钱|富|债|破产|投资|买房|置业|收入|资产|股票|生意"),
    ("事业", r"事业|职业|工作|行业|职位|升职|官|创业|公司|从事|任职|老板"),
    ("性格外貌", r"性格|外貌|相貌|身高|长相|为人|脾气|体型|特征"),
    ("流年应期", r"哪一年|何年|哪年|几岁|什么时候|何时|[12][0-9]{3}\s*年|流年|大运"),
)


def category_of(question: str) -> str:
    for name, pattern in CATEGORY_RULES:
        if re.search(pattern, question):
            return name
    return "其他"


def load_contest(root="external/BaziQA"):
    """Every contest question, grouped by person, with provenance fields."""
    root = Path(root)
    files = sorted((root / "data").glob("hkjfma_*.json")) + sorted(
        (root / "data").glob("contest8_*.json")
    )
    persons = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data[0]
        charts = json.loads((root / "charts" / "bazi" / path.name).read_text(encoding="utf-8"))
        for person in data[1:]:
            method = (person.get("setter") or {}).get("method") or ""
            published = charts["charts"].get(person["person_id"], {}).get("chart", {})
            pillars = published.get("pillars") or {}
            persons.append(
                {
                    "file": path.stem,
                    "year": int(meta["current_year"]),
                    "answer_source": meta.get("answer_source", "official"),
                    "person_id": person["person_id"],
                    "case": f"{meta['current_year']}-{str(person.get('case_no', person['person_id'])).rstrip('ab')}",
                    "gender": person["profile"]["gender"],
                    "birth": person["profile"]["birth"],
                    "method": method or None,
                    "bazi": "八字" in method,
                    "context": list(person.get("context") or []),
                    "published_pillars": {
                        k: (pillars[k] or {}).get("ganzhi") if pillars.get(k) else None
                        for k in PILLARS
                    }
                    if pillars
                    else None,
                    "questions": [
                        {
                            "question_id": q["question_id"],
                            "question": q["question"],
                            "options": list(q["options"]),
                            "accepted": list(q.get("accepted_answers", [q["answer"]])),
                            "category": category_of(q["question"]),
                        }
                        for q in person["questions"]
                    ],
                }
            )
    return persons


def engine_request(person):
    """(inputs, convention, assumed_zone) for the calendar engine."""
    birth = person["birth"]
    day = f"{birth['year']:04d}-{birth['month']:02d}-{birth['day']:02d}"
    raw = (
        day
        if birth.get("hour") is None
        else f"{day}T{birth['hour']:02d}:{birth.get('minute') or 0:02d}"
    )
    zone = birth.get("timezone")
    assumed = zone is None or zone not in tzdata_keys()
    inputs = {"datetime": raw, "timezone": ASSUMED_ZONE if assumed else zone}
    return inputs, Convention(CONTEST_SECT).to_dict(), assumed


def tool_chart(person):
    inputs, convention, assumed = engine_request(person)
    chart = verified_chart(inputs, convention, person["gender"], until_year=person["year"])
    return chart, assumed


def tool_block(person) -> str:
    chart, assumed = tool_chart(person)
    text = chart_tool_text(chart)
    if assumed and chart["status"] == "ok":
        text += (
            "\n【注意】原始资料没有给出出生地或时区；以上是把所写的钟面时间直接当作 UTC+08:00 "
            "得到的排盘，属于未经核验的假设。"
        )
    return text


def profile_line(person) -> str:
    birth = person["birth"]
    day = f"{birth['year']:04d}-{birth['month']:02d}-{birth['day']:02d}"
    if birth.get("hour") is None:
        when = f"{day}（时辰不详）"
    elif birth.get("approximate"):
        when = f"{day} {birth.get('time_raw') or ''}（原文只给时辰，取中点 {birth['hour']:02d}:{birth.get('minute') or 0:02d}）"
    else:
        when = f"{day} {birth['hour']:02d}:{birth.get('minute') or 0:02d}"
    place = birth.get("place") or "原文未写"
    zone = birth.get("timezone") or "未知"
    gender = "男" if person["gender"] == "male" else "女"
    return f"性别：{gender}；出生（公历）：{when}；出生地：{place}；时区：{zone}"


def self_chart_messages(person):
    """Stage one of the no-tool conditions: the model derives the chart on its own."""
    inputs, convention, _assumed = engine_request(person)
    birth = person["birth"]
    parts = [f"出生时间：{inputs['datetime'].replace('T', ' ')}"]
    parts.append(f"时区：{birth['timezone']}" if birth.get("timezone") else "时区：（未提供）")
    parts.append(f"日柱流派：sect={convention['sect']}")
    return chat_messages(f"请计算四柱。{'，'.join(parts)}。\n{OUTPUT_HINTS['chart']}", P2_SYSTEM)


def self_chart_block(answer):
    """The model's own chart as a prompt line, or None if it did not give a valid one."""
    pillars = answer.get("pillars") if isinstance(answer, dict) else None
    if (
        isinstance(answer, dict)
        and answer.get("status") == "ok"
        and isinstance(pillars, dict)
        and set(pillars) == set(PILLARS)
        and all(v in CYCLE for v in pillars.values())
    ):
        return f"{SELF_CHART_LABEL} {pillars_text(pillars)}"
    return None


def question_messages(person, question, chart_block):
    user = mcq_user(
        profile_line(person),
        question["question"],
        question["options"],
        chart_block,
        person["context"],
        person["year"],
    )
    return chat_messages(user, MCQ_SYSTEM)


def tracks(persons):
    """Question-id sets of the reporting tracks."""
    every = [(p, q) for p in persons for q in p["questions"]]
    return {
        "full": {q["question_id"] for p, q in every},
        "bazi_only": {q["question_id"] for p, q in every if p["bazi"]},
        "official_key": {q["question_id"] for p, q in every if p["answer_source"] == "official"},
        "bazi_only_official_key": {
            q["question_id"] for p, q in every if p["bazi"] and p["answer_source"] == "official"
        },
        "contest8_2021_2026": {q["question_id"] for p, q in every if p["year"] >= 2021},
        "paper_years_2021_2025": {q["question_id"] for p, q in every if 2021 <= p["year"] <= 2025},
    }


def chance_level(persons, ids) -> float:
    sizes = [len(q["options"]) for p in persons for q in p["questions"] if q["question_id"] in ids]
    return sum(1 / n for n in sizes) / len(sizes)


def _index(persons):
    return {q["question_id"]: (p, q) for p in persons for q in p["questions"]}


def accuracy_table(persons, predictions, ids):
    """Accuracy over runs for one condition; unanswered questions are wrong.

    ``predictions`` maps run -> {question_id: letter or None}. Returns the mean
    over runs, the per-run values, the macro average over contest years (mean
    over years of the mean over runs, the BaziQA headline definition) and the
    majority vote with fractional credit for ties.
    """
    index = _index(persons)
    chosen = [qid for qid in index if qid in ids]
    runs = sorted(predictions)
    per_run, unanswered = [], 0
    by_year = defaultdict(lambda: defaultdict(list))
    for run in runs:
        hits = []
        for qid in chosen:
            person, question = index[qid]
            answer = predictions[run].get(qid)
            unanswered += answer is None
            hit = answer in question["accepted"]
            hits.append(hit)
            by_year[person["year"]][run].append(hit)
        per_run.append(sum(hits) / len(hits))
    votes = 0.0
    for qid in chosen:
        counts = defaultdict(int)
        for run in runs:
            answer = predictions[run].get(qid)
            if answer is not None:
                counts[answer] += 1
        if counts:
            top = max(counts.values())
            tied = [a for a, n in counts.items() if n == top]
            votes += sum(a in index[qid][1]["accepted"] for a in tied) / len(tied)
    year_means = {
        year: sum(sum(v) / len(v) for v in runs_.values()) / len(runs_)
        for year, runs_ in sorted(by_year.items())
    }
    mean = sum(per_run) / len(per_run)
    spread = (sum((x - mean) ** 2 for x in per_run) / len(per_run)) ** 0.5
    return {
        "n_questions": len(chosen),
        "n_runs": len(runs),
        "accuracy_mean": mean,
        "accuracy_sd_over_runs": spread,
        "accuracy_per_run": per_run,
        "macro_average_over_years": sum(year_means.values()) / len(year_means),
        "by_year": year_means,
        "majority_vote": votes / len(chosen),
        "unanswered_rate": unanswered / (len(chosen) * len(runs)),
        "chance": chance_level(persons, set(chosen)),
    }


def breakdown(persons, predictions, ids, key):
    """Mean accuracy over runs by a question attribute (year, category, person, method)."""
    groups = defaultdict(list)
    for person in persons:
        for question in person["questions"]:
            if question["question_id"] not in ids:
                continue
            value = {
                "year": person["year"],
                "category": question["category"],
                "person": person["person_id"],
                "method": person["method"] or "未注明",
                "answer_source": person["answer_source"],
            }[key]
            for run in predictions:
                groups[value].append(
                    predictions[run].get(question["question_id"]) in question["accepted"]
                )
    return {
        str(value): {"n": len(hits), "accuracy": sum(hits) / len(hits)}
        for value, hits in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def paired_by_person(persons, first, second, ids, resamples=5000, seed=1802026):
    """second - first accuracy with a person-level cluster bootstrap.

    Questions about one chart are correlated, so whole persons are resampled.
    Each person's value is the mean over its questions and runs.
    """
    rows = []
    for person in persons:
        qids = [q for q in person["questions"] if q["question_id"] in ids]
        if not qids:
            continue

        def hits(predictions, qids=qids):
            return [
                predictions[run].get(q["question_id"]) in q["accepted"]
                for run in predictions
                for q in qids
            ]

        a, b = hits(first), hits(second)
        rows.append((sum(b) / len(b) - sum(a) / len(a), len(qids)))
    total = sum(weight for _, weight in rows)
    estimate = sum(delta * weight for delta, weight in rows) / total
    rng, draws = random.Random(seed), []
    for _ in range(resamples):
        sample = [rng.choice(rows) for _ in rows]
        draws.append(sum(d * w for d, w in sample) / sum(w for _, w in sample))
    draws.sort()
    return {
        "n_persons": len(rows),
        "n_questions": total,
        "difference": estimate,
        "interval95": [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]],
        "resampling_unit": "person (chart)",
    }


def accuracy_interval(persons, predictions, ids, resamples=5000, seed=1802026):
    """Person-cluster bootstrap interval for one condition's mean accuracy."""
    rows = []
    for person in persons:
        qids = [q for q in person["questions"] if q["question_id"] in ids]
        if qids:
            hits = [
                predictions[run].get(q["question_id"]) in q["accepted"]
                for run in predictions
                for q in qids
            ]
            rows.append((sum(hits) / len(hits), len(qids)))
    rng, draws = random.Random(seed), []
    for _ in range(resamples):
        sample = [rng.choice(rows) for _ in rows]
        draws.append(sum(a * w for a, w in sample) / sum(w for _, w in sample))
    draws.sort()
    return [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]]


_PILLAR_CLAIM = re.compile(
    r"(年|月|日|时)柱\s*(?:为|是|：|:)?\s*([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
)
_MASTER_CLAIM = re.compile(r"日(?:主|元|干)\s*(?:为|是|：|:)?\s*([甲乙丙丁戊己庚辛壬癸])")
_YEAR_CLAIM = re.compile(
    r"([12][0-9]{3})\s*年?\s*[（(]?\s*(?:为|是|流年)?\s*([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
)
_GOD_CLAIM = re.compile(
    r"(年|月|时)干\s*([甲乙丙丁戊己庚辛壬癸])?\s*(?:为|是|乃)\s*(?:日主的)?\s*(比肩|劫财|食神|伤官|偏财|正财|七杀|正官|偏印|正印)"
)
_POSITION = {"年": "year", "月": "month", "日": "day", "时": "hour"}


def audit_explanation(text, pillars):
    """Check the verifiable claims in an explanation against the verified chart.

    Only statements with a fixed, checkable form are counted: a named pillar, the
    day master, the pillar of a calendar year, and the Ten God of a named stem.
    This measures factual consistency with the chart; it does not grade
    whether the interpretation is persuasive.
    """
    claims = []
    if not isinstance(text, str) or not pillars:
        return claims
    for position, pair in _PILLAR_CLAIM.findall(text):
        claims.append(("pillar", pair == pillars[_POSITION[position]]))
    for stem in _MASTER_CLAIM.findall(text):
        claims.append(("day_master", stem == pillars["day"][0]))
    for year, pair in _YEAR_CLAIM.findall(text):
        claims.append(("flow_year", pair == rules.year_pillar(int(year))))
    for position, _stem, god in _GOD_CLAIM.findall(text):
        stem = pillars[_POSITION[position]][0]
        claims.append(("ten_god", god == rules.ten_god(pillars["day"][0], stem)))
    return claims


def audit_summary(records, charts):
    """Share of verifiable claims that are right, per condition."""
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for record in records:
        pillars = charts.get(record["person_id"])
        for kind, right in audit_explanation(record.get("raw_output"), pillars):
            for key in (kind, "all"):
                totals[record["condition"]][key][0] += right
                totals[record["condition"]][key][1] += 1
    return {
        condition: {
            kind: {"n_claims": n, "correct": k, "rate": ratio(k, n)}
            for kind, (k, n) in kinds.items()
        }
        for condition, kinds in totals.items()
    }
