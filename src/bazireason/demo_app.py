"""BaZiReason local chat demo: verified charts plus a real local model.

    python -m bazireason.demo_app                       # http://127.0.0.1:8765, this machine only
    python -m bazireason.demo_app --share-lan           # same Wi-Fi, protected by a printed code

What it does
------------
* The four pillars always come from the verified calendar engine, never from
  the language model. A request the engine cannot answer shows its status and
  what is missing instead of a guessed chart.
* Questions are answered by a real local MLX model. The interface switches
  between the untuned base model, the Phase 2 adapter and the Phase 3 adapter;
  all three receive the same prompt.
* A question is phrased in the closest format the adapters were trained on
  (``bazireason.demo_deep.plan`` for the Phase 3 reasoning formats), and the
  reply's structured conclusion is compared with the rule code.
* Each answer is shown next to the rule cards it touches and the known
  uncertainties of the chart.

Privacy
-------
The server binds to 127.0.0.1 unless ``--share-lan`` is given. It uses only the
Python standard library, loads no remote resource, and keeps no record of birth
data or conversations. Feedback is written to a local file only after the
participant ticks an explicit consent box; conversation text is added only with
a second, separate consent.
"""

import argparse
import json
import queue
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bazireason import claim_check, deep_rules, demo_deep, rules
from bazireason import grounded as G
from bazireason import worked as W
from bazireason.calendar_engine import temporal_facts, tzdata_keys
from bazireason.charttools import chart_tool_text, pillars_text, verified_chart
from bazireason.conventions import PILLARS, Convention
from bazireason.p2_prompts import FINAL_MARK, OUTPUT_HINTS, P2_SYSTEM, facts_block
from bazireason.train_tasks import chart_line, compare_charts

STATIC = Path(__file__).parent / "demo_static"
DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"
# The experimental demo model: Phase 3 round 3, chosen after the Phase 3 test results had been
# inspected (results/phase3/final_choice.json). Round 2, the pre-registered validation winner,
# loads with --adapter; Phase 4 adapters likewise.
DEFAULT_ADAPTER = "artifacts/adapters/bazireason-qwen3-4b-lora-p3-r3"
DEFAULT_ADAPTER_P2 = "artifacts/adapters/bazireason-qwen3-4b-lora-v2"
# Phase 4 approach-A adapter, offered next to the demo model (not the default).
DEFAULT_ADAPTER_P4 = "artifacts/adapters/bazireason-qwen3-4b-lora-p4-a1"
MODEL_IDS = ("p3", "p4", "p2", "base")
COMMON_ZONES = (
    "Asia/Shanghai",
    "Asia/Hong_Kong",
    "Asia/Taipei",
    "Asia/Singapore",
    "Asia/Kuala_Lumpur",
    "Asia/Tokyo",
    "Asia/Seoul",
    "America/Los_Angeles",
    "America/New_York",
    "America/Chicago",
    "America/Toronto",
    "Europe/London",
    "Europe/Paris",
    "Australia/Sydney",
    "UTC",
)
PROFILE_NOTE = (
    "本演示只使用已验证的口径：固定 UTC+08:00 参考时钟、立春换年、十二节换月，不做真太阳时或经度修正，"
    "支持参考年份 1901–2099。这不是“出生地当地时间”排盘；在非 UTC+8 地区出生时，结果可能与按当地钟面"
    "排盘的软件不同。"
)
RULE_KEYWORDS = {
    "R-SS-01": ("十神", "比肩", "劫财", "食神", "伤官", "财", "官", "杀", "印"),
    "R-CG-01": ("藏干", "人元", "本气"),
    "R-WX-01": ("五行", "阴阳", "属性"),
    "R-WX-02": ("生克", "相生", "相克"),
    "R-HE-01": ("五合", "天干合", "相合", "天干冲"),
    "R-ZH-01": ("六合", "六冲", "相冲", "冲"),
    "R-ZH-02": ("三合", "半合", "三会"),
    "R-ZH-03": ("刑", "害", "破"),
    "R-KW-01": ("旬空", "空亡"),
    "R-WS-01": ("旺", "相", "休", "囚", "月令", "当令"),
    "R-CS-01": ("长生", "帝旺", "墓", "十二长生"),
    "R-GJ-01": ("格局", "格"),
    "R-QR-01": ("强", "弱", "得令", "得地", "得势", "身强", "身弱"),
    "R-DY-01": ("大运", "顺行", "逆行"),
    "R-DY-02": ("起运", "交运"),
    "R-LN-01": ("流年",),
    "R-LQ-01": ("父", "母", "妻", "夫", "子女", "兄弟", "六亲", "婚", "配偶"),
    "R-GZ-01": ("六十甲子", "干支"),
}
SPECULATIVE = (
    "会不会",
    "什么时候",
    "哪一年",
    "能不能",
    "是不是",
    "发财",
    "结婚",
    "离婚",
    "升职",
    "寿",
    "病",
    "克",
)


def birth_request(birth: dict):
    """(inputs, convention) from the form; nothing is defaulted that the form left empty."""
    date_part, time_part = (birth.get("date") or "").strip(), (birth.get("time") or "").strip()
    raw = None
    if date_part:
        raw = date_part + (f"T{time_part}" if time_part else "")
    inputs = {"datetime": raw, "timezone": (birth.get("timezone") or "").strip() or None}
    if birth.get("fold") in (0, 1):
        inputs["fold"] = birth["fold"]
    convention = Convention().to_dict()
    sect = birth.get("sect")
    if sect in (1, 2):
        convention["sect"] = sect
    else:
        convention.pop("sect")
    return inputs, convention


def uncertainties(inputs, convention, chart, facts):
    """Plain-language notes on what could change this chart. Deterministic, no model."""
    notes = [PROFILE_NOTE]
    if chart["status"] != "ok":
        explained = W.explain_request(inputs, convention, "facts")
        notes.insert(0, "未给出四柱：" + " ".join(explained["steps"]))
        return notes
    entry = facts["candidates"][0]
    reference = datetime.fromisoformat(entry["utc8_start"])
    wall = inputs["datetime"]
    if wall[:16] != entry["utc8_start"][:16]:
        notes.append(
            f"时区换算：所填钟面时间 {wall.replace('T', ' ')}（{inputs['timezone']}）对应参考时刻 "
            f"{entry['utc8_start'][:19].replace('T', ' ')}（UTC+08:00），四柱按后者计算。"
        )
    for key, word in (("previous_jie", "之后"), ("next_jie", "之前")):
        moment = datetime.fromisoformat(entry[key]["utc8"])
        gap = abs((reference - moment).total_seconds()) / 60
        if gap <= 180:
            affected = "年柱和月柱" if entry[key]["name"] == "立春" else "月柱"
            notes.append(
                f"临近节气：出生时刻在{entry[key]['name']}交节（{entry[key]['utc8'][:19].replace('T', ' ')}）"
                f"{word}约 {gap:.0f} 分钟。出生时间若有这么大的误差，{affected}会改变。"
            )
    minutes = reference.hour * 60 + reference.minute
    edge = min(abs(minutes - (60 * h)) for h in range(1, 26, 2))
    if edge <= 10:
        notes.append(f"临近时辰交界：距离时辰分界只有 {edge} 分钟，出生时间稍有出入时柱就会不同。")
    if reference.hour == 23:
        notes.append(
            "子时（23 点后）：日柱取决于流派，sect=1 记作次日，sect=2 仍记当日；"
            f"当前使用 sect={convention.get('sect')}。"
        )
    if len(facts["candidates"]) == 2:
        notes.append("该钟面时间因夏令时结束而重复，但两个可能的时刻给出相同的四柱。")
    if "luck" in chart:
        notes.append("大运的顺逆与排列由规则确定并经库核对；起运岁数按“三天折一年”，仅有单一来源。")
    notes.append(
        "格局与强弱只是按声明约定得到的分类标签，各流派判定不同；十神与六亲是传统的象征对应，不能确定现实中的人和事。"
    )
    return notes


def relevant_rules(text: str, limit=5):
    hits = []
    for rule_id, words in RULE_KEYWORDS.items():
        score = sum(word in text for word in words)
        if score:
            hits.append((score, rule_id))
    hits.sort(reverse=True)
    deep = demo_deep.relevant_cards(text, 3)
    return deep + [
        {
            k: rules.RULES_BY_ID[rule_id][k]
            for k in ("id", "kind", "title", "statement", "cross_check")
        }
        for _, rule_id in hits[: max(2, limit - len(deep))]
    ]


_POS = {"年": "year", "月": "month", "日": "day", "时": "hour"}


def engine_answer(message: str, chart: dict):
    """(label, value) the rule tables give for a recognised question, else None.

    Only questions with a single table-determined answer are recognised. The
    value has the same shape as the structured answer the model is trained to
    give, so the two can be compared directly.
    """
    pillars = chart["pillars"]
    day = pillars["day"][0]
    stem = re.search(r"(年|月|时)干.{0,12}十神", message)
    if stem:
        position = _POS[stem.group(1)]
        return f"{stem.group(1)}干{pillars[position][0]}的十神", rules.ten_god(
            day, pillars[position][0]
        )
    hidden = re.search(r"(年|月|日|时)支.{0,8}藏", message)
    if hidden:
        branch = pillars[_POS[hidden.group(1)]][1]
        value = [f"{h}:{rules.ten_god(day, h)}" for h in rules.HIDDEN_STEMS[branch]]
        return f"{hidden.group(1)}支{branch}的藏干与十神", value
    if re.search(r"合.{0,3}冲|冲.{0,3}合|刑.{0,2}害", message):
        return "原局的合冲刑害破", rules.chart_relations(pillars)
    if re.search(r"旬空|空亡", message):
        return "日柱旬空", rules.xun_kong(pillars["day"])
    if re.search(r"什么格|格局", message):
        return "月令本气取格（简化约定）", rules.month_pattern(pillars)
    if re.search(r"旺相休囚死|旺衰|月令", message):
        return "日主在月令的状态", rules.seasonal_state(
            rules.STEM_ELEMENT[day], pillars["month"][1]
        )
    if "大运" in message and re.search(r"顺|逆|前三步", message) and "luck" in chart:
        direction = chart["luck"]["direction"]
        first = [cycle["pillar"] for cycle in chart["luck"]["cycles"][:3]]
        return "大运方向与前三步", {"direction": direction, "first": first}
    return None


def check_structured(message: str, text: str, chart: dict):
    """Compare the reply's final JSON with the rule tables for a recognised question."""
    known = engine_answer(message, chart)
    if known is None:
        return None
    from bazireason.p2_eval import _same
    from bazireason.p2_prompts import extract_final_json

    label, truth = known
    answer = extract_final_json(text)
    value = None
    if isinstance(answer, dict):
        value = answer.get("answer") if "answer" in answer else None
        if value is None and len(answer) == 1:
            value = next(iter(answer.values()))
        if value is None and isinstance(truth, dict) and set(truth) <= set(answer):
            value = {key: answer[key] for key in truth}
    shown = truth if isinstance(truth, str) else json.dumps(truth, ensure_ascii=False)
    if isinstance(truth, list) and truth and ":" in truth[0] and truth[0][1] != ":":
        shown = "；".join(rules.relation_text(key) for key in truth)
    elif isinstance(truth, list) and not truth:
        shown = "无"
    return {"label": label, "engine": shown, "agrees": value is not None and _same(truth, value)}


def audit_reply(text: str, birth, message: str = "", first_turn: bool = True) -> dict:
    """Deterministic check of a model reply against the verified chart.

    The model is not trusted on facts the engine and rule tables can decide. Only
    fixed-form statements are checked (``bazireason.claim_check``: pillars, Ten Gods,
    hidden stems, relations, element counts, 旬空, life stages, stars, strength label,
    pattern, spouse star against the gender, book titles, loops), plus the direction of
    the luck cycles and the reply's structured conclusion.
    """
    if not birth:
        return {"checked": 0, "problems": []}

    chart = chart_payload(birth)["chart"]
    if chart["status"] != "ok":
        claimed = bool(re.search(r"[年月日时]柱\s*(?:为|是|：|:)?\s*[甲-癸][子-亥]", text))
        problems = (
            ["引擎没有给出四柱，但回答里写出了具体的柱；请以“未给出四柱”为准。"] if claimed else []
        )
        return {"checked": int(claimed), "problems": problems}
    # Every fixed-form chart and rule claim, checked by code with no model involved.
    claims = claim_check.check_claims(text, chart["pillars"], birth.get("gender"))
    problems = claim_check.problems(claims)
    checked = len(claims)
    direction = re.search(r"大运\s*(顺|逆)行", text)
    if direction:
        checked += 1
        if "luck" not in chart:
            problems.append("回答判断了大运顺逆，但没有填写性别，大运方向无法确定。")
        elif (direction.group(1) == "顺") != (chart["luck"]["direction"] == "forward"):
            truth = "顺行" if chart["luck"]["direction"] == "forward" else "逆行"
            problems.append(f"回答说大运{direction.group(1)}行，按规则应为{truth}。")
    planned = demo_deep.plan(message, chart, first_turn=first_turn) if message else None
    if planned is not None:
        structured = demo_deep.check(planned, text)
    else:
        structured = check_structured(message, text, chart) if message else None
    if structured:
        checked += 1
        if not structured["agrees"]:
            problems.append(
                f"结构化结论与规则引擎不符：{structured['label']}按规则应为 {structured['engine']}。"
            )
    shown = [c for c in claims if c["kind"] not in ("pillar", "day_master", "flow_year", "ten_god")]
    return {"checked": checked, "problems": problems, "engine": structured, "claims": shown}


def chart_payload(birth: dict, until_year=None):
    inputs, convention = birth_request(birth)
    gender = birth.get("gender") if birth.get("gender") in ("male", "female") else None
    chart = verified_chart(inputs, convention, gender, until_year)
    facts = temporal_facts(inputs)
    return {
        "inputs": inputs,
        "convention": convention,
        "chart": chart,
        "tool_text": chart_tool_text(chart),
        "facts": facts,
        "uncertainties": uncertainties(inputs, convention, chart, facts),
        # Phase 3 structure, computed by the rule code (never by the model).
        "deep": demo_deep.deep_lines(chart["pillars"]) if chart["status"] == "ok" else [],
    }


def compare_payload(first: dict, second: dict):
    a, b = chart_payload(first), chart_payload(second)
    result = {"a": a, "b": b, "changed": None, "listing": None}
    if a["chart"]["status"] == "ok" and b["chart"]["status"] == "ok":
        changed, listing = compare_charts(a["chart"]["pillars"], b["chart"]["pillars"])
        result.update(
            changed=changed,
            listing=listing,
            day_master_changed=a["chart"]["pillars"]["day"][0] != b["chart"]["pillars"]["day"][0],
        )
    return result


DERIVATION_WORDS = (
    "为什么",
    "怎么算",
    "怎样算",
    "如何算",
    "怎么来",
    "怎么得",
    "推一遍",
    "推导",
    "计算过程",
)
CHART_WORDS = ("柱", "四柱", "八字", "排盘")
RECHECK_WORDS = ("不对", "算错", "确定吗", "再核对", "重新检查", "再检查", "可靠吗")
UNVERIFIABLE = re.compile(
    r"(难以成功|注定|必定|必然会|一定会|肯定会|绝对会|克夫|克妻|克父|克母|短命|必有大|必遭|大富大贵)"
)


def route(message: str, first_turn: bool) -> str:
    """Which trained request format a question is closest to (keyword rules, no model)."""
    if not first_turn and any(word in message for word in RECHECK_WORDS):
        return "followup"
    if any(w in message for w in DERIVATION_WORDS) and any(w in message for w in CHART_WORDS):
        return "chart"
    if any(word in message for word in SPECULATIVE):
        return "policy"
    return "rule"


def first_turn_text(birth, message, kind):
    """The first user turn, phrased the way the model was trained to read it."""
    payload = chart_payload(birth)
    chart, inputs, convention = payload["chart"], payload["inputs"], payload["convention"]
    if chart["status"] != "ok" or kind == "chart":
        # A calendar request: the same wording and tool facts as the training data.
        parts = [
            f"出生时间：{inputs['datetime']}" if inputs["datetime"] else "出生时间：（未提供）"
        ]
        parts.append(f"时区：{inputs['timezone']}" if inputs["timezone"] else "时区：（未提供）")
        if "sect" in convention:
            parts.append(f"日柱流派：sect={convention['sect']}")
        if "fold" in inputs:
            parts.append(f"重复钟点取第{'一二'[inputs['fold']]}次（fold={inputs['fold']}）")
        ask = "请计算四柱。" if kind == "chart" else message
        text = (
            f"{ask}{'，'.join(parts)}。\n{facts_block(payload['facts'])}\n{OUTPUT_HINTS['chart']}"
        )
        return text, "chart"
    gender = {"male": "，男命", "female": "，女命"}.get(chart.get("gender"), "")
    return f"{chart_line(chart['pillars'])}{gender}。{message}\n{OUTPUT_HINTS[kind]}", kind


def compare_turn_text(first, second, message):
    """The comparison request in the trained counterfactual format."""
    both = compare_payload(first, second)
    lines = ["同一个人的两种出生信息："]
    for label, side in (("情形A", both["a"]), ("情形B", both["b"])):
        inputs, convention = side["inputs"], side["convention"]
        sect = f"，日柱流派：sect={convention['sect']}" if "sect" in convention else ""
        lines.append(f"{label}：出生时间：{inputs['datetime']}，时区：{inputs['timezone']}{sect}")
    lines.append("[排盘工具·已核验]")
    for label, side in (("情形A", both["a"]), ("情形B", both["b"])):
        chart = side["chart"]
        if chart["status"] == "ok":
            moment = chart["reference_utc8"][0][:19].replace("T", " ")
            lines.append(
                f"{label}：参考时刻 {moment}（UTC+08:00），四柱 {pillars_text(chart['pillars'])}"
            )
        else:
            lines.append(
                f"{label}：未给出四柱（{chart['status']}：{'、'.join(chart['reason_codes'])}）"
            )
    lines.append(message)
    lines.append(OUTPUT_HINTS["counterfactual"])
    return "\n".join(lines)


INTENT_NAMES = {
    "fact": "单项事实", "strength": "强弱计数", "pattern": "透干取格", "interact": "合冲刑害",
    "gods": "十神与组合", "overview": "整体结构", "yongshen": "用神（指定流派）", "ambig": "没有唯一答案",
    "star": "神煞", "flow": "流年", "domain": "传统看法（某一方面）", "prediction": "个人预测（不作答）",
    "followup": "追问", "unknown": "未识别（给出全部已核验事实）", "claims": "核对说法",
}  # fmt: skip


def grounded_chat(birth, history, message):
    """(messages, plan, chart) for the grounded mode, or None when there is no verified chart.

    Every user turn is routed (bazireason.intent); the model receives the chart's verified
    fact sheet on the first turn and the rule code's conclusion for each routed question.
    """
    chart = chart_payload(birth)["chart"] if birth else None
    if not chart or chart["status"] != "ok":
        return None
    pillars, gender = chart["pillars"], chart.get("gender")
    users = [m["content"] for m in history if m.get("role") == "user" and m.get("content")] + [message]
    replies = [m["content"] for m in history if m.get("role") == "assistant"]
    messages, plan = [{"role": "system", "content": P2_SYSTEM}], None
    for index, question in enumerate(users):
        plan = G.turn_plan(question, pillars, gender, plan, today=datetime.now().year)
        user = G.turn_user_text(question, plan, pillars, gender, "gr", index == 0, chart)
        messages.append({"role": "user", "content": user})
        if index < len(replies):
            messages.append({"role": "assistant", "content": replies[index]})
    return messages, plan, chart


def chat_messages(birth, history, message, compare=None):
    """(messages, format name) for the model.

    The verified chart is re-attached to the first user turn on every request,
    and each turn is phrased in one of the formats used in training. The model
    still writes every answer; the application only chooses the request wording.
    """
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    turns.append({"role": "user", "content": message})
    first = turns[0]["content"]
    kind = route(first, True)
    chart = chart_payload(birth)["chart"] if birth and not compare else None
    planned = demo_deep.plan(first, chart) if chart else None
    if compare:
        turns[0]["content"], kind = compare_turn_text(*compare, first), "counterfactual"
    elif planned:
        turns[0]["content"], kind = demo_deep.turn_text(planned, chart, True), planned["kind"]
    elif birth:
        turns[0]["content"], kind = first_turn_text(birth, first, kind)
    for index in range(2, len(turns), 2):
        later = route(turns[index]["content"], False)
        planned = (
            demo_deep.plan(turns[index]["content"], chart, first_turn=False)
            if chart and later != "followup"
            else None
        )
        # A later turn keeps its own format: re-check, prediction request or rule question.
        later_kind = later if later in ("followup", "policy") else "rule"
        if planned:
            turns[index]["content"] = demo_deep.turn_text(planned, chart, False)
        else:
            turns[index]["content"] += "\n" + OUTPUT_HINTS[later_kind]
        if index == len(turns) - 1:
            kind = planned["kind"] if planned else later_kind
    return [{"role": "system", "content": P2_SYSTEM}, *turns], kind


class Models:
    """The base model and the two adapters, served from one dedicated worker thread.

    MLX keeps per-thread state, and the HTTP server answers every request on a new
    thread, so loading and generation all happen on a single worker. Requests are
    queued, which also means one generation at a time.
    """

    def __init__(
        self, model_path, adapter_path, adapter_p2=DEFAULT_ADAPTER_P2, adapter_p4=DEFAULT_ADAPTER_P4
    ):
        self.model_path = model_path
        self.adapters = {"p3": adapter_path, "p4": adapter_p4, "p2": adapter_p2, "base": None}
        self.bank = None
        self.jobs = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def has(self, which):
        path = self.adapters.get(which)
        return which == "base" or bool(path and Path(path, "adapters.safetensors").exists())

    def available(self):
        return [
            {
                "id": "p3",
                "name": "BaZiReason 第三阶段（规则推理，演示默认）",
                "available": self.has("p3"),
            },
            {"id": "p4", "name": "BaZiReason 第四阶段 A1（实验）", "available": self.has("p4")},
            {"id": "p2", "name": "BaZiReason 第二阶段（历法）", "available": self.has("p2")},
            {"id": "base", "name": "未微调的基础模型", "available": True},
        ]

    def default(self):
        return next(m["id"] for m in self.available() if m["available"])

    def _get(self, which):
        """One base model in memory; the adapter is swapped in place (bazireason.adapter_bank)."""
        if self.bank is None:
            from bazireason.adapter_bank import AdapterBank

            first = self.adapters.get(which) or next(p for p in self.adapters.values() if p and Path(p).exists())
            self.bank = AdapterBank(self.model_path, first)
        self.bank.use(self.adapters[which] or "base")
        self.bank.model = which
        return self.bank

    def _work(self):
        while True:
            which, messages, max_tokens, temperature, out = self.jobs.get()
            try:
                chat = self._get(which)
                if messages is None:  # preload request
                    out.put({"loaded": which})
                    continue
                from mlx_lm.generate import stream_generate
                from mlx_lm.sample_utils import make_sampler

                last = None
                for last in stream_generate(
                    chat.net,
                    chat.tokenizer,
                    chat.prompt_ids(messages),
                    max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature, top_p=0.8 if temperature else 0.0),
                ):
                    out.put({"token": last.text})
                if last is not None:
                    out.put(
                        {
                            "done": True,
                            "stats": {
                                "prompt_tokens": last.prompt_tokens,
                                "completion_tokens": last.generation_tokens,
                                "tokens_per_second": round(last.generation_tps, 1),
                                "peak_memory_gb": round(last.peak_memory, 2),
                                "model": chat.model,
                                "adapter": chat.adapter_path,
                            },
                        }
                    )
            except Exception as error:  # report to the page instead of killing the worker
                out.put({"error": f"模型出错：{type(error).__name__}"})
            finally:
                out.put(None)

    def preload(self, which):
        out = queue.Queue()
        self.jobs.put((which, None, 0, 0.0, out))
        while out.get() is not None:
            pass

    def stream(self, which, messages, max_tokens, temperature):
        out = queue.Queue()
        self.jobs.put((which, messages, max_tokens, temperature, out))
        while (event := out.get()) is not None:
            yield event


def make_handler(models, access_code, feedback_path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BaZiReasonDemo/1"

        def log_message(self, *_args):  # never log request lines: they could hold birth data
            return

        def _authorised(self):
            if access_code is None:
                return True
            supplied = self.headers.get("X-Access-Code") or ""
            return secrets.compare_digest(supplied, access_code)

        def _json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > 200_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/meta":
                if not self._authorised():
                    return self._json({"error": "access code required", "needs_code": True}, 401)
                others = sorted(tzdata_keys() - set(COMMON_ZONES))
                return self._json(
                    {
                        "zones": list(COMMON_ZONES) + others,
                        "models": models.available(),
                        "profile_note": PROFILE_NOTE,
                        "rule_cards": [
                            {
                                k: card[k]
                                for k in ("id", "kind", "title", "statement", "cross_check")
                            }
                            for card in rules.RULE_CARDS
                        ]
                        + [demo_deep.card_view(card) for card in deep_rules.RULE_CARDS],
                        "formats": demo_deep.FORMAT_NAMES,
                        "needs_code": access_code is not None,
                    }
                )
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._authorised():
                return self._json({"error": "access code required", "needs_code": True}, 401)
            try:
                data = self._body()
                if self.path == "/api/chart":
                    return self._json(chart_payload(data.get("birth") or {}))
                if self.path == "/api/compare":
                    return self._json(compare_payload(data.get("a") or {}, data.get("b") or {}))
                if self.path == "/api/feedback":
                    return self._feedback(data)
                if self.path == "/api/chat":
                    return self._chat(data)
            except (ValueError, KeyError, TypeError) as error:
                return self._json({"error": f"bad request: {type(error).__name__}"}, 400)
            self._json({"error": "not found"}, 404)

        def _feedback(self, data):
            if data.get("consent") is not True:
                return self._json({"saved": False, "reason": "未勾选同意，未保存任何内容。"})
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": data.get("model"),
                "rating": data.get("rating"),
                "comment": str(data.get("comment") or "")[:2000],
            }
            if data.get("consent_conversation") is True:
                record["conversation"] = data.get("conversation")
                record["birth"] = data.get("birth")
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with feedback_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return self._json({"saved": True, "stored_conversation": "conversation" in record})

        def _chat(self, data):
            message = str(data.get("message") or "").strip()
            which = data.get("model") if data.get("model") in MODEL_IDS else models.default()
            if not message:
                return self._json({"error": "empty message"}, 400)
            if not models.has(which):
                return self._json({"error": "未找到这个适配器，请先训练或改用其他模型。"}, 409)
            compare = data.get("compare")
            compare = (compare["a"], compare["b"]) if isinstance(compare, dict) else None
            grounded = None
            if data.get("mode", "grounded") == "grounded" and not compare:
                grounded = grounded_chat(data.get("birth"), data.get("history") or [], message)
            if grounded:
                messages, plan, chart = grounded
                kind = plan["intent"].split(":")[0]
            else:
                messages, kind = chat_messages(
                    data.get("birth"), data.get("history") or [], message, compare
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def send(event):
                self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            started = time.perf_counter()
            try:
                send(
                    {
                        "rules": relevant_rules(message),
                        "speculative": any(word in message for word in SPECULATIVE),
                        "format": kind,
                        # What the model was actually asked, so the rewording is visible.
                        "asked": messages[-1]["content"],
                    }
                )
                temperature = float(data.get("temperature") or 0)
                reply = []
                for event in models.stream(
                    which, messages, int(data.get("max_tokens") or 900), temperature
                ):
                    if "token" in event:
                        reply.append(event["token"])
                    if event.get("done") and grounded:
                        event["verification"] = self._verify_grounded(
                            which, messages, "".join(reply), plan, chart, temperature, send
                        )
                        reply = [event["verification"]["final"]]
                    if event.get("done"):
                        event["stats"]["seconds"] = round(time.perf_counter() - started, 1)
                        first_question = next(
                            (
                                m["content"]
                                for m in data.get("history") or []
                                if m.get("role") == "user"
                            ),
                            message,
                        )
                        asked = message if kind != "followup" else first_question
                        if grounded:  # the routed plan is checked above; here the prose claims
                            asked = ""
                        event["audit"] = audit_reply(
                            "".join(reply),
                            data.get("birth"),
                            asked,
                            first_turn=not (data.get("history") or []) or kind == "followup",
                        )
                    send(event)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _verify_grounded(self, which, messages, reply, plan, chart, temperature, send):
            """Check the reply with code; one value-supplied repair; else the engine's derivation."""
            pillars, gender = chart["pillars"], chart.get("gender")
            first = G.verify_turn(plan, reply, pillars, gender)
            result = {
                "intent": plan["intent"],
                "label": INTENT_NAMES.get(plan["intent"].split(":")[0], plan["intent"]),
                "first_ok": first["ok"],
                "first_problems": first["problems"],
                "conclusion_checked": first["conclusion_agrees"] is not None,
                "repaired": False,
                "fallback": False,
                "final": reply,
                "engine": plan.get("text"),
            }
            if first["ok"] or plan.get("reference") is None:
                return result
            send({"status": "规则代码发现不一致，正在请模型按核验结果改正…"})
            second = "".join(
                e["token"]
                for e in models.stream(which, G.repair_messages(messages, reply, plan, first), 900, temperature)
                if "token" in e
            )
            check = G.verify_turn(plan, second, pillars, gender)
            result.update(repaired=True, second_ok=check["ok"])
            if check["ok"]:
                result["final"] = second
            else:
                result["fallback"] = True
                result["final"] = (
                    G.engine_answer(plan)
                    if plan["intent"] != "followup"
                    else "（规则代码的结论不变）\n" + (plan.get("text") or "")
                )
            return result

    return Handler


def split_answer(text: str):
    """(explanation, structured conclusion or None) of a reply in the trained format."""
    if FINAL_MARK in text:
        body, tail = text.rsplit(FINAL_MARK, 1)
        return body.strip(), tail.strip()
    return text.strip(), None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Phase 3 adapter folder")
    parser.add_argument("--adapter-p2", default=DEFAULT_ADAPTER_P2, help="Phase 2 adapter folder")
    parser.add_argument("--adapter-p4", default=DEFAULT_ADAPTER_P4, help="Phase 4 adapter folder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--share-lan",
        action="store_true",
        help="Listen on every interface so people on the same network can connect with a code",
    )
    parser.add_argument("--feedback-file", default="demo_feedback/feedback.jsonl")
    parser.add_argument("--preload", choices=("none", *MODEL_IDS, "all"), default="none")
    args = parser.parse_args()
    code = secrets.token_urlsafe(6) if args.share_lan else None
    models = Models(args.model, args.adapter, args.adapter_p2, args.adapter_p4)
    wanted = (
        MODEL_IDS if args.preload == "all" else () if args.preload == "none" else (args.preload,)
    )
    for which in wanted:
        if models.has(which):
            models.preload(which)
    host = "0.0.0.0" if args.share_lan else "127.0.0.1"
    server = ThreadingHTTPServer(
        (host, args.port), make_handler(models, code, Path(args.feedback_file))
    )
    print(f"BaZiReason demo: http://127.0.0.1:{args.port}", flush=True)
    if code:
        print(
            f"Shared on the local network. Give testers your address and this access code: {code}"
        )
        print("Their questions are processed on this computer; nothing is stored without consent.")
    else:
        print("Local only (127.0.0.1). Nothing is stored unless a tester consents to feedback.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


__all__ = ["PILLARS", "birth_request", "chart_payload", "chat_messages", "compare_payload", "main"]

if __name__ == "__main__":
    main()
