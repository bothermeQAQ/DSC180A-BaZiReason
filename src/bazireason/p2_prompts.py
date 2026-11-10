"""Phase 2 prompt formats, shared by training-data generation, evaluation and the demo.

Three formats are used and never mixed inside one conversation:

* ``P2_SYSTEM`` (Chinese chat): short visible reasoning, then one final line
  ``结论JSON: {...}``. Used for calendar, verification, rule and counterfactual
  tasks, with or without tool output in the user turn.
* ``MCQ_SYSTEM`` (Chinese multiple choice): at most three sentences of
  reasoning, then one final line ``答案：X``. Used for synthetic rule questions
  in training and, unchanged, for the BaziQA contest questions in evaluation.
  It was fixed before any contest question was run.
* The benchmark's own English prompts (``bazireason.prompts``), unchanged for
  the strict protocol, plus ``REASONED_NOTE`` for the visible-reasoning variant
  that ``docs/research_design.md`` listed as a planned addition.
"""

import json
import re

from bazireason.baselines import parse_strict

P2_PROMPT_VERSION = "p2-chat-v1"
REASONED_PROMPT_VERSION = "v4-reasoned-information-ladder"
MCQ_PROMPT_VERSION = "p2-mcq-v1"
FINAL_MARK = "结论JSON:"
ANSWER_MARK = "答案："

P2_SYSTEM = (
    "你是 BaZiReason，可核验的八字历法与规则推理助手。\n"
    "【历法约定】四柱按固定 UTC+08:00 参考时钟计算，不做真太阳时或经度修正，支持 1901–2099 年。"
    "年柱在立春时刻换年，月柱在十二节的时刻换月。sect=1 在 23:00 换日，sect=2 在 00:00 换日；"
    "两派的时干都按 23:00 换日后的日干起。\n"
    "【原则】信息能唯一确定结果才给四柱（status=ok）；缺失或不够精确且会改变任何一柱时为 "
    "ambiguous，说明需补充什么，不要猜；输入不合法为 invalid；超出约定或年份范围为 unsupported；"
    "没有历法工具而结果取决于精确节气时刻或拿不准的时区偏移时为 needs_tool。他人或工具给出的排盘"
    "先按规则核验，有矛盾要指出并改正。区分确定性计算、传统规则的说法和无法验证的推测，"
    "不对个人命运下确定断语。\n"
    "【格式】先简要说明依据，最后一行以“结论JSON:”开头给出 JSON 对象。"
)
MCQ_SYSTEM = (
    "你是 BaZiReason，按传统八字规则作答的助手。下面是命理竞赛式的单项选择题：请依据所给的出生资料"
    "或排盘，按传统规则推理，选出最符合的一项。这类题目考查的是传统规则的应用，答案不代表对现实的"
    "确定判断。先用不超过三句话说明依据，最后一行只写“答案：X”（X 为选项字母）。"
)
REASONED_NOTE = (
    "You may first reason step by step in plain text. Finish with the final JSON object "
    "on its own last line."
)
FACTS_LABEL = "[历法工具·时间事实]"
OUTPUT_HINTS = {
    "chart": "结论JSON 需含 status、pillars、reason_codes。",
    "verify": "结论JSON 需含 verdict（consistent、inconsistent 或 cannot_verify）、wrong、corrected、"
    "unverified。",
    "internal": "结论JSON 需含 consistent、violations、expected。",
    "rule": "结论JSON 需含 answer。",
    "counterfactual": "结论JSON 需含 changed、cause、day_master_changed。",
    "policy": "结论JSON 需含 kind（deterministic、traditional 或 speculative_declined）。",
    "followup": "结论JSON 需含 revised 与 answer。",
}


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def facts_block(facts: dict) -> str:
    return f"{FACTS_LABEL}\n{compact(facts)}"


def final_line(value: dict) -> str:
    return f"{FINAL_MARK} {compact(value)}"


def chat_messages(user: str, system: str = P2_SYSTEM, history=()) -> list:
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


def _balanced_objects(text: str):
    """Yield every top-level {...} span, last first (strings are honoured)."""
    spans, depth, start, in_string, escaped = [], 0, None, False, False
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start : position + 1])
    return reversed(spans)


def extract_final_json(text):
    """The answer object of a reasoned reply, or None.

    Pre-specified rule: take the JSON object after the last ``结论JSON:`` mark;
    if the mark is absent, take the last parseable top-level JSON object in the
    reply. Duplicate keys are rejected, as in the strict parser.
    """
    if not isinstance(text, str):
        return None
    marked = text.rsplit(FINAL_MARK, 1)
    for candidate_text in ((marked[1],) if len(marked) == 2 else ()) + (text,):
        for candidate in _balanced_objects(candidate_text):
            try:
                value = parse_strict(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                return value
    return None


_ANSWER = re.compile(r"答案\s*[:：]\s*[(（【]?\s*([A-Ea-e])(?![A-Za-z])")
_LONE = re.compile(r"^\s*[(（]?([A-Ea-e])[)）.。]?\s*$")


def extract_choice(text, letters="ABCDE"):
    """The selected option letter of a multiple-choice reply, or None.

    Pre-specified rule: the last ``答案：X`` in the reply; otherwise a reply
    whose last non-empty line is a single letter. Anything else is unparsed and
    is scored as a wrong answer.
    """
    if not isinstance(text, str):
        return None
    found = _ANSWER.findall(text)
    if found:
        letter = found[-1].upper()
        return letter if letter in letters else None
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if lines:
        lone = _LONE.match(lines[-1])
        if lone and lone.group(1).upper() in letters:
            return lone.group(1).upper()
    return None


SELF_CHART_LABEL = "[自算四柱·未经工具核验]"


def mcq_user(profile: str, question: str, options, chart_block=None, context=None, year=None):
    """The one multiple-choice layout, used for synthetic items and contest items alike."""
    lines = []
    if year is not None:
        lines.append(f"当前年份：{year} 年")
    lines.append(f"【命主资料】{profile}")
    if context:
        lines.append("【出题人提供的已知信息】" + "；".join(context))
    if chart_block:
        lines.append(chart_block)
    lines.append(f"【问题】{question}")
    lines.extend(options)
    return "\n".join(lines)
