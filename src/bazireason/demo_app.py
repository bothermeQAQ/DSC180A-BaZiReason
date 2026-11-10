"""BaZiReason local chat demo: verified charts plus a real local model.

    python -m bazireason.demo_app                       # http://127.0.0.1:8765, this machine only
    python -m bazireason.demo_app --share-lan           # same Wi-Fi, protected by a printed code

What it does
------------
* The four pillars always come from the verified calendar engine, never from
  the language model. A request the engine cannot answer shows its status and
  what is missing instead of a guessed chart.
* Questions are answered by a real local MLX model. The interface switches
  between the untuned base model and the fine-tuned BaZiReason adapter; both
  receive the same prompt.
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
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bazireason import rules
from bazireason import worked as W
from bazireason.calendar_engine import temporal_facts, tzdata_keys
from bazireason.charttools import chart_tool_text, verified_chart
from bazireason.conventions import PILLARS, Convention
from bazireason.p2_prompts import FINAL_MARK, P2_SYSTEM, facts_block
from bazireason.train_tasks import compare_charts

STATIC = Path(__file__).parent / "demo_static"
DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"
DEFAULT_ADAPTER = "artifacts/adapters/bazireason-qwen3-4b-lora-v1"
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
    return [
        {
            k: rules.RULES_BY_ID[rule_id][k]
            for k in ("id", "kind", "title", "statement", "cross_check")
        }
        for _, rule_id in hits[:limit]
    ]


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


def chat_messages(birth, history, message, compare=None):
    """Prompt for the model: system rules, the verified tool output, then the dialogue."""
    context = []
    if birth:
        payload = chart_payload(birth, until_year=datetime.now().year + 1)
        context.append(payload["tool_text"])
        if payload["facts"]["candidates"] or payload["facts"]["normalization_error"]:
            context.append(facts_block(payload["facts"]))
    if compare:
        both = compare_payload(*compare)
        for label, side in (("情形A", both["a"]), ("情形B", both["b"])):
            context.append(f"{label}：\n{side['tool_text']}")
    turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    turns.append({"role": "user", "content": message})
    if context:
        # The tool output is attached to the first user turn so that follow-ups can refer to it.
        turns[0] = {"role": "user", "content": "\n".join(context) + "\n" + turns[0]["content"]}
    return [{"role": "system", "content": P2_SYSTEM}, *turns]


class Models:
    """Lazy holder for the base and the fine-tuned model; one generation at a time."""

    def __init__(self, model_path, adapter_path):
        self.model_path, self.adapter_path = model_path, adapter_path
        self.loaded, self.lock = {}, threading.Lock()

    def available(self):
        tuned = Path(self.adapter_path, "adapters.safetensors").exists()
        return [
            {"id": "tuned", "name": "BaZiReason（微调适配器）", "available": tuned},
            {"id": "base", "name": "未微调的基础模型", "available": True},
        ]

    def get(self, which):
        if which not in self.loaded:
            from bazireason.mlx_backend import MLXChatModel

            adapter = self.adapter_path if which == "tuned" else None
            self.loaded[which] = MLXChatModel(self.model_path, adapter, label=which)
        return self.loaded[which]

    def stream(self, which, messages, max_tokens, temperature):
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        with self.lock:
            chat = self.get(which)
            last = None
            for last in stream_generate(
                chat.net,
                chat.tokenizer,
                chat.prompt_ids(messages),
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature, top_p=0.8 if temperature else 0.0),
            ):
                yield {"token": last.text}
            if last is not None:
                yield {
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
                        ],
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
            which = data.get("model") if data.get("model") in ("base", "tuned") else "tuned"
            if not message:
                return self._json({"error": "empty message"}, 400)
            if which == "tuned" and not models.available()[0]["available"]:
                return self._json({"error": "未找到微调适配器，请先训练或改用基础模型。"}, 409)
            compare = data.get("compare")
            compare = (compare["a"], compare["b"]) if isinstance(compare, dict) else None
            messages = chat_messages(data.get("birth"), data.get("history") or [], message, compare)
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
                    }
                )
                temperature = float(data.get("temperature") or 0)
                for event in models.stream(
                    which, messages, int(data.get("max_tokens") or 900), temperature
                ):
                    if event.get("done"):
                        event["stats"]["seconds"] = round(time.perf_counter() - started, 1)
                    send(event)
            except (BrokenPipeError, ConnectionResetError):
                return

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
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--share-lan",
        action="store_true",
        help="Listen on every interface so people on the same network can connect with a code",
    )
    parser.add_argument("--feedback-file", default="demo_feedback/feedback.jsonl")
    parser.add_argument("--preload", choices=("none", "tuned", "base", "both"), default="none")
    args = parser.parse_args()
    code = secrets.token_urlsafe(6) if args.share_lan else None
    models = Models(args.model, args.adapter)
    for which in {"tuned": ["tuned"], "base": ["base"], "both": ["tuned", "base"], "none": []}[
        args.preload
    ]:
        models.get(which)
    host = "0.0.0.0" if args.share_lan else "127.0.0.1"
    server = ThreadingHTTPServer(
        (host, args.port), make_handler(models, code, Path(args.feedback_file))
    )
    print(f"BaZiReason demo: http://127.0.0.1:{args.port}")
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
