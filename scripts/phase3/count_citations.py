"""Count book citations and repetition loops in the saved BaziQA replies (no model).

    python scripts/phase3/count_citations.py

A reply "names a book" when it contains 《…》; a title counts as unknown when it is
not in a list of real works on BaZi and related classics (the list is generous, so
the unknown count is a lower bound on invented titles). A repetition loop is a span
of at least 20 characters repeated three times in a row.
"""

import json
import re
from pathlib import Path

FOLDER = Path("results/phase3/baziqa1024")
MODELS = (("base", "base"), ("p2", "tuned"), ("p3", "tuned"), ("p3r3", "tuned"))
REAL = {
    "渊海子平", "淵海子平", "三命通会", "三命通會", "子平真诠", "子平真詮", "滴天髓", "穷通宝鉴",
    "窮通寶鑑", "五行大义", "五行大義", "滴天髓阐微", "滴天髓闡微", "命理约言", "神峰通考",
    "星平会海", "李虚中命书", "玉井奥诀", "兰台妙选", "千里命稿", "命理探原", "造化元钥", "栏江网",
    "欄江網", "继善篇", "繼善篇", "喜忌篇", "四言独步", "五言独步", "醉醒子", "子平粹言", "命理正宗",
    "三命指迷赋", "珞琭子", "珞琭子三命消息赋", "玉照定真经", "三命消息赋", "滴天髓征义",
    "命学讲义", "千里命钞", "子平四言集腋",
}  # fmt: skip
TITLE = re.compile(r"《([^》]{1,12})》")
QUOTE = re.compile(r"《([^》]{1,12})》[^。\n]{0,8}?[：:]\s*[“\"「]")
LOOP = re.compile(r"(.{20,}?)\1\1", re.S)


def main():
    out = {}
    for model, prefix in MODELS:
        for tools in ("no_tools", "tools"):
            path = FOLDER / model / f"{prefix}_{tools}.jsonl"
            if not path.exists():
                continue
            replies = [
                json.loads(line)["raw_output"] or "" for line in path.read_text().splitlines()
            ]
            out[f"{model}_{tools}"] = {
                "replies": len(replies),
                "names_a_book": sum(bool(TITLE.search(t)) for t in replies),
                "quotes_a_book": sum(bool(QUOTE.search(t)) for t in replies),
                "names_unknown_book": sum(
                    any(x not in REAL for x in TITLE.findall(t)) for t in replies
                ),
                "repetition_loop": sum(bool(LOOP.search(t)) for t in replies),
            }
            print(model, tools, out[f"{model}_{tools}"])
    (FOLDER / "citations_and_loops.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
