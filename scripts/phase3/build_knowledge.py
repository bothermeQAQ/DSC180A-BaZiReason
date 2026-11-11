"""Export the Phase 3 rule cards and check their quotations against the source pages.

    python scripts/phase3/build_knowledge.py            # write data/phase3/knowledge/rules_v2.json
    python scripts/phase3/build_knowledge.py --fetch    # download the Wikisource pages first

The pages are public-domain texts as transcribed on zh.wikisource.org. They are
kept under the git-ignored ``external/classics`` folder; the export records
each page's SHA-256 so a later reader can tell whether the transcription moved.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from bazireason import deep_rules, rules

PAGES = {
    "yuanhai.txt": ["淵海子平"],
    "sanming2.txt": ["三命通會/卷二"],
    "sanming3.txt": ["三命通會/卷三"],
    "qiongtong.txt": ["穷通宝鉴"],
    "ditian.txt": [f"滴天髓/{n:02d}" for n in (1, 2, 3, 9, 10, 11, 12)],
}
RAW = "https://zh.wikisource.org/w/index.php"


def fetch(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for name, titles in PAGES.items():
        parts = []
        for title in titles:
            parts.append(
                subprocess.run(
                    ["curl", "-s", "-A", "research-script/1.0", "--get", RAW]
                    + ["--data-urlencode", f"title={title}", "--data-urlencode", "action=raw"],
                    check=True,
                    capture_output=True,
                ).stdout
            )
        (folder / name).write_bytes(b"".join(parts))


def squeeze(text: str) -> str:
    return "".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classics", default="external/classics")
    parser.add_argument("--output", default="data/phase3/knowledge/rules_v2.json")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    folder = Path(args.classics)
    if args.fetch:
        fetch(folder)
    texts, hashes = {}, {}
    for name in PAGES:
        path = folder / name
        if path.exists():
            payload = path.read_bytes()
            texts[name] = squeeze(payload.decode("utf-8"))
            hashes[name] = hashlib.sha256(payload).hexdigest()
    checked = missing = 0
    for card in deep_rules.RULE_CARDS:
        for quote in card["quotations"]:
            text = texts.get(quote["local_file"])
            if text is None:
                quote["verified"] = None
                continue
            quote["verified"] = squeeze(quote["text"]) in text
            checked += 1
            if not quote["verified"]:
                missing += 1
                print(f"NOT FOUND in {quote['local_file']}: {card['id']} {quote['text']}")
    export = {
        "version": "rules_v2",
        "note": (
            "Phase 3 rule cards. Quotations are verbatim from the zh.wikisource.org "
            "transcriptions named in source_pages (retrieved 2026-11-11), including any "
            "transcription errors there. Statements are this project's paraphrases. Phase 2 "
            "cards (rules_v1) are unchanged and listed by id only."
        ),
        "source_pages": {
            name: {
                "wikisource_titles": titles,
                "sha256_of_downloaded_wikitext": hashes.get(name),
            }
            for name, titles in PAGES.items()
        },
        "kinds": sorted({card["kind"] for card in deep_rules.RULE_CARDS}),
        "cards": list(deep_rules.RULE_CARDS),
        "phase2_card_ids": [card["id"] for card in rules.RULE_CARDS],
        "tables": {
            "禄": deep_rules.LU,
            "阳刃": deep_rules.YANG_REN,
            "天乙贵人": deep_rules.TIAN_YI,
            "驿马": deep_rules.YI_MA,
            "桃花": deep_rules.XIAN_CHI,
            "华盖": deep_rules.HUA_GAI,
            "十神五类": deep_rules.GOD_CLASS,
            "十神组合": [list(map(list, c[1:3])) + [c[0], c[3]] for c in deep_rules.CONFIGS],
            "格局顺逆": deep_rules.PATTERN_USE,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(deep_rules.RULE_CARDS)} cards, {checked} quotations checked, {missing} not found")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
