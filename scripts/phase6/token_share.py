"""Where the training tokens go: system prompt, user turns and replies, per training file.

    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/token_share.py
        -> results/phase6/token_share.json

Counts content tokens with the model's tokenizer (CPU only). The loss is computed on the
final reply only (mask_prompt), so the other tokens cost compute without a training signal.
Also checks that every stem and branch is one token and that no stem-branch pair is merged.
"""

import json
import sys
from pathlib import Path

from bazireason import train_data6 as T6
from bazireason.conventions import CYCLE
from bazireason.p2_prompts import P2_SYSTEM
from bazireason.prompts6 import SHORT_SYSTEM

OUT = Path("results/phase6/token_share.json")


def main():
    tok = T6._tokenizer()
    if tok is None:
        sys.exit("needs the tokenizers package (.venv-mlx)")

    def n(text):
        return len(tok.encode(text, add_special_tokens=False).ids)

    out = {
        "main_system_prompt_tokens": n(P2_SYSTEM),
        "short_system_prompt_tokens": n(SHORT_SYSTEM),
        "files": {},
    }
    for path in sorted(Path("data/phase6/train").glob("*/train.jsonl")):
        parts = {"system": 0, "user": 0, "earlier_replies": 0, "final_reply": 0}
        rows = main_rows = main_tokens = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            messages = json.loads(line)["messages"]
            rows += 1
            for i, m in enumerate(messages):
                k = n(m["content"])
                if m["role"] == "system":
                    parts["system"] += k
                    if m["content"] == P2_SYSTEM:
                        main_rows += 1
                        main_tokens += k
                elif m["role"] == "user":
                    parts["user"] += k
                elif i == len(messages) - 1:
                    parts["final_reply"] += k
                else:
                    parts["earlier_replies"] += k
        total = sum(parts.values())
        out["files"][path.parent.name] = {
            "rows": rows,
            "content_tokens": total,
            "tokens": parts,
            "share": {k: round(v / total, 4) for k, v in parts.items()},
            "rows_with_main_system_prompt": main_rows,
            "main_system_prompt_share": round(main_tokens / total, 4),
        }
    pairs = list(CYCLE)
    out["tokenization"] = {
        "single_characters_not_one_token": [c for c in sorted(set("".join(pairs))) if n(c) != 1],
        "pairs_merged_into_one_token": [p for p in pairs if n(p) == 1],
        "pairs_checked": len(pairs),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    brief = {k: (v["share"], v["main_system_prompt_share"]) for k, v in out["files"].items()}
    print(json.dumps({"files": brief, "tokenization": out["tokenization"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
