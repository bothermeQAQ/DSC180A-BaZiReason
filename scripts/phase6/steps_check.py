"""Derivation version 2 against version 1 (CPU only): same answers, no wrong statement, length.

    PYTHONPATH=src .venv-mlx/bin/python scripts/phase6/steps_check.py
        -> results/phase6/steps_v2_check.json

For random charts and every question type whose derivation changed, checks that the final
JSON equals the rule code's answer in both versions and that the claim checker finds no
wrong statement, and records the reply length in tokens (the reply cap is 1,100).
"""

import json
import random
import sys
from pathlib import Path

from bazireason import splits6 as S
from bazireason import steps6 as S6
from bazireason import train_data6 as T6
from bazireason.grounded import claim_failures
from bazireason.p2_prompts import extract_final_json

OUT = Path("results/phase6/steps_v2_check.json")
INTENTS = ("strength", "yongshen:扶抑", "flow:2037", "flow:1999", "domain:marriage:2031",
           "domain:wealth:2029", "domain:children:2027", "domain:career:2040", "overview")  # fmt: skip
CHARTS = 1000


def main():
    tok = T6._tokenizer()
    if tok is None:
        sys.exit("needs the tokenizers package (.venv-mlx)")
    rng = random.Random(20261118)
    out = {
        "charts": CHARTS,
        "derivations": 0,
        "answers_differ": 0,
        "claim_failures": 0,
        "tokens": {},
    }
    for _ in range(CHARTS):
        pillars, gender = S.random_pillars(rng), rng.choice(("male", "female"))
        for intent in INTENTS:
            texts = {}
            for version in (1, 2):
                S6.set_steps(version)
                try:
                    derived = S6.derive(intent, pillars, gender)
                finally:
                    S6.set_steps(1)
                if derived is None:
                    break
                texts[version] = derived
            if len(texts) < 2:
                continue
            out["derivations"] += 1
            out["answers_differ"] += texts[1][1] != texts[2][1] or extract_final_json(texts[2][0]) != texts[2][1]  # fmt: skip
            out["claim_failures"] += bool(claim_failures(texts[2][0], pillars, gender))
            cell = out["tokens"].setdefault(intent.split(":")[0], {"v1_max": 0, "v2_max": 0, "v1_sum": 0, "v2_sum": 0, "n": 0})  # fmt: skip
            for version in (1, 2):
                k = len(tok.encode(texts[version][0], add_special_tokens=False).ids)
                cell[f"v{version}_max"] = max(cell[f"v{version}_max"], k)
                cell[f"v{version}_sum"] += k
            cell["n"] += 1
    for cell in out["tokens"].values():
        cell["v1_mean"] = round(cell.pop("v1_sum") / cell["n"])
        cell["v2_mean"] = round(cell.pop("v2_sum") / cell["n"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
