"""Build the Phase 6 development, validation and sealed sets (everything except blind wording).

    PYTHONPATH=src .venv/bin/python scripts/phase6/build_eval6.py            # write (never overwrites)
    PYTHONPATH=src .venv/bin/python scripts/phase6/build_eval6.py --check    # regenerate, compare hashes

Parts:
  rules.jsonl      Phase 3 rule families with an exact oracle plan, template wordings
                   (meta.template == 2 marks the wording held out of every training set)
  indep.jsonl      rule questions labelled by lunar-python, wordings never used in training
  calendar.jsonl   chart requests (answerable and non-answerable twins), model-alone track
  combo.jsonl      held-out combinations of rules (bazireason.splits6)
  cf.jsonl         counterfactual pairs: one character changed, the answer changes

Blocks (bazireason.splits6.phase6_years): dev6 and valid6 split the validation years;
sealed6 is the test block. Every prompt of every earlier data file is excluded, and the
validation and sealed sets also exclude every chart of an earlier evaluation split. For the
sealed set only counts and hashes are printed.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from bazireason import indep_rules as I
from bazireason import splits6 as S
from bazireason import train_data as TD
from bazireason import train_data3 as P3
from bazireason.indep_rules import eight_char
from bazireason.p5_eval import template_intent

OUT = Path("data/phase6")
# name: (generator split, rules (seed, scale), calendar (seed, scale), indep (seed, n), combo (seed, n), cf (seed, pairs))
PLAN = {
    "dev6": ("valid", (2026111611, 1.0), (2026111612, 2.0), (2026111613, 140), (2026111614, 100), (2026111615, 50)),
    "valid6": ("valid", (2026111621, 1.0), (2026111622, 2.4), (2026111623, 140), (2026111624, 100), (2026111625, 50)),
    "sealed6": ("test", (2026111631, 1.6), (2026111632, 3.3), (2026111633, 280), (2026111634, 160), (2026111635, 80)),
}  # fmt: skip
USED = sorted({str(p) for p in Path("data").glob("phase[2345]/**/*.jsonl")
               if "knowledge" not in str(p) and "attempts" not in p.name})  # fmt: skip


def first_user(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def used_prompts():
    seen = set()
    for path in USED:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("messages"):
                seen.add(first_user(row))
    return seen


def block_charts(block, seed, count, exclude):
    """(pillars, gender) from real dates in a Phase 6 year block, skipping excluded charts."""
    years = sorted(S.phase6_years()[block])
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    rng = random.Random(f"p6-charts:{seed}")
    out, keys = [], set()
    while len(out) < count:
        day = date(rng.choice(years), 1, 1) + timedelta(days=rng.randrange(365))
        if day in blocked or day.year not in years:
            continue
        chart = eight_char(
            datetime(day.year, day.month, day.day, rng.randrange(24), rng.randrange(60))
        )
        pillars = dict(
            zip(
                S.PILLARS,
                (chart.getYear(), chart.getMonth(), chart.getDay(), chart.getTime()),
                strict=True,
            )
        )
        key = S.chart_key(pillars)
        if key in exclude or key in keys:
            continue
        keys.add(key)
        out.append((pillars, rng.choice(("male", "female"))))
    return out


def build(name, seen, earlier):
    gen_split, rules_p, cal_p, ind_p, combo_p, cf_p = PLAN[name]
    exclude = earlier if name != "dev6" else set()
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    tag = {"dev6": "D", "valid6": "V", "sealed6": "S"}[name]
    with S.years_as(name):
        rules = []
        # The "test" generator includes the wording held out of every training set (template 2);
        # years_as() keeps the charts inside this Phase 6 block.
        for row in P3.generate_split(
            "test", seed=rules_p[0], blocked=blocked, scale=rules_p[1], version=1
        ):
            if row["family"].startswith("replay_") or first_user(row) in seen:
                continue
            if (
                S.chart_key(
                    row["meta"].get("pillars") or {"year": "", "month": "", "day": "", "hour": ""}
                )
                in exclude
            ):
                continue
            intent = template_intent(row)
            if intent:
                row["meta"] = dict(row["meta"], intent=intent)
                rules.append(row)
        calendar = [r for r in TD.generate_split(gen_split, cal_p[0], blocked, cal_p[1], 2)
                    if r["task"] == "chart" and "expected" in r["meta"] and first_user(r) not in seen]  # fmt: skip
        indep = I.build(gen_split, ind_p[1], ind_p[0], blocked, frozenset(seen))
    rng = random.Random(f"p6-combo:{combo_p[0]}")
    combo = []
    for i, (pillars, gender) in enumerate(block_charts(name, combo_p[0], combo_p[1], exclude)):
        kind = S.COMBO_KINDS[i % len(S.COMBO_KINDS)]
        combo.append(
            S.combo_item(
                f"P6{tag}-combo-{i + 1:05d}", name, kind, pillars, gender, rng, f"combo-{i}"
            )
        )
    rng = random.Random(f"p6-cf:{cf_p[0]}")
    cf, index = [], 0
    for pillars, gender in block_charts(name, cf_p[0], cf_p[1] * 4, exclude):
        if len(cf) >= 2 * cf_p[1]:
            break
        head = S.CF_INTENTS[index % len(S.CF_INTENTS)]
        pair = S.cf_pair(f"P6{tag}-cf-{index + 1:05d}", name, head, pillars, gender, rng)
        if pair and S.chart_key(pair[1]["meta"]["pillars"]) not in exclude:
            cf.extend(pair)
            index += 1
    for part, rows in (("rules", rules), ("calendar", calendar)):
        for i, row in enumerate(rows, 1):
            row["id"] = f"P6{tag}-{part}-{i:05d}"
            row["split"] = name
    for row in indep:
        row["id"] = row["id"].replace("IND-", f"P6{tag}-ind-")
        row["split"] = name
    return {"rules": rules, "indep": indep, "calendar": calendar, "combo": combo, "cf": cf}


def payload(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--splits", nargs="+", default=list(PLAN))
    args = parser.parse_args()
    seen, earlier = used_prompts(), S.earlier_eval_charts()
    bad = False
    for name in args.splits:
        parts = build(name, seen, earlier)
        folder = OUT / name
        manifest = {"block": name, "years": sorted(S.phase6_years()[name]), "plan": PLAN[name],
                    "excluded_prompts_from": USED, "excluded_earlier_eval_charts": len(earlier) if name != "dev6" else 0,
                    "files": {}}  # fmt: skip
        for part, rows in parts.items():
            data = payload(rows)
            digest = hashlib.sha256(data).hexdigest()
            path = folder / f"{part}.jsonl"
            families = Counter(
                (r["meta"].get("intent") or r["family"]).split(":")[0]
                if part != "calendar"
                else r["family"]
                for r in rows
            )
            manifest["files"][f"{part}.jsonl"] = {
                "rows": len(rows),
                "sha256": digest,
                "families": dict(sorted(families.items())),
            }
            if args.check:
                same = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                bad |= not same
                print(f"{path}: {'ok' if same else 'MISMATCH'}")
                continue
            if path.exists():
                sys.exit(f"{path} exists; Phase 6 splits are never rewritten")
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(
                f"{path}: {len(rows)} rows sha256 {digest[:16]}"
                + ("" if name == "sealed6" else f" {dict(families)}")
            )
        if not args.check:
            (folder / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
            )
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
