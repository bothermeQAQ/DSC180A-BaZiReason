"""Secondary Phase 5 measures computed from saved replies (no model).

    .venv/bin/python scripts/phase5/extras5.py      # results/phase5/addenda/extras.json

1. Format compliance: the share of replies with a readable 结论JSON, and accuracy among those,
   so the untuned base is not judged only on format (execution plan, WP1 metrics).
2. Loops and book titles outside the rule cards, per system and condition.
3. Adapter routing (WP3): a per-intent adapter table chosen on development data, then measured
   on validation (and sealed, where the runs exist) against the best single adapter. Routing is
   simulated exactly from stored replies, since every item has a reply from every adapter.
4. Engine fallbacks: the runners exempt the engine's own derivation from the claim check, so
   every fallback reply shown is checked here.
"""

import json
from pathlib import Path

from bazireason import workflow as W
from bazireason.claim_check import CARD_WORKS
from bazireason.grounded import claim_failures

ROOT = Path("results/phase5")
SAFE = str.maketrans({":": "-", "/": "-"})
PARTS = ("rules", "indep", "blind")


def load(split, label, part, cond, tag=""):
    path = ROOT / split / label.translate(SAFE) / f"{part}_{cond}{tag}.jsonl"
    if not path.exists() and cond == "w0":
        path = path.with_name(f"{part}_w0_p2.jsonl")
    if not path.exists():
        return None
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def folders(split):
    return (
        sorted(p.name for p in (ROOT / split).iterdir() if p.is_dir())
        if (ROOT / split).exists()
        else []
    )


def label_of(folder):
    if folder.startswith("mix-"):
        _, a, b, alpha = folder.split("-", 3)
        return f"mix:{a}:{b}:{alpha}"
    return folder


def format_and_loops():
    out = {}
    for split in ("dev", "valid", "sealed"):
        for folder in folders(split):
            label = label_of(folder)
            for cond in ("w0", "gf", "go", "gr"):
                tag = "_p2" if split == "dev" and cond != "w0" else ""
                parts = [load(split, label, p, cond, tag) for p in PARTS]
                if not all(parts):
                    continue
                recs = [r for p in parts for r in p]
                parsed = [r for r in recs if r["score"].get("parsed")]
                out[f"{split}/{label}/{cond}{tag}"] = {
                    "n": len(recs),
                    "format_ok": len(parsed),
                    "correct": sum(bool(r["score"]["correct"]) for r in recs),
                    "correct_when_format_ok": sum(bool(r["score"]["correct"]) for r in parsed),
                    "loops": sum(W.has_loop(r["raw_output"]) for r in recs),
                    "titles_not_in_cards": sum(
                        bool(W.unknown_titles(r["raw_output"], CARD_WORKS)) for r in recs
                    ),
                }
    return out


def family(rec):
    return (rec.get("intent_oracle") or "?").split(":")[0]


def by_item(split, label, cond, tag=""):
    parts = [load(split, label, p, cond, tag) for p in PARTS]
    if not all(parts):
        return None
    return {r["id"]: r for p in parts for r in p}


def routing(cond, candidates, dev_tag):
    dev = {lab: by_item("dev", lab, cond, dev_tag) for lab in candidates}
    dev = {lab: v for lab, v in dev.items() if v}
    if len(dev) < 2:
        return None
    fams = sorted({family(r) for v in dev.values() for r in v.values()})
    table = {}
    for f in fams:
        score = {
            lab: sum(bool(r["score"]["correct"]) for r in v.values() if family(r) == f)
            for lab, v in dev.items()
        }
        table[f] = max(dev, key=lambda lab: (score[lab], lab == "p4a1"))  # ties to A1
    out = {"condition": cond, "candidates": list(dev), "table_from_dev": table}
    for split in ("valid", "sealed"):
        runs = {lab: by_item(split, lab, cond) for lab in dev}
        runs = {lab: v for lab, v in runs.items() if v}
        if len(runs) < 2:
            continue
        ids = sorted(set.intersection(*(set(v) for v in runs.values())))
        single = {lab: sum(bool(v[i]["score"]["correct"]) for i in ids) for lab, v in runs.items()}
        routed = sum(
            bool(
                runs.get(table.get(family(runs[next(iter(runs))][i])), runs["p4a1"])[i]["score"][
                    "correct"
                ]
            )
            for i in ids
        )
        best = max(single, key=single.get)
        oracle = sum(any(bool(v[i]["score"]["correct"]) for v in runs.values()) for i in ids)
        out[split] = {"n": len(ids), "single": single, "routed": routed, "best_single": best,
                      "routed_minus_best_single": routed - single[best], "oracle_any_adapter": oracle}  # fmt: skip
    return out


def fallback_check():
    import sys

    sys.path.insert(0, "scripts/phase5")
    import run5

    out = {"fallback_replies": 0, "with_a_contradicted_statement": 0, "examples": []}
    for split in ("valid", "sealed"):
        for part in PARTS:
            items = {i["id"]: i for i in run5.load_items(split, part)}
            for folder in folders(split):
                recs = load(split, label_of(folder), part, "grv")
                for r in recs or []:
                    if not r["fallback"]:
                        continue
                    meta = items[r["id"]]["meta"]
                    failures = claim_failures(r["final_text"], meta["pillars"], meta.get("gender"))
                    out["fallback_replies"] += 1
                    if failures:
                        out["with_a_contradicted_statement"] += 1
                        out["examples"].append({"split": split, "label": folder, "id": r["id"]})
    return out


def main():
    result = {
        "format_and_loops": format_and_loops(),
        "adapter_routing": {
            "w0": routing("w0", ["p4a1", "p5g1", "mix:p3r3:p4a1:0.15"], ""),
            "gr": routing("gr", ["p4a1", "p5g1"], "_p2"),
        },
        "engine_fallbacks": fallback_check(),
    }
    path = ROOT / "addenda" / "extras.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("adapter_routing", "engine_fallbacks")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
