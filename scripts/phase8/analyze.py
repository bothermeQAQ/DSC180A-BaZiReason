"""Audit paired, fixed-checkpoint comparisons on the locked Phase 8 panels."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from bazireason import ladder
from bazireason.p2_eval import score_row
from graphs import calc, correct, parse, rows

MODELS = ("a", "b", "g", "t")
DRAWS = 10_000
SEED = 82143
OUT = ROOT / "results/phase8"
DIFFICULT = {"stage", "god"}


@lru_cache()
def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def indexed(path):
    records = list(rows(path))
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate IDs in {path}")
    return by_id


def adapter_digest(model):
    name = ("bazireason-qwen3-4b-lora-p6-u8" if model == "u8" else
            f"bazireason-phase{7 if model in ('g','t') else 8}-{model}")
    return digest(ROOT / "artifacts/adapters" / name / "adapters.safetensors")


def audited_panel(panel, models):
    data_phase = 7 if panel == "dev" else 8
    source = ROOT / f"data/phase{data_phase}/{panel}.jsonl"
    if panel == "dev":
        previous = json.loads((ROOT / "results/phase7/input_audit.json").read_text())
        expected = previous["sha256"]["data/phase7/dev.jsonl"]
    else:
        manifest = json.loads((source.parent / f"{panel}_manifest.json").read_text())
        expected = manifest["sha256"]
    assert digest(source) == expected, f"Locked panel mismatch at {source}"
    cases = indexed(source)
    provenance = {"inputs_sha256": digest(source), "replies_sha256": {}, "adapter_sha256": {}}
    for m in models:
        legacy = panel == "dev" and m in ("g", "t")
        path = ROOT / f"results/phase{7 if legacy else 8}/{m}_{panel}.jsonl"
        obs = indexed(path)
        if cases.keys() != obs.keys():
            raise ValueError(f"Incomplete or excess IDs in {path}")
        weight_hash = adapter_digest(m)
        provenance["replies_sha256"][m] = digest(path)
        provenance["adapter_sha256"][m] = weight_hash
        for key, item in cases.items():
            r = obs[key]
            assert r["group"] == item["group"] and r["condition"] == item["condition"], key
            assert r["adapter_hash"] == weight_hash and r["input_hash"] == digest(source), key
            if panel == "retention":
                fn = score_row if item["track"] == "calendar" else ladder.score
                sc = fn(item["item"], r["raw"])
                ok = bool(sc["correct"])
                if item["track"] == "calendar":
                    for flag in ("false_refusal", "definitive_on_nonanswer"):
                        assert sc[flag] == r["score"][flag], key
            else:
                ok = correct(r["raw"], item["answer"])
            assert isinstance(r["correct"], bool) and r["correct"] == ok, key
            item.setdefault("outcome", {})[m] = ok
            item.setdefault("responses", {})[m] = r
    return list(cases.values()), provenance


def edge(row):
    return row["source"] + "→" + row["op"]


def fraction(n, total):
    return {"right": int(n), "n": int(total), "pct": 100 * n / total if total else None}


def counts(cases, models):
    return {m: fraction(sum(r["outcome"][m] for r in cases), len(cases)) for m in models}


def quantiles(values):
    return [float(v) for v in np.percentile(values * 100, [2.5, 97.5])]


def graph_contrast(cases, left, right):
    """Bootstraps matched chart groups within each edge, then with edges resampled."""
    by_edge = defaultdict(lambda: defaultdict(list))
    wins = losses = 0
    for row in cases:
        delta = int(row["outcome"][left]) - int(row["outcome"][right])
        by_edge[edge(row)][row["group"]].append(delta)
        wins += delta == 1
        losses += delta == -1
    rng = np.random.default_rng(SEED)
    groups = [[np.mean(v) for v in charts.values()] for charts in by_edge.values()]
    point = np.array([np.mean(g) for g in groups])
    fixed = np.zeros(DRAWS)
    # Separate repeat draws ensure independent chart samples each time a family recurs.
    choices = rng.integers(len(groups), size=(DRAWS, len(groups)))
    broad = np.zeros(DRAWS)
    for idx, values in enumerate(groups):
        arr = np.array(values)
        samples = rng.choice(arr, (DRAWS, len(arr)), replace=True).mean(axis=1)
        fixed += samples / len(groups)
        draw_ids = np.nonzero(choices == idx)[0]
        blocks = rng.choice(arr, (len(draw_ids), len(arr)), replace=True).mean(axis=1)
        np.add.at(broad, draw_ids, blocks / len(groups))
    lofo = (point.sum() - point) / (len(point) - 1) if len(point) > 1 else point
    return {"pp": float(100 * point.mean()), "chart_ci": quantiles(fixed),
            "family_and_chart_ci": quantiles(broad),
            "leave_one_family_out_range": [float(100*min(lofo)), float(100*max(lofo))],
            "left_only": wins, "right_only": losses}


def retention_contrast(cases, left, right):
    """Row accuracy with the generated family mix fixed and whole events resampled."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in cases:
        groups[r["condition"]][r["group"]].append(
            int(r["outcome"][left]) - int(r["outcome"][right]))
    rng = np.random.default_rng(SEED)
    sim = np.zeros(DRAWS)
    wins = losses = 0
    for group in groups.values():
        deltas = list(group.values())
        wins += sum(v.count(1) for v in deltas)
        losses += sum(v.count(-1) for v in deltas)
        sizes = np.array([len(v) for v in deltas])
        sums = np.array([sum(v) for v in deltas])
        ix = rng.integers(len(sizes), size=(DRAWS, len(sizes)))
        sim += sums[ix].sum(1) / sizes[ix].sum(1) * sizes.sum() / len(cases)
    return {"pp": 100*(wins-losses)/len(cases), "chart_ci": quantiles(sim),
            "left_only": wins, "right_only": losses}


def endpoint(cases, models, retention=False):
    if not cases:
        raise ValueError("Expected endpoint has no rows")
    contrast = retention_contrast if retention else graph_contrast
    contrasts = {a+"-"+b: contrast(cases, a, b)
                 for i, a in enumerate(models) for b in models[i+1:]}
    return {"counts": counts(cases, models), "paired": contrasts}


def family_table(cases, models):
    buckets = defaultdict(list)
    for r in cases:
        buckets[edge(r)].append(r)
    return {k: counts(v, models) for k, v in sorted(buckets.items())}


def conditional_checks(cases, models):
    groups = defaultdict(dict)
    for r in cases:
        cell = groups[r["group"]]
        assert r["condition"] not in cell
        cell[r["condition"]] = r
    out = {}
    for m in models:
        cells = defaultdict(Counter)
        for rec in groups.values():
            base = rec["novel"]
            if base["op"] not in DIFFICULT:
                continue
            other = rec["switch"]
            solo = rec["consumer"]
            assert base["answer"] == solo["answer"]
            novel, switch, isolated = (r["outcome"][m] for r in (base, other, solo))
            different = other["answer"] != base["answer"]
            cells["differing_switch" if different else "same_value_switch"].update(
                n=1, both=int(novel and switch), neither=int(not novel and not switch))
            cells["hard"].update(n=1, novel=int(novel), isolated=int(isolated),
                                  both=int(novel and isolated), isolated_only=int(isolated and not novel),
                                  novel_only=int(novel and not isolated))
        out[m] = cells
    return out


def first_steps(cases, models):
    """A narrow trace check. Unrecognised wording remains unclassified."""
    out = {}
    for m in models:
        metrics = Counter()
        examples = []
        for r in cases:
            if r["outcome"][m]:
                continue
            raw = r["responses"][m]["raw"]
            line = next((s for s in raw.splitlines() if s.strip()), "")
            match = re.search(r"(?:可得|所得地支=)([子丑寅卯辰巳午未申酉戌亥])", line)
            metrics.update(misses=1, parsed_final=int(parse(raw) is not None),
                           recognized_first_step=int(bool(match)),
                           correct_first_branch=int(bool(match) and match[1] == r["branch"]))
            if len(examples) < 3:
                examples.append({"id": r["id"], "edge": edge(r), "gold": r["answer"],
                                 "branch": r["branch"], "raw": raw})
        out[m] = {"counts": metrics, "examples": examples}
    return out


def transfer(cases, models):
    panel = {}
    for condition in ("novel", "familiar", "switch", "consumer"):
        sample = [r for r in cases if r["condition"] == condition]
        panel[condition] = endpoint(sample, models)
        panel[condition]["by_family"] = family_table(sample, models)
    novel = [r for r in cases if r["condition"] == "novel"]
    for name, cond in (("day_dependent", True), ("other", False)):
        panel[name] = endpoint([r for r in novel if (r["op"] in DIFFICULT) == cond], models)
    forms = sorted({r["writer_form"] for r in novel})
    panel["novel_by_writer_form"] = {str(form): counts([r for r in novel if r["writer_form"] == form], models)
                                      for form in forms}
    alternate = {r["group"]: r for r in cases if r["condition"] == "familiar"}
    panel["matched_wordings"] = {m: dict(n=len(novel),
        both=sum(r["outcome"][m] and alternate[r["group"]]["outcome"][m] for r in novel),
        novel_only=sum(r["outcome"][m] and not alternate[r["group"]]["outcome"][m] for r in novel),
        familiar_only=sum(not r["outcome"][m] and alternate[r["group"]]["outcome"][m] for r in novel))
        for m in models}
    panel["matched_hard_charts"] = conditional_checks(cases, models)
    panel["novel_trace_on_misses"] = first_steps(novel, models)
    natal = sum(calc(r["op"], r["chart"], r["chart"][r["pos"]][1]) == r["answer"] for r in novel)
    panel["novel_natal_shortcut"] = fraction(natal, len(novel))
    return panel


def retention(cases, models):
    result = {}
    for track in ("rules", "calendar"):
        selected = [r for r in cases if r["track"] == track]
        result[track] = endpoint(selected, models, retention=True)
        subgroups = defaultdict(list)
        for r in selected:
            subgroups[r["condition"]].append(r)
        result[track]["by_task"] = {k: counts(v, models) for k, v in sorted(subgroups.items())}
    calendar = [r for r in cases if r["track"] == "calendar"]
    result["calendar_behavior"] = {m: {
        "answerable": sum(r["item"]["meta"]["expected"]["status"] == "ok" for r in calendar),
        "nonanswerable": sum(r["item"]["meta"]["expected"]["status"] != "ok" for r in calendar),
        **{flag: sum(r["responses"][m]["score"][flag] for r in calendar)
           for flag in ("false_refusal", "definitive_on_nonanswer")}
    } for m in models}
    return result


def screening(transfer_stats, retain):
    primary = transfer_stats["novel"]["paired"]
    checks = {"a_minus_b_at_least_4pp": primary["a-b"]["pp"] >= 4,
              "a_minus_b_fixed_edge_95_lower_above_zero": primary["a-b"]["chart_ci"][0] > 0,
              "a_minus_g_positive": primary["a-g"]["pp"] > 0}
    for track, threshold in [("rules", -1.5), ("calendar", -2)]:
        for reference in ("g", "u8"):
            delta = retain[track]["paired"]["a-" + reference]
            checks[f"a_minus_{reference}_{track}_meets_point_margin"] = delta["pp"] >= threshold
    return {"checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "comparison.json")
    args = parser.parse_args()
    result = {"bootstrap_draws": DRAWS, "bootstrap_seed": SEED, "provenance": {}}
    for panel in ("transfer", "retention", "dev"):
        models = MODELS + (("u8",) if panel == "retention" else ())
        cases, proven = audited_panel(panel, models)
        result["provenance"][panel] = proven
        if panel == "transfer":
            result[panel] = transfer(cases, models)
        elif panel == "retention":
            result[panel] = retention(cases, models)
        else:
            result[panel] = {}
            for condition in ("full", "twin", "alternate"):
                selected = [r for r in cases if r["condition"] == condition]
                result[panel][condition] = endpoint(selected, models)
                result[panel][condition]["by_family"] = family_table(selected, models)
    result["screening"] = screening(result["transfer"], result["retention"])
    temp = args.out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(args.out)


if __name__ == "__main__":
    main()
