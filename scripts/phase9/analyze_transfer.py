"""Re-score the locked transfer panel and compare models on matched charts."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from graphs import R, calc, correct, parse, rows

DATA = ROOT / "data/phase9"
OUT = ROOT / "results/phase9"
MODELS = ("g", "a", "s", "x", "p")
PAIRS = ("x-s", "x-g", "p-x", "p-g", "s-g", "a-g", "p-s", "x-a", "p-a", "s-a")
EXPECTED = dict(novel=192, familiar=192, closed=96, producer=96, consumer=192, operation=96, deep=96)
DRAWS = 10000
SEED = 910227


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def weights_path(model):
    phase = "7" if model == "g" else "8" if model == "a" else "9"
    return ROOT / f"artifacts/adapters/bazireason-phase{phase}-{model}/adapters.safetensors"


def verified_cases():
    path = DATA / "transfer.jsonl"
    input_hash = digest(path)
    if not (DATA / "transfer_manifest.json").is_file():
        raise FileNotFoundError("Missing frozen transfer panel manifest")
    manifest = json.loads((DATA / "transfer_manifest.json").read_text())
    assert manifest["sha256"] == input_hash
    source = list(rows(path))
    by_id = {r["id"]: r for r in source}
    assert len(source) == len(by_id) == sum(EXPECTED.values())
    assert Counter(r["condition"] for r in source) == EXPECTED
    for r in source:
        r["ok"] = {}
        r["parsed"] = {}
    provenance = dict(input_sha256=input_hash, raw_reply_sha256={}, adapter_sha256={})
    for model in MODELS:
        raw_path = OUT / f"{model}_transfer.jsonl"
        weight_hash = digest(weights_path(model))
        phase = "7" if model == "g" else "8" if model == "a" else "9"
        summary_path = ROOT / f"results/phase{phase}/training/{model}/training_summary.json"
        assert json.loads(summary_path.read_text())["selected_adapter_sha256"] == weight_hash
        raw = list(rows(raw_path))
        assert len(raw) == len(by_id)
        assert {r["id"] for r in raw} == set(by_id)
        provenance["raw_reply_sha256"][model] = digest(raw_path)
        provenance["adapter_sha256"][model] = weight_hash
        for record in raw:
            case = by_id[record["id"]]
            assert record["condition"] == case["condition"] and record["group"] == case["group"]
            assert record["input_hash"] == input_hash and record["adapter_hash"] == weight_hash
            ok = correct(record["raw"], case["answer"])
            assert type(record["correct"]) is bool and record["correct"] == ok
            case["ok"][model] = ok
            case["parsed"][model] = parse(record["raw"])
    return source, provenance


def counts(cases):
    return {m: fraction(sum(r["ok"][m] for r in cases), len(cases)) for m in MODELS}


def fraction(k, n):
    return dict(right=int(k), n=n, pct=100*k/n if n else None)


def by(cases, key):
    bucket = defaultdict(list)
    for case in cases:
        bucket[key(case)].append(case)
    return {key: counts(items) for key, items in sorted(bucket.items())}


def percentiles(values):
    return list(map(float, np.percentile(100*values, [2.5, 97.5])))


def differences(cases, groups, models=MODELS, pairs=PAIRS):
    """Resample whole charts within producer and operation-pair stratum, then producers."""
    by_source = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for case in cases:
        group = case["group"]
        original = groups[group]
        source = original["novel"][0]["source"]
        stratum = tuple(sorted(r["op"] for r in original["novel"]))
        by_source[source][stratum][group].append([int(case["ok"][m]) for m in models])
    n_sources = len(by_source)
    assert n_sources == 4
    rng = np.random.default_rng(SEED)
    fixed = np.zeros((DRAWS, len(models)))
    broader = np.zeros_like(fixed)
    choices = rng.integers(n_sources, size=(DRAWS,n_sources))
    source_stats = []
    for i, strata in enumerate(by_source.values()):
        draw_ids = np.nonzero(choices == i)[0]
        num_cases = sum(len(records) for charts in strata.values() for records in charts.values())
        point = np.zeros(len(models))
        for charts in strata.values():
            lengths = {len(records) for records in charts.values()}
            assert len(lengths) == 1
            per_chart = lengths.pop()
            chart_stats = np.array([np.mean(replies, axis=0) for replies in charts.values()])
            k = len(charts)
            coeff = k * per_chart / num_cases
            point += coeff * chart_stats.mean(axis=0)
            fixed += (coeff/n_sources) * chart_stats[rng.integers(k,size=(DRAWS,k))].mean(axis=1)
            sample = chart_stats[rng.integers(k, size=(len(draw_ids),k))].mean(axis=1)
            np.add.at(broader, draw_ids, (coeff/n_sources)*sample)
        source_stats.append(point)
    source_stats = np.array(source_stats)
    contrasts = {}
    for contrast in pairs:
        l,r = contrast.split("-")
        ix,jx = models.index(l), models.index(r)
        d = source_stats[:,ix]-source_stats[:,jx]
        lofo = 100*(d.sum()-d)/(len(d)-1)
        wins = sum(case["ok"][l] and not case["ok"][r] for case in cases)
        losses = sum(case["ok"][r] and not case["ok"][l] for case in cases)
        contrasts[contrast] = dict(pp=float(100*d.mean()), chart_ci=percentiles(fixed[:,ix]-fixed[:,jx]),
                                   family_and_chart_ci=percentiles(broader[:,ix]-broader[:,jx]),
                                   leave_one_producer_out_range=[float(lofo.min()),float(lofo.max())],
                                   left_only=wins, right_only=losses)
    return contrasts


def validate_groups(cases):
    groups = defaultdict(lambda: defaultdict(list))
    for r in cases:
        groups[r["group"]][r["condition"]].append(r)
    assert len(groups) == 96
    for group in groups.values():
        assert {k:len(v) for k,v in group.items()} == {k:n//96 for k,n in EXPECTED.items()}
        primary = group["novel"]
        assert len({r["op"] for r in primary}) == 2 and primary[0]["answer"] != primary[1]["answer"]
        assert primary[0]["branch"] == primary[1]["branch"]
        for name in ("consumer", "familiar", "closed"):
            for r in group[name]:
                base = next(n for n in primary if n["op"] == r["op"])
                assert r["answer"] == base["answer"] and r["branch"] == base["branch"]
    return groups


def matched(groups):
    # Retain joint counts rather than treating multiple prompts on one chart as independent.
    results = {m:defaultdict(Counter) for m in MODELS}
    for group in groups.values():
        for m in MODELS:
            both = all(r["ok"][m] for r in group["novel"])
            measures = results[m]["operation_switch"]
            probes_right = group["producer"][0]["ok"][m] and all(r["ok"][m] for r in group["consumer"])
            measures.update(n=1, both=int(both), all_prereqs=int(probes_right),
                            both_correct_and_all_prereqs=int(probes_right and both))
            for base in group["novel"]:
                full = base["ok"][m]
                operation = base["op"]
                given = next(r for r in group["consumer"] if r["op"] == operation)["ok"][m]
                familiar = next(r for r in group["familiar"] if r["op"] == operation)["ok"][m]
                producer = group["producer"][0]["ok"][m]
                results[m]["prerequisites"].update(n=1, full=int(full), source=int(producer), consumer=int(given),
                    both_prereqs=int(producer and given), both_prereqs_but_full_wrong=int(producer and given and not full),
                    consumer_right_full_wrong=int(given and not full), full_right_consumer_wrong=int(full and not given))
                results[m]["wording"].update(n=1, both=int(familiar and full), novel_only=int(full and not familiar),
                                            familiar_only=int(familiar and not full))
                closed_probe = group["closed"][0]
                if closed_probe["op"] == operation:
                    closed = closed_probe["ok"][m]
                    planner = group["operation"][0]["ok"][m]
                    results[m]["reference"].update(n=1, reference=int(full), closed=int(closed), both=int(closed and full), reference_only=int(full and not closed),
                                                   closed_only=int(closed and not full))
                    results[m]["operation_probe"].update(n=1, both=int(planner and full),
                                                         probe_only=int(planner and not full),
                                                         full_only=int(full and not planner))
    for m, metrics in results.items():
        switches = metrics["operation_switch"]
        metrics["two_operations_given_all_prereqs"] = fraction(switches["both_correct_and_all_prereqs"], switches["all_prereqs"])
        target = metrics["prerequisites"]
        n = target["both_prereqs"]
        k = n - target["both_prereqs_but_full_wrong"]
        metrics["full_given_both_prereqs"] = fraction(k,n)
        ref = metrics["reference"]
        ref["reference_pct"] = 100 * ref["reference"] / ref["n"]
        ref["closed_pct"] = 100 * ref["closed"] / ref["n"]
        ref["reference_minus_closed_pp"] = ref["reference_pct"] - ref["closed_pct"]
    return results


def shortcut(novel):
    def right(r):
        pred = calc(r["op"], r["chart"], r["chart"][r["pos"]][1])
        return type(pred) is type(r["answer"]) and pred == r["answer"]
    return dict(natal_branch_skipping_producer=fraction(sum(right(r) for r in novel),len(novel)))


def answer_domain(value):
    """The four primary operation vocabularies are disjoint."""
    stems = "甲乙丙丁戊己庚辛壬癸"
    if isinstance(value, str):
        if value in R.LIFE_STAGES:
            return "stage"
        if value in {R.ten_god(a, b) for a in stems for b in stems}:
            return "god"
        if value in set(R.BRANCH_ELEMENT.values()):
            return "element"
    if type(value) is list and any(value == list(v) for v in R.HIDDEN_STEMS.values()):
        return "hidden"
    return "unparsed" if value is None else "other"


def error_classes(novel):
    results = {m:defaultdict(Counter) for m in MODELS}
    for case in novel:
        expected = case["op"]
        assert answer_domain(case["answer"]) == expected
        for m in MODELS:
            domain = answer_domain(case["parsed"][m])
            category = "correct" if case["ok"][m] else (
                "wrong_in_requested_domain" if domain == expected else "wrong_" + domain)
            results[m][expected][category] += 1
    return results


def summarize(cases):
    groups = validate_groups(cases)
    endpoints = {}
    for name in EXPECTED:
        condition = [case for case in cases if case["condition"] == name]
        endpoints[name] = dict(counts=counts(condition), paired=differences(condition, groups),
                               by_source=by(condition, lambda r: groups[r["group"]]["novel"][0]["source"]),
                               by_op=by(condition, lambda r: r["op"]))
    novel = [r for r in cases if r["condition"] == "novel"]
    for label, hard in [("day_dependent",True),("other",False)]:
        part = [r for r in novel if (r["op"] in ("stage","god")) == hard]
        endpoints[label] = dict(counts=counts(part), paired=differences(part, groups))
    changed = [r for r in novel if r["chart"][r["pos"]][1] != r["branch"]]
    changed_chart_ids = {r["group"] for r in changed}
    assert len(changed) == 2*len(changed_chart_ids)
    endpoints["changed_branch"] = dict(counts=counts(changed), paired=differences(changed, groups),
                                        shortcut=shortcut(changed))
    paired_op = [
        {"group": key, "ok": {m: all(r["ok"][m] for r in group["novel"]) for m in MODELS}}
        for key, group in groups.items()
    ]
    endpoints["both_operations_correct"] = dict(counts=counts(paired_op), paired=differences(paired_op, groups))
    selected_pairs = [r for r in paired_op if r["group"] in changed_chart_ids]
    endpoints["both_operations_changed_branch"] = dict(counts=counts(selected_pairs),
                                                       paired=differences(selected_pairs, groups))
    joint = matched(groups)
    for model in MODELS:
        paired_ref = []
        for key, group in groups.items():
            closed = group["closed"][0]
            full = next(r for r in group["novel"] if r["op"] == closed["op"])
            paired_ref.append(dict(group=key, ok=dict(ref=full["ok"][model], closed=closed["ok"][model])))
        joint[model]["reference"]["paired"] = differences(
            paired_ref, groups, models=("ref","closed"), pairs=("ref-closed",))["ref-closed"]
    operation_rows = [case for case in cases if case["condition"] == "operation"]
    for model in MODELS:
        valid_choice = sum(isinstance(r["parsed"][model], str) and r["parsed"][model] in r["options"]
                           for r in operation_rows)
        joint[model]["operation_probe"]["valid_option_count"] = valid_choice
    change = endpoints["novel"]["paired"]
    gate = dict(at_least_four_pp=change["x-s"]["pp"] >= 4, chart_lower_positive=change["x-s"]["chart_ci"][0] > 0,
                improvement_over_parent=change["x-g"]["pp"] > 0)
    return dict(endpoints=endpoints, joint=joint, errors_by_requested_operation=error_classes(novel),
                novel_by_edge=by(novel, lambda r:r["source"]+"→"+r["op"]),
                novel_by_writer=by(novel, lambda r:str(r["writer_form"])),
                shortcut=shortcut(novel),
                primary_screen=dict(checks=gate, passes=all(gate.values())),
                bootstrap=dict(replicates=DRAWS, seed=SEED, charts=96, producers=4,
                               strata="Within each primary producer and pair of requested operations; cluster by chart."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT/"transfer_comparison.json")
    args = parser.parse_args()
    cases, provenance = verified_cases()
    result = summarize(cases)
    result["provenance"] = provenance
    temp = args.out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(args.out)


if __name__ == "__main__":
    main()
