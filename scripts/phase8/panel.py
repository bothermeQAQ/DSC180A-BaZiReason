"""Build or verify the closed Phase 8 transfer panel without reading model replies."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase7"))

from bazireason.conventions import PILLARS
from bazireason.indep_rules import eight_char
from bazireason.p5_eval import head_line
from bazireason.train_data import split_years
from graphs import HINTS, calc, case, derived, historical_keys, independent, pool, rows, write_rows

DATA = ROOT / "data/phase8"
PANEL = DATA / "transfer.jsonl"
MANIFEST = DATA / "transfer_manifest.json"
SEED = 810821
PER_FAMILY = 20
HARD = {"stage", "god"}
FOUR = re.compile("[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]")
POSITION = {"year": "年支", "day": "日支"}
# Independent wording supplied on the evaluation side. Preserve the response contract.
OUTPUT_REQUEST = {
    "element": ("这个支归到金木水火土中的哪一类？", "说出该支的五行。"),
    "main": ("这个支里哪一个藏干是本气？", "所问的是该支的本气天干。"),
    "hidden": ("里面藏着哪些天干？按本气在前的常用次序列全。", "请依藏干表的本气、中气、余气顺序列出该支的所有藏干。"),
    "number": ("这个支单独算有几项藏干？", "请报出该支所藏天干的总数。"),
    "stage": ("本盘日干在这个支上处于十二长生的哪一宫？", "就日主而言，该支对应的十二运星叫什么？"),
    "god": ("这个支取本气藏干，以本盘日干为我，该藏干属于哪个十神？",
            "请判断该支本气天干与本盘日主的十神关系，报十神名。"),
    "positions": ("再看原盘的四个地支，哪些柱的地支和这个支同五行？按年月日时先后列柱名。",
                  "以该支所属的五行为条件，从原局年、月、日、时四柱里找出地支符合的柱，按原顺序列名。"),
    "count": ("回到四柱，地支与这个支同属一行的柱子一共几柱？",
              "逐一核对原局四个地支，和该支五行一致的有几个？"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chart_key(chart):
    return "".join(chart[p] for p in PILLARS)


def keys_in(record):
    """Extract only chart identifiers, including Phase 7 graph rows that have no chart metadata."""
    item = record.get("item", record)
    metadata = item.get("meta", {})
    keys = set()
    for candidate in (record.get("chart"), item.get("pillars"), metadata.get("pillars")):
        if isinstance(candidate, dict) and all(isinstance(candidate.get(p), str) for p in PILLARS):
            keys.add(chart_key(candidate))
    for message in record.get("messages", []):
        if message["role"] == "user":
            for line in message["content"].splitlines():
                matches = FOUR.findall(line)
                if len(matches) >= 4:
                    keys.add("".join(matches[:4]))
            break
    return keys


def all_old_keys():
    keys = historical_keys()
    paths = list((ROOT / "data").glob("phase[234567]/**/*.jsonl"))
    paths += list((ROOT / "results/phase6").glob("own_replies*.jsonl"))
    for path in sorted(paths):
        for record in rows(path):
            keys.update(keys_in(record.get("row",record)))
    return keys


def new_chart(rng, years, used):
    while True:
        year = rng.choice(years)
        start = datetime(year, 1, 1)
        days = (datetime(year + 1, 1, 1) - start).days
        moment = start + timedelta(days=rng.randrange(days), minutes=rng.randrange(24 * 60))
        ec = eight_char(moment)
        chart = dict(zip(PILLARS, (ec.getYear(), ec.getMonth(), ec.getDay(), ec.getTime())))
        key = chart_key(chart)
        if key not in used:
            used.add(key)
            return year, chart


def question(chart, source, op, pos, style, branch=None):
    base = POSITION[pos]
    if branch is not None:
        setup = (f"已指定本题查的支为{branch}。", f"本题另取地支{branch}作为参照。")[style]
    elif source in {"华盖", "桃花", "驿马"}:
        setup = (f"这盘按{base}取{source}，以查出的{source}支为准。",
                 f"单从{base}起{source}，圈出它所落的地支。")[style]
    else:
        relation = {"六合": "相合（六合）", "害": "相害", "破": "相破"}[source]
        setup = (f"先找和本命{base}{relation}的那个地支。",
                 f"从原局{base}出发，另取与之{relation}的一支。")[style]
    return head_line(chart, None) + setup + OUTPUT_REQUEST[op][style] + f"\n结论JSON 需含 answer（{HINTS[op]}）。"


def create():
    if PANEL.exists() or MANIFEST.exists():
        raise FileExistsError("Transfer panel or lock already exists. Run verify to audit it.")
    rng = random.Random(SEED)
    years = sorted(split_years()["test"])
    edges = pool("reserve")
    assert len(edges) == 12 and sum(op in HARD for _, op in edges) == 4
    assert all(set(edges).isdisjoint(pool(s)) for s in ("train", "diagnostic", "dev"))
    used = all_old_keys()
    old_count = len(used)
    # The sexagenary cycle can repeat a complete chart across disjoint Gregorian years.
    # Exclude realized intervention charts before taking the first draw.
    live_paths = [DATA / "train_chart_keys.json",
                  *(DATA / arm / split for arm in ("a", "b") for split in ("train.jsonl", "valid.jsonl"))]
    used.update(json.loads(live_paths[0].read_text()))
    for path in live_paths[1:]:
        for row in rows(path):
            used.update(keys_in(row))
    live_hashes = {str(path.relative_to(ROOT)): digest(path) for path in live_paths}
    items = []
    for i, (source, op) in enumerate(edges):
        # Both position and writing style balanced independently within each family.
        layout = [(p, style) for p in POSITION for style in range(2)] * (PER_FAMILY // 4)
        rng.shuffle(layout)
        for j, (pos, style) in enumerate(layout):
            year, chart = new_chart(rng, years, used)
            group = f"transfer-{i:02}-{j:02}"
            conditions = ["novel", "familiar"] + (["switch", "consumer"] if op in HARD else [])
            for cond in conditions:
                which_pos = ("year" if pos == "day" else "day") if cond == "switch" else pos
                item = case(chart, source, op, which_pos, style)
                if cond != "familiar":
                    given = item["branch"] if cond == "consumer" else None
                    item["messages"][1]["content"] = question(chart, source, op, which_pos, style, given)
                items.append(dict(item, id=f"{group}-{cond}", group=group, family=f"{source}→{op}",
                                  sampled_year=year, condition=cond, writer_form=style))
    write_rows(PANEL, items)
    manifest = {
        "sha256": digest(PANEL), "panel": PANEL.name, "seed": SEED,
        "generator_sha256": digest(Path(__file__)), "sampled_years": years, "historical_chart_keys": old_count,
        "phase8_exclusions_sha256": live_hashes,
        "preflight": {"provisional_sha256": "6e328d27af822a49b844fcf5f1d4723679b1768a31db7ea82727078efcb85d1c",
                      "provisional_collision_count": 1, "collision_four_pillars": "乙亥己卯丙戌戊戌",
                      "resolution": "Before any model inference, regenerated with the original seed and excluded realized Phase 8 training and validation charts, then generated retention with these transfer charts excluded. No outcomes consulted."},
        "edges": edges, "groups_per_edge": PER_FAMILY, "conditions": dict(Counter(r["condition"] for r in items)),
        "design": "Simple random charts without answer or disagreement filtering. Within each edge, source position and evaluator wording are crossed and balanced. All 12 previously reserved directed edges are present. Novel and familiar have identical gold. Switch changes year/day source on 4 day-stem-dependent edges. Consumer gives the correct branch and omits the source lookup.",
        "exclusions": "Unique four-pillar tuples across groups. Exclude project Phase 2–6 historical input keys as in Phase 7 plus Phase 7 chart metadata and prompts for all panels and splits. Use project test-year blocks. Gold cross-checked with project rule tables and lunar-python-backed independent lookup.",
        "primary": "Mean strict accuracy across the twelve edges on condition novel. Each contributes 20 charts.",
        "scope": "Independent evaluation-agent wording, two hand-authored phrasings per operator. Both use the project's chart header and answer format. All data are synthetic. Reserve families are consumed by this experiment.",
    }
    with MANIFEST.open("x") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def verify(training_paths, chart_keys):
    manifest = json.loads(MANIFEST.read_text())
    assert digest(PANEL) == manifest["sha256"] and digest(Path(__file__)) == manifest["generator_sha256"]
    for name, hash_value in manifest["phase8_exclusions_sha256"].items():
        assert digest(ROOT / name) == hash_value, name
    obs = list(rows(PANEL))
    assert len(obs) == len({r["id"] for r in obs}) == 640
    assert len(obs) == len({r["messages"][1]["content"] for r in obs})
    assert Counter(r["condition"] for r in obs) == {"novel": 240, "familiar": 240, "switch": 80, "consumer": 80}
    groups = defaultdict(dict)
    keys = set()
    old = all_old_keys()
    for r in obs:
        assert r["sampled_year"] in split_years()["test"]
        assert (r["source"], r["op"]) in pool("reserve")
        key = chart_key(r["chart"])
        assert key not in old
        keys.add(key)
        assert independent(r["source"], r["op"], r["chart"], r["pos"]) == (r["branch"], r["answer"])
        assert r["branch"] == derived(r["source"], r["chart"], r["pos"])
        assert r["answer"] == calc(r["op"], r["chart"], r["branch"])
        groups[r["group"]][r["condition"]] = r
    assert len(groups) == len(keys) == 240
    for g in groups.values():
        assert g["novel"]["answer"] == g["familiar"]["answer"]
        assert g["novel"]["pos"] == g["familiar"]["pos"]
        if "switch" in g:
            assert g["consumer"]["answer"] == g["novel"]["answer"]
            assert g["switch"]["pos"] != g["novel"]["pos"]
        assert len({chart_key(r["chart"]) for r in g.values()}) == 1
    for path in training_paths:
        for r in rows(path):
            assert keys.isdisjoint(keys_in(r)), f"overlapping chart in {path}"
            if r.get("source") and r.get("op"):
                assert (r["source"], r["op"]) not in pool("reserve"), f"reserved edge in {path}"
    if chart_keys:
        train_keys = set(json.loads(chart_keys.read_text()))
        assert keys.isdisjoint(train_keys), "overlap with Phase 8 explicit training chart keys"
    print(f"Verified {len(obs)} prompts and {len(keys)} chart groups. SHA-256 {digest(PANEL)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--train", nargs="+", type=Path, default=[])
    parser.add_argument("--chart-keys", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        create()
    verify(args.train, args.chart_keys)
