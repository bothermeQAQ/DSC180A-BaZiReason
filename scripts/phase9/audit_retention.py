"""Check fresh retention identities against historical and actual fitted input streams."""
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import sys
from importlib.util import spec_from_file_location, module_from_spec
# Use exactly the archived identity extraction (the Phase 9 transfer generator is a separate module).
spec = spec_from_file_location("historical_panel", Path("scripts/phase8/panel.py"))
old = module_from_spec(spec)
spec.loader.exec_module(old)
keys_in, rows = old.keys_in, old.rows


def identities(raw):
    row = raw.get("row", raw)
    item = row.get("item",row)
    keys = keys_in(row)
    prompts = {m["content"] for r in [row,item] for m in r.get("messages", []) if m["role"] == "user"}
    dt = item.get("meta", {}).get("inputs", {}).get("datetime")
    dates = {dt[:10]} if isinstance(dt,str) else set()
    return dict(chart=keys, prompt=prompts, calendar_date=dates)


def input_files():
    past = list(Path("data").glob("phase[2345678]/**/*.jsonl"))
    past += list(Path("results/phase6").glob("own_replies*.jsonl"))
    past += list(Path("data/phase9").glob("[sxp]/*.jsonl"))
    past.extend([Path("data/phase9/transfer.jsonl"), Path("data/phase9/bridge.jsonl")])
    return sorted(past)


def main():
    source = Path("data/phase9/retention.jsonl")
    panel = list(rows(source))
    sets = {field:defaultdict(list) for field in ("chart","prompt","calendar_date")}
    for r in panel:
        for field, values in identities(r).items():
            for value in values:
                sets[field][value].append(r["id"])
    census = {}
    past = input_files()
    for path in past:
        hit = defaultdict(set)
        for r in rows(path):
            for field, values in identities(r).items():
                for value in values:
                    hit[field].update(sets[field].get(value,[]))
        any_hit = {k:sorted(v) for k,v in hit.items() if v}
        if any_hit:
            census[str(path)] = any_hit
    audit = dict(panel_sha256=sha256(source.read_bytes()).hexdigest(),input_files=len(past),
                 rows=len(panel),groups=len({r["group"] for r in panel}),collisions=census)
    outfile = Path("results/phase9/retention_audit.json")
    outfile.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(audit,ensure_ascii=False))
    if census:
        raise ValueError("retention identity collisions")

if __name__ == "__main__":
    main()
