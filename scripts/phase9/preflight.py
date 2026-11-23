"""Lock the realized training and evaluation inputs independently."""
from collections import Counter
from hashlib import sha256
import json
import sys
from pathlib import Path
import yaml

DATA = Path("data/phase9")
OUT = Path("results/phase9")
SCRIPTS = Path("scripts/phase9")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def freeze(name, paths):
    current = {str(p):digest(p) for p in sorted(set(paths))}
    target = OUT / f"{name}_lock.json"
    if target.exists():
        assert read(target) == current
        return
    temp = target.with_suffix(".part")
    temp.write_text(json.dumps(current,indent=2)+"\n")
    temp.replace(target)


def training():
    manifest = read(DATA / "training_manifest.json")
    paths = [Path("docs/phase9_protocol.md"),DATA/"training_manifest.json",SCRIPTS/"build_training.py",
             SCRIPTS/"train.py",SCRIPTS/"run.py",SCRIPTS/"evaluate.py",Path("scripts/phase2/train_lora.py")]
    snapshots = {}
    for m in "sxp":
        cp = Path(f"configs/phase9/{m}.yaml")
        cfg = yaml.safe_load(cp.read_text())
        hashes = manifest["hashes"][m]
        for k,p in [("config",cp),("train",Path(cfg["data"]["train"])),("valid",Path(cfg["data"]["valid"]))]:
            assert digest(p) == hashes[k]
            paths.append(p)
        assert digest(Path(cfg["init_adapter"])) == "790f87c137011c24e4cdf21dfe9dfabc7936612f0b19d23b9092cdae8e6778f9"
        snapshots[m] = rows(Path(cfg["data"]["train"]))
    s,x,p = (snapshots[m] for m in "sxp")
    assert len(s) == len(x) == len(p)
    assert all(r["messages"][:-1] == q["messages"][:-1] for r,q in zip(x,p,strict=True))
    for u,v in zip(x,p,strict=True):
        assert u["messages"][-1]["content"].splitlines()[-1] == v["messages"][-1]["content"].splitlines()[-1]
    protected = {("害","stage"),("破","god"),("驿马","positions"),("桃花","element")}
    assert all((r.get("source"),r.get("op")) not in protected for arm in snapshots.values() for r in arm)
    freeze("training",paths)


def evaluation():
    audit = read(OUT / "retention_audit.json")
    assert not audit["collisions"] and audit["panel_sha256"] == digest(DATA/"retention.jsonl")
    paths = [OUT/"training_lock.json",OUT/"retention_audit.json",SCRIPTS/"panel.py",
             SCRIPTS/"evaluate.py",SCRIPTS/"retention.py",SCRIPTS/"audit_retention.py"]
    for panel in ("transfer","bridge","retention"):
        p = DATA / f"{panel}.jsonl"
        lock = DATA / f"{panel}_manifest.json"
        case = rows(p)
        assert read(lock)["sha256"] == digest(p)
        assert len(case) == len({r["id"] for r in case})
        assert all([m["role"] for m in r["messages"]] == ["system","user"] for r in case)
        paths.extend([p,lock])
    freeze("evaluation",paths)


if __name__ == "__main__":
    {"training": training,"evaluation":evaluation}[sys.argv[1]]()
