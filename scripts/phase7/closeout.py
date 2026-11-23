"""Verify the archived pilot replies and summarize secondary context and trace checks."""
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re

from bazireason import ladder
from bazireason.p2_eval import score_row
from graphs import correct, rows
from inspect_diagnostic import semantic

DATA = Path("data/phase7")
OUT = Path("results/phase7")
MODELS = ("u8", "r", "g", "t")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify():
    audit = read(OUT / "input_audit.json")
    for name, expected in audit["sha256"].items():
        assert digest(Path(name)) == expected, name
    assert digest(DATA / "context.jsonl") == read(DATA / "context_manifest.json")["sha256"]
    sizes = {**audit["panels"], "context": 280}
    result = {"rows": {}, "context": {}, "trace_on_wrong_primary": {}, "format_recoveries_on_primary": {}}
    last_reply = 0
    training = read(DATA / "training_manifest.json")
    for model in MODELS:
        weights = Path("artifacts/adapters/bazireason-qwen3-4b-lora-p6-u8/adapters.safetensors")
        if model != "u8":
            summary = read(OUT / "training" / model / "training_summary.json")
            assert digest(weights) == summary["init_adapter_sha256"]
            weights = Path(f"artifacts/adapters/bazireason-phase7-{model}/adapters.safetensors")
            assert digest(weights) == summary["selected_adapter_sha256"]
            for key, path in [("config", Path(f"configs/phase7/{model}.yaml")),
                              ("train", DATA / model / "train.jsonl"), ("valid", DATA / model / "valid.jsonl")]:
                assert digest(path) == training["hashes"][model][key]
        adapter_hash = digest(weights)
        context = {c: Counter() for c in ("L0", "L1", "L2", "L3", "L3w")}
        trace = Counter()
        recovered = 0
        for panel, size in sizes.items():
            if panel == "diagnostic" and model != "u8":
                continue
            source = DATA / f"{panel}.jsonl"
            inputs = {r["id"]: r for r in rows(source)}
            path = OUT / f"{model}_{panel}.jsonl"
            last_reply = max(last_reply, path.stat().st_mtime)
            obs = list(rows(path))
            assert len(inputs) == len(obs) == size
            assert set(inputs) == {r["id"] for r in obs}
            input_hash = digest(source)
            for r in obs:
                case = inputs[r["id"]]
                assert (r["input_hash"], r["adapter_hash"], r["condition"], r["group"]) == (
                    input_hash, adapter_hash, case["condition"], case["group"])
                if panel == "retention":
                    scorer = score_row if case["track"] == "calendar" else ladder.score
                    ok = bool(scorer(case["item"], r["raw"])["correct"])
                else:
                    ok = correct(r["raw"], case["answer"])
                assert r["correct"] == ok, r["id"]
                if panel == "context":
                    adopted = correct(r["raw"], case["wrong"])
                    assert r["adopts_wrong"] == adopted
                    context[r["condition"]].update(n=1, right=int(ok), wrong_hint=int(adopted))
                if panel == "dev" and r["condition"] == "full" and not ok:
                    recovered += semantic(r["raw"], case["answer"])
                    if model in ("g", "t"):
                        pattern = "可得([子丑寅卯辰巳午未申酉戌亥])" if model == "g" else "所得地支=([子丑寅卯辰巳午未申酉戌亥])"
                        match = re.search(pattern, r["raw"].splitlines()[0])
                        trace.update(misses=1, readable=int(bool(match)),
                                     correct_first_branch=int(bool(match) and match[1] == case["branch"]))
            result["rows"][f"{model}_{panel}"] = size
        assert all(c["n"] == 56 for c in context.values())
        result["context"][model] = context
        result["format_recoveries_on_primary"][model] = recovered
        if trace:
            result["trace_on_wrong_primary"][model] = trace
    result["replies_verified"] = sum(result["rows"].values())
    result["last_reply_time"] = datetime.fromtimestamp(last_reply).astimezone().isoformat()
    return result


if __name__ == "__main__":
    (OUT / "closeout.json").write_text(json.dumps(verify(), ensure_ascii=False, indent=2) + "\n")
