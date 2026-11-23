"""Rescore the unaided bridge and fresh preservation panels; maintain chart/event pairing."""
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
import json
from pathlib import Path

spec = spec_from_file_location("archived_analysis", Path("scripts/phase8/analyze.py"))
archive = module_from_spec(spec)
spec.loader.exec_module(archive)
OUT = Path("results/phase9")
MODELS = ("p","x","s","g","a")
U8_HASH = "bf4007c5fce9202c7bce7791fc8ad019fee7b266821d50b81408cc74ba8475a8"


def adapter_hash(m):
    if m == "u8":
        path = Path("artifacts/adapters/bazireason-qwen3-4b-lora-p6-u8/adapters.safetensors")
        expected = U8_HASH
    else:
        phase = {"g":"phase7","a":"phase8"}.get(m,"phase9")
        summary = json.loads(Path(f"results/{phase}/training/{m}/training_summary.json").read_text())
        path = Path(summary["selected_adapter"])
        expected = summary["selected_adapter_sha256"]
    assert archive.digest(path) == expected
    return expected


def verified(panel, models):
    path = Path(f"data/phase9/{panel}.jsonl")
    lock = json.loads((OUT/"evaluation_lock.json").read_text())
    inp = archive.digest(path)
    assert inp == lock[str(path)]
    items = archive.indexed(path)
    hashes = {}
    for m in models:
        p = OUT/f"{m}_{panel}.jsonl"
        data = archive.indexed(p)
        hashes[m] = archive.digest(p)
        assert items.keys() == data.keys()
        adapter = adapter_hash(m)
        for key, r in items.items():
            reply = data[key]
            assert (reply["input_hash"], reply["adapter_hash"], reply["condition"], reply["group"]) == (
                inp, adapter, r["condition"], r["group"])
            if panel == "bridge":
                ok = archive.correct(reply["raw"],r["answer"])
            else:
                scorer = archive.score_row if r["track"] == "calendar" else archive.ladder.score
                score = scorer(r["item"],reply["raw"])
                ok = bool(score["correct"])
                if r["track"] == "calendar":
                    for field in ("false_refusal","definitive_on_nonanswer"):
                        assert score[field] == reply["score"][field]
            assert reply["correct"] is ok
            r.setdefault("outcome",{})[m] = ok
            r.setdefault("responses",{})[m] = reply
    return list(items.values()),hashes


def main():
    bridge, hb = verified("bridge", MODELS)
    ret, hr = verified("retention", (*MODELS,"u8"))
    result = dict(bridge=archive.endpoint(bridge,MODELS),bridge_by_family=archive.family_table(bridge,MODELS),
                  retention=archive.retention(ret,(*MODELS,"u8")), replies_sha256=dict(bridge=hb,retention=hr))
    (OUT/"preservation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
