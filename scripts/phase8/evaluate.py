"""Resume only model/input-hash-matched observations and use the historical scorers."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from bazireason import ladder
from bazireason.mlx_backend import MLXChatModel
from bazireason.p2_eval import score_row
from graphs import correct, rows

OUT = Path("results/phase8")
STOP = datetime(2026,11,22,10,17,tzinfo=ZoneInfo("America/Los_Angeles"))
ADAPTERS = {"u8":"bazireason-qwen3-4b-lora-p6-u8", "g":"bazireason-phase7-g",
            "t":"bazireason-phase7-t","a":"bazireason-phase8-a","b":"bazireason-phase8-b"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def evaluate(arm, panels):
    adapter = Path("artifacts/adapters") / ADAPTERS[arm]
    adapter_hash = digest(adapter / "adapters.safetensors")
    if arm == "u8":
        expected = "bf4007c5fce9202c7bce7791fc8ad019fee7b266821d50b81408cc74ba8475a8"
    else:
        study = "phase7" if arm in ("g","t") else "phase8"
        summary = json.loads(Path(f"results/{study}/training/{arm}/training_summary.json").read_text())
        expected = summary["selected_adapter_sha256"]
    assert adapter_hash == expected
    model = MLXChatModel("models/qwen3-4b-instruct-2507-mlx-4bit", adapter, arm)
    for panel in panels:
        source = Path("data/phase7" if panel == "dev" else "data/phase8") / f"{panel}.jsonl"
        input_hash = digest(source)
        path = OUT / f"{arm}_{panel}.jsonl"
        previous = list(rows(path)) if path.exists() else []
        assert all(r["input_hash"] == input_hash and r["adapter_hash"] == adapter_hash for r in previous)
        done = {r["id"] for r in previous}
        assert len(done) == len(previous)
        inputs = list(rows(source))
        assert done <= {r["id"] for r in inputs}
        todo = [r for r in inputs if r["id"] not in done]
        for cap in (1100,700):
            selected = sorted((r for r in todo if r.get("cap",1100) == cap),
                              key=lambda r: len(r["messages"][-1]["content"]))
            for start in range(0,len(selected),24):
                if datetime.now(STOP.tzinfo) >= STOP:
                    raise TimeoutError("Phase 8 model-work deadline")
                batch = selected[start:start+24]
                texts, stats = model.generate_batch([r["messages"] for r in batch], max_tokens=cap,
                                                    batch=24, temperature=0, token_budget=60000)
                with path.open("a") as out:
                    for r, raw in zip(batch, texts, strict=True):
                        score = {}
                        if panel == "retention":
                            scorer = score_row if r["track"] == "calendar" else ladder.score
                            score = scorer(r["item"],raw)
                            ok = bool(score["correct"])
                        else:
                            ok = correct(raw, r["answer"])
                        result = dict(id=r["id"], group=r["group"], condition=r["condition"], raw=raw,
                                      score=score, correct=ok, adapter_hash=adapter_hash, input_hash=input_hash)
                        out.write(json.dumps(result,ensure_ascii=False)+"\n")
                print(arm,panel,cap,start+len(batch),len(selected),round(stats["seconds"],1),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ADAPTERS)
    parser.add_argument("panels", nargs="+", choices=("transfer","retention","dev"))
    args = parser.parse_args()
    evaluate(args.arm, args.panels)
