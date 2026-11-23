"""Greedy, resumable Phase 7 evaluation, storing raw outputs and prompt provenance."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from graphs import correct, rows
from bazireason.mlx_backend import MLXChatModel
from bazireason import ladder
from bazireason.p2_eval import score_row

END=1795378740  # replaced from the explicit Pacific deadline below
from datetime import datetime
from zoneinfo import ZoneInfo
END=datetime(2026,11,22,12,19,tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
ROOT=Path("results/phase7")
DATA=Path("data/phase7")
BASE="models/qwen3-4b-instruct-2507-mlx-4bit"
U8="artifacts/adapters/bazireason-qwen3-4b-lora-p6-u8"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("model",choices=["u8","r","g","t"])
    ap.add_argument("panels",nargs="+",choices=["diagnostic","dev","mix","retention"])
    a=ap.parse_args()
    adapter=Path(U8 if a.model=="u8" else f"artifacts/adapters/bazireason-phase7-{a.model}")
    weights=sha(adapter/"adapters.safetensors")
    if a.model=="u8":
        assert weights=="bf4007c5fce9202c7bce7791fc8ad019fee7b266821d50b81408cc74ba8475a8"
    model=MLXChatModel(BASE,str(adapter),a.model)
    for panel in a.panels:
        src=DATA/(panel+".jsonl")
        identity=sha(src)
        out=ROOT/f"{a.model}_{panel}.jsonl"
        old={r["id"]:r for r in rows(out)} if out.exists() else {}
        assert all(r["input_hash"]==identity and r["adapter_hash"]==weights for r in old.values())
        candidates=[r for r in rows(src) if r["id"] not in old]
        for cap in [1100,700]:
            selected=[r for r in candidates if r.get("cap",1100)==cap]
            selected.sort(key=lambda r:len(r["messages"][-1]["content"]))
            for start in range(0,len(selected),24):
                if time.time()>END-300:
                    raise RuntimeError("24-hour deadline")
                batch=selected[start:start+24]
                texts,stats=model.generate_batch([r["messages"] for r in batch],max_tokens=cap,
                                                batch=24,temperature=0,token_budget=60000)
                with out.open("a") as f:
                    for r,raw in zip(batch,texts,strict=True):
                        if panel=="retention":
                            sc=score_row(r["item"],raw) if r["track"]=="calendar" else ladder.score(r["item"],raw)
                            ok=bool(sc["correct"])
                        else:
                            ok=correct(raw,r["answer"])
                            sc={}
                        f.write(json.dumps(dict(id=r["id"],group=r["group"],condition=r["condition"],correct=ok,
                                               raw=raw,score=sc,input_hash=identity,adapter_hash=weights),ensure_ascii=False)+"\n")
                print(a.model,panel,cap,start+len(batch),len(selected),round(stats["seconds"],1),flush=True)
if __name__=="__main__":
    main()
