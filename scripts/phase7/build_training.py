"""Make token-matched, one-epoch Phase 7 arms. No model inference or test outcomes."""
import hashlib
import json
import math
import random
from collections import Counter,defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from bazireason.finetune import PromptCompletionDataset
from graphs import case, chart, historical_keys, pool, rows, states, write_rows, compact
from build_panel import years

ROOT=Path("data/phase7")
TARGET=3_000_000
SEED=710223
MODEL="models/qwen3-4b-instruct-2507-mlx-4bit"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def category(r):
    if r["task"]=="chart":return "calendar"
    if r["family"]=="procedure":return "procedure"
    if r["family"]=="contrast":return "contrast"
    return "other"

def main():
    rng=random.Random(SEED)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ds=PromptCompletionDataset([],tokenizer)
    def size(r):
        seq,off=ds.process(r)
        return len(seq),len(seq)-off
    def pack(r):
        n,loss=size(r)
        assert n<=2048
        return r,n,loss
    excluded=historical_keys()
    for name in ("diagnostic","dev","mix","retention"):
        for r in rows(ROOT/(name+".jsonl")):
            p=r.get("chart") or r.get("item",{}).get("meta",{}).get("pillars")
            if p:excluded.add("".join(p[k] for k in ("year","month","day","hour")))
    used=set()
    buckets=defaultdict(list)
    for k in ["u8","u7","u6","u4","u3v","u2"]:
        for row in rows(Path(f"data/phase6/train/{k}/train.jsonl")):
            digest=hashlib.sha256(compact(row["messages"]).encode()).digest()
            if digest in used:continue
            used.add(digest)
            buckets[category(row)].append(row)
    for rs in buckets.values():rng.shuffle(rs)
    common=[]
    by_share={}
    for typ,pct in dict(calendar=.30,procedure=.38,contrast=.08,other=.04).items():
        candidates=buckets[typ]
        if typ=="calendar":
            # Round robin across original request family/status, removing strata as exhausted.
            bins=defaultdict(list)
            for row in candidates:
                bins[(row["family"],row["meta"]["expected"]["status"])].append(row)
            candidates=[]
            while any(bins.values()):
                candidates.extend(b.pop() for b in bins.values() if b)
            candidates.reverse()
        n=0
        while n<pct*TARGET:
            rr=pack(candidates.pop())
            common.append(rr);n+=rr[1]
        by_share[typ]=n
    graph_g=[];graph_t=[]
    # Use a paired source switch where available, plus ordinary random charts to retain natural cases.
    i=0;tokens=0
    train_edges=pool("train")
    while tokens<.2*TARGET or (len(common)+len(graph_g))%4:
        src,op=train_edges[i%len(train_edges)]
        p=chart(rng,years("train"),excluded)
        pos=rng.choice(["year","day"])
        for where in (pos, "day" if pos=="year" else "year"):
            it=case(p,src,op,where,rng.randrange(2))
            prose,structured=states(it)
            ident=f"p7-train-{i}-{where}"
            def target(body):
                return dict(id=ident,family="p7_graph",task="p7_graph",source=src,op=op,
                            messages=it["messages"]+[dict(role="assistant",content=body)])
            g,t=pack(target(prose)),pack(target(structured))
            graph_g.append(g);graph_t.append(t)
            tokens+=(g[1]+t[1])/2
            if tokens>=.2*TARGET and (len(common)+len(graph_g))%4==0:break
        i+=1
    print("common",by_share,"rows",len(common),"graph rows",len(graph_g),flush=True)
    # Stratify candidate R lengths. Minimize discrepancy in both completion and full sequence budgets.
    bins=defaultdict(list)
    for row in buckets["procedure"]:
        rr=pack(row)
        bins[(rr[1]//8,rr[2]//8)].append(rr)
        if sum(len(v) for v in bins.values())>=18000:break
    ctrl=[];dseq=dloss=0
    for g,t in zip(graph_g,graph_t):
        n=(g[1]+t[1])/2
        loss=(g[2]+t[2])/2
        target_n=(n+min(50,max(-50,dseq)))/8
        target_loss=(loss+min(50,max(-50,dloss)))/8
        b=min(bins,key=lambda xy:(xy[0]+.5-target_n)**2+(xy[1]+.5-target_loss)**2)
        rr=bins[b].pop()
        if not bins[b]:del bins[b]
        ctrl.append(rr);dseq+=n-rr[1];dloss+=loss-rr[2]
    arms={arm:common+tail for arm,tail in [("r",ctrl),("g",graph_g),("t",graph_t)]}
    shared_lengths=[round(sum(a[1] for a in triplet)/3) for triplet in zip(*arms.values(),strict=True)]
    order=list(range(len(arms["r"])));rng.shuffle(order)
    batch_order=sorted(order,key=shared_lengths.__getitem__)
    def stats(group):
        lens=[group[i][1] for i in batch_order]
        padded=sum(2*min(2048,1+32*math.ceil(max(lens[i:i+2])/32)) for i in range(0,len(lens),2))
        return dict(rows=len(group),updates=len(group)//4,sequence=sum(lens),loss=sum(x[2] for x in group),
                    padded=padded,max_length=max(lens))
    summary={k:stats(rs) for k,rs in arms.items()}
    print(summary,flush=True)
    for key in ("updates","sequence","loss","padded"):
        vs=[s[key] for s in summary.values()]
        assert max(vs)/min(vs)<=1.03,(key,vs)
    # Validation loss is reported only. Save/eval at the fixed final iteration exclusively.
    valid=list(rows("data/phase6/train/u8/valid.jsonl"))[:64]
    basecfg=yaml.safe_load(Path("configs/phase6/qwen3_4b_lora_p6_u8.yaml").read_text())
    files={}
    for arm,records in arms.items():
        folder=ROOT/arm;folder.mkdir(exist_ok=False)
        write_rows(folder/"train.jsonl",(dict(records[i][0],phase7_sort_length=shared_lengths[i]) for i in order))
        write_rows(folder/"valid.jsonl",valid)
        cfg=dict(basecfg,run_name="bazireason-phase7-"+arm,seed=SEED,
                 data=dict(train=str(folder/"train.jsonl"),valid=str(folder/"valid.jsonl"),valid_rows=len(valid)),
                 init_adapter=basecfg["adapter_dir"]+"/adapters.safetensors",
                 log_dir=f"results/phase7/training/{arm}",adapter_dir="artifacts/adapters/bazireason-phase7-"+arm,
                 steps_per_save=len(records)//2,steps_per_eval=len(records)//2)
        cp=Path(f"configs/phase7/{arm}.yaml");cp.parent.mkdir(exist_ok=True)
        with cp.open("x") as f:yaml.safe_dump(cfg,f,allow_unicode=True,sort_keys=False)
        files[arm]=dict(train=sha(folder/"train.jsonl"),config=sha(cp),valid=sha(folder/"valid.jsonl"))
    (ROOT/"training_manifest.json").write_text(json.dumps(dict(seed=SEED,common_tokens=by_share,
            common_rows=len(common),graph_rows=len(graph_g),stats=summary,hashes=files),indent=2))
if __name__=="__main__":main()
