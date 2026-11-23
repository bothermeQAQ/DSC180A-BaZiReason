"""Construct new chart-disjoint pilot panels without consulting any model responses."""
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from bazireason import train_data as TD
from graphs import case, calc, chart, historical_keys, pool, rows, write_rows

OUT=Path("data/phase7")
def years(part):
    # New chart identities within past evaluation year blocks, disjoint across new panels.
    if part=="train":
        return sorted(TD.split_years()["train"])
    valid=sorted(TD.split_years()["valid"])
    return valid[::2] if part=="diagnostic" else valid[1::2]

def panel(part,graphs,n,excluded,rng):
    rows_out=[]
    for i in range(n):
        src,op=graphs[i%len(graphs)]
        while True:
            p=chart(rng,years(part),excluded)
            pos1,pos2=rng.sample(["year","day"],2)
            primary=case(p,src,op,pos1,i%2)
            twin=case(p,src,op,pos2,i%2)
            if primary["answer"]!=twin["answer"]:
                break
        g=f"{part}-{i:04d}"
        conditions = [("full",primary),("twin",twin)]
        if part=="diagnostic":
            conditions.extend((v,case(p,src,op,pos1,i%2,v)) for v in ("producer","consumer","oracle","distractor","cue"))
        else:
            conditions.append(("alternate",case(p,src,op,pos1,2)))
        for v,it in conditions:
            rows_out.append(dict(it,id=g+"-"+v,group=g,condition=v))
    return rows_out

def main():
    excluded=historical_keys()
    print("Historical chart keys",len(excluded),flush=True)
    rng=random.Random(710203)
    for part,n in [("diagnostic",120),("dev",140)]:
        out=panel(part,pool(part),n,excluded,rng)
        write_rows(OUT/(part+".jsonl"),out)
    # Independent natural-mix panel: no disagreement filtering or answer balancing.
    # Seen graph / familiar wording, unseen graph / familiar wording, and both paraphrases.
    mixed=[]
    for i in range(140):
        category="seen" if i%2==0 else "unseen"
        src,op=rng.choice(pool("train" if category=="seen" else "dev"))
        p=chart(rng,years("dev"),excluded)
        pos=rng.choice(["year","day"])
        for render in [0,2]:
            it=case(p,src,op,pos,render)
            mixed.append(dict(it,id=f"mix-{i:03d}-{render}",group=f"mix-{i:03d}",condition=category+str(render)))
    write_rows(OUT/"mix.jsonl",mixed)
    manifest={"seed":710203,"graph_split":{s:pool(s) for s in ("train","diagnostic","dev","reserve")},
              "panels":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.jsonl")}}
    (OUT/"panel_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
