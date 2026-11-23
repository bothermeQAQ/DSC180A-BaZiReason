"""Bounded secondary prompt sensitivity panel; it has no role in the primary pilot gate."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from graphs import (R,D,SOURCES,BRANCHES,PILLARS,calc,case,chart,compact,correct,derived,
                    historical_keys,parse,pool,rows,write_rows)
from build_panel import years,OUT as DATA
from bazireason.ladder import raw_sheet

def rule(src,op):
    if src in D.TRIPLE_STARS:
        source="按题目指定的年支或日支所属三合组查目标支："+"、".join(k+"取"+v for k,v in D.TRIPLE_STARS[src].items())
    else:
        source=f"以指定本命地支为基准，逐对查{src}表："+ "、".join(b+D.branch_card(b)[src] for b in BRANCHES[::1])
    if op in ("main","hidden","god","number"):
        suffix="所得支查藏干表（本气在先）："+"、".join(b+"藏"+"".join(h) for b,h in R.HIDDEN_STEMS.items())
        if op=="god":suffix+="。以日干为我，取本气藏干的十神"
    elif op=="stage":
        suffix="所得地支为目标，以日干长生支为起点，阳干顺数阴干逆数。十二宫顺序为"+"、".join(R.LIFE_STAGES)
    else:
        suffix="地支五行："+"、".join(b+R.BRANCH_ELEMENT[b] for b in BRANCHES)
        if op in ("positions","count"):suffix+="。用所得支的五行与四个原局地支逐柱比对"
    return "[规则说明]"+source+"。"+suffix+"。\n"

def build():
    rng=random.Random(710239)
    ex=historical_keys()
    for name in ("diagnostic","dev","mix","retention"):
        for r in rows(DATA/(name+".jsonl")):
            p=r.get("chart") or r.get("item",{}).get("meta",{}).get("pillars")
            if p:ex.add("".join(p[k] for k in PILLARS))
    out=[]
    for i in range(56):
        src,op=pool("dev")[i%7]
        while True:
            p=chart(rng,years("dev"),ex)
            it=case(p,src,op,rng.choice(["year","day"]),2)
            possible=[calc(op,p,b) for b in BRANCHES if calc(op,p,b)!=it["answer"]]
            if possible:break
        wrong=rng.choice(possible)
        headers=dict(L0="",L1=raw_sheet(p,None,"combo:p7"),L2=rule(src,op),
                     L3="[规则代码已核验的结论] "+compact({"answer":it["answer"]})+"\n",
                     L3w="[规则代码已核验的结论] "+compact({"answer":wrong})+"\n")
        for lv,prefix in headers.items():
            user=it["messages"][-1]["content"]
            out.append(dict(it,id=f"ctx-{i}-{lv}",group=f"ctx-{i}",condition=lv,wrong=wrong,
                            messages=[it["messages"][0],dict(role="user",content=prefix+user)]))
    write_rows(DATA/"context.jsonl",out)
    p=DATA/"context.jsonl"
    print(hashlib.sha256(p.read_bytes()).hexdigest())

def evaluate(m):
    from evaluate import MLXChatModel,BASE,U8,END
    if time.time()+1800>END-3600:raise TimeoutError("report reserve")
    ad=U8 if m=="u8" else "artifacts/adapters/bazireason-phase7-"+m
    model=MLXChatModel(BASE,ad,m)
    src=DATA/"context.jsonl"; out=Path(f"results/phase7/{m}_context.jsonl")
    inp=hashlib.sha256(src.read_bytes()).hexdigest()
    adhash=hashlib.sha256((Path(ad)/"adapters.safetensors").read_bytes()).hexdigest()
    done={r["id"] for r in rows(out)} if out.exists() else set()
    batch=sorted([r for r in rows(src) if r["id"] not in done],key=lambda r:len(r["messages"][1]["content"]))
    for start in range(0,len(batch),24):
        chunk=batch[start:start+24]
        txt,stats=model.generate_batch([r["messages"] for r in chunk],max_tokens=1100,batch=24,token_budget=60000)
        with out.open("a") as f:
            for r,t in zip(chunk,txt,strict=True):
                f.write(json.dumps(dict(id=r["id"],group=r["group"],condition=r["condition"],raw=t,
                                        correct=correct(t,r["answer"]),adopts_wrong=correct(t,r["wrong"]),
                                        adapter_hash=adhash,input_hash=inp),ensure_ascii=False)+"\n")
        print(m,start+len(chunk),len(batch),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["build","u8","r","g","t"])
    a=p.parse_args().action
    if a=="build":build()
    else:evaluate(a)
