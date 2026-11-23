"""Paired pilot estimates conditional on the sampled chart groups, with no checkpoint selection."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from graphs import calc, compact, parse, rows
from bazireason.p2_prompts import extract_final_json

DATA=Path("data/phase7")
OUT=Path("results/phase7")
MODELS=("u8","r","g","t")
def load(panel,model):
    obs={r["id"]:r for r in rows(OUT/f"{model}_{panel}.jsonl")}
    items=list(rows(DATA/(panel+".jsonl")))
    assert len(obs)==len(items)==len({r["id"] for r in items})
    return [dict(r,observation=obs[r["id"]]) for r in items]

def diagnostic():
    data=load("diagnostic","u8")
    groups=defaultdict(dict)
    for r in data:groups[r["group"]][r["condition"]]=r
    by_family=defaultdict(Counter)
    counts=Counter()
    examples=[]
    for gid,records in groups.items():
        x=records["full"]
        obs={k:v["observation"]["correct"] for k,v in records.items()}
        metrics={"groups":1,**obs,"both_primitives":obs["producer"] and obs["consumer"],
                 "both_twins":obs["full"] and obs["twin"],"miss":not obs["full"],
                 "miss_bad_pre":not obs["full"] and not (obs["producer"] and obs["consumer"]),
                 "oracle_rescue":not obs["full"] and obs["oracle"],"oracle_break":obs["full"] and not obs["oracle"],
                 "unparsed":parse(x["observation"]["raw"]) is None}
        natal=calc(x["op"],x["chart"],x["chart"][x["pos"]][1])
        metrics["wrong_equals_natal"]=not obs["full"] and parse(x["observation"]["raw"])==natal
        for k,v in metrics.items():
            by_family[x["source"]+"→"+x["op"]][k]+=int(v)
            counts[k]+=int(v)
        if not obs["full"]:
            examples.append(dict(id=x["id"],family=x["source"]+"→"+x["op"],branch=x["branch"],
                                 natal=natal,gold=x["answer"],got=parse(x["observation"]["raw"]),obs=obs,
                                 raw=x["observation"]["raw"]))
    result=dict(counts=dict(counts),by_family=by_family,prerequisite_flag=counts["miss_bad_pre"]>counts["miss"]/2,
                examples=examples)
    (OUT/"diagnostic_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=1))
    print(json.dumps({k:result[k] for k in ("counts","by_family","prerequisite_flag")},ensure_ascii=False))
    return result

def paired_boot(groups,families):
    rng=np.random.default_rng(710234)
    per=[];vals=[]
    for fam in sorted(set(families.values())):
        v=np.array([np.mean(scores) for g,scores in groups.items() if families[g]==fam])
        vals.append(v.mean())
        per.append(v[rng.integers(0,len(v),(4000,len(v)))].mean(axis=1))
    draws=np.array(per)
    fixed=draws.mean(axis=0)
    j=rng.integers(0,len(per),(len(per),4000))
    broader=draws[j,np.arange(4000)].mean(axis=0)
    lofo=[np.mean([x for j,x in enumerate(vals) if j!=i]) for i in range(len(vals))] if len(vals)>1 else vals
    return dict(pp=100*np.mean(vals),fixed_ci=(100*np.quantile(fixed,[.025,.975])).tolist(),
                family_ci=(100*np.quantile(broader,[.025,.975])).tolist(),
                lofo=[100*min(lofo),100*max(lofo)],groups=len(groups))

def compare(part,by_model,filter_fn,aggregate_family=True):
    ls={k:[r for r in rs if filter_fn(r)] for k,rs in by_model.items()}
    totals={}
    for k,rs in ls.items():
        totals[k]=[sum(r["observation"]["correct"] for r in rs),len(rs)]
    paired={}
    for a,b in [("r","t"),("g","t"),("r","g"),("u8","r"),("u8","g"),("u8","t")]:
        groups=defaultdict(list);families={}
        for x,y in zip(ls[a],ls[b],strict=True):
            assert x["id"]==y["id"]
            key=x["group"]
            groups[key].append(int(y["observation"]["correct"])-int(x["observation"]["correct"]))
            families[key]=x["source"]+"→"+x["op"] if aggregate_family else "all"
        paired[b+"-"+a]=paired_boot(groups,families)
    return dict(counts=totals,paired=paired)

def final():
    db={p:{m:load(p,m) for m in MODELS} for p in ("dev","mix","retention")}
    out={}
    for s in ("full","twin","alternate"):
        out[s]=compare(s,db["dev"],lambda r:r["condition"]==s)
    for s in ("unseen0","unseen2","seen0","seen2"):
        out[s]=compare(s,db["mix"],lambda r:r["condition"]==s,False)
    for s in ("rules","calendar"):
        out[s]=compare(s,db["retention"],lambda r:r["track"]==s,False)
    # Preserve the ordinary row-weighted retention estimand while resampling whole event groups.
    for s in ("calendar","rules"):
        ls={m:[r for r in v if r["track"]==s] for m,v in db["retention"].items()}
        for key,v in out[s]["paired"].items():
            b,a=key.split("-")
            gg=defaultdict(list)
            for x,y in zip(ls[a],ls[b],strict=True):
                gg[x["group"]].append(int(y["observation"]["correct"])-int(x["observation"]["correct"]))
            sz=np.array([len(z) for z in gg.values()]);tot=np.array([sum(z) for z in gg.values()])
            pick=np.random.default_rng(710235).integers(len(sz),size=(4000,len(sz)))
            sim=100*tot[pick].sum(axis=1)/sz[pick].sum(axis=1)
            v.update(pp=100*tot.sum()/sz.sum(),row_ci=np.quantile(sim,[.025,.975]).tolist())
    family={};negatives={};subgroups={}
    for m,rs in db["dev"].items():
        full={r["group"]:r for r in rs if r["condition"]=="full"}
        twin={r["group"]:r for r in rs if r["condition"]=="twin"}
        cs=defaultdict(Counter)
        for k,r in full.items():
            ob=r["observation"];t=twin[k]["observation"];fam=r["source"]+"→"+r["op"]
            got=parse(ob["raw"]); nat=calc(r["op"],r["chart"],r["chart"][r["pos"]][1])
            cs[fam].update(n=1,right=int(ob["correct"]),both=int(ob["correct"] and t["correct"]),
                           no_answer=int(got is None),wrong_natal=int(not ob["correct"] and got==nat))
        family[m]=cs
        ret=[r for r in db["retention"][m] if r["track"]=="calendar"]
        negatives[m]=dict(refusal=sum(r["observation"]["score"]["false_refusal"] for r in ret),
                          unwarranted=sum(r["observation"]["score"]["definitive_on_nonanswer"] for r in ret),
                          answerable=sum(r["item"]["meta"]["expected"]["status"]=="ok" for r in ret))
        subgroups[m]={}
        for r in db["retention"][m]:
            key=r["track"]+"."+r["condition"]
            cell=subgroups[m].setdefault(key,[0,0])
            cell[0]+=int(r["observation"]["correct"]);cell[1]+=1
    # Fit modal labels from training graphs pooled by consumer (held-out edges have no training rows).
    priors=defaultdict(Counter)
    for r in rows(DATA/"g/train.jsonl"):
        if r["family"]=="p7_graph":
            priors[r["op"]][compact(parse(r["messages"][-1]["content"]))]+=1
    modes={op:json.loads(hist.most_common(1)[0][0]) for op,hist in priors.items()}
    simple={}
    for p,ls in [("dev",[r for r in db["dev"]["u8"] if r["condition"]=="full"]),
                 ("mix_unseen",[r for r in db["mix"]["u8"] if r["condition"]=="unseen0"])]:
        simple[p]=dict(n=len(ls),mode=sum(modes[r["op"]]==r["answer"] for r in ls),
                       natal=sum(calc(r["op"],r["chart"],r["chart"][r["pos"]][1])==r["answer"] for r in ls))
    result=dict(endpoints=out,by_family=family,calendar=negatives,retention_cells=subgroups,baselines=simple)
    (OUT/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=1))
    print(json.dumps({k:v["counts"] for k,v in out.items()}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("diagnostic","final"))
    (diagnostic if p.parse_args().stage=="diagnostic" else final)()
