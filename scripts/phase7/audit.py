"""Independent preflight over frozen files; refuses release for mismatched arms or leakage."""
import codecs
import hashlib
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
from graphs import rows,pool,correct,compact,calc,derived,independent
from build_training import ROOT

def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    panels={p:list(rows(ROOT/(p+".jsonl"))) for p in ("diagnostic","dev","mix","retention")}
    for data in panels.values():
        assert len(data)==len({r["id"] for r in data})
    assert len(panels["diagnostic"])==840 and len(panels["dev"])==420
    def key(p):return "".join(p[k] for k in ("year","month","day","hour"))
    charts=defaultdict(set); prompts=defaultdict(set)
    for panel,records in panels.items():
        for r in records:
            p=r.get("chart") or r.get("item",{}).get("meta",{}).get("pillars")
            if p:charts[panel].add(key(p))
            prompts[panel].add(compact(r["messages"]))
            if "chart" in r:
                b,ans=independent(r["source"],r["op"],p,r["pos"])
                assert r["answer"]==(b if r["variant"]=="producer" else ans)
                assert r["branch"]==derived(r["source"],p,r["pos"])
        for other in charts:
            if panel!=other:assert charts[panel].isdisjoint(charts[other])
    sets=[set(pool(s)) for s in ("train","diagnostic","dev","reserve")]
    assert all(a.isdisjoint(b) for i,a in enumerate(sets) for b in sets[i+1:])
    for panel in ("diagnostic","dev"):
        groups=defaultdict(dict)
        for r in panels[panel]:groups[r["group"]][r["condition"]]=r
        assert all(z["full"]["answer"]!=z["twin"]["answer"] for z in groups.values())
    phase7charts=set.union(*charts.values())
    phase7prompts=set.union(*prompts.values())
    common=graph=0
    from graphs import R,PILLARS
    four=re.compile("[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]")
    for r,g,t in zip(*(rows(ROOT/a/"train.jsonl") for a in "rgt"),strict=True):
        assert len({x["phase7_sort_length"] for x in (r,g,t)})==1
        if r==g==t:common+=1
        else:
            assert g["messages"][:-1]==t["messages"][:-1] and g["id"]==t["id"]
            assert g["messages"][-1]["content"].splitlines()[-1]==t["messages"][-1]["content"].splitlines()[-1]
            assert (g["source"],g["op"]) in sets[0]
            for x in [g,t]:
                branches=four.findall(x["messages"][1]["content"].split("。")[0])
                assert len(branches)==4 and "".join(branches) not in phase7charts
            graph+=1
        for x in (r,g,t):
            assert compact(x["messages"][:-1]) not in phase7prompts
            p=x.get("meta",{}).get("pillars")
            if p and all(k in p for k in PILLARS):assert key(p) not in phase7charts
    for bad in ('{"answer":true}','{"answer":1,"answer":2}','{"answer":"1"}','{"answer":1,"other":1}'):
        assert not correct("结论JSON: "+bad,1)
    frozen=[Path("docs/phase7_pilot_protocol.md"),*Path("scripts/phase7").glob("*.py"),*ROOT.glob("*.jsonl"),*ROOT.glob("*manifest.json")]
    report=dict(common_rows=common,experiment_rows=graph,panels={k:len(v) for k,v in panels.items()},
              chart_panels={k:len(v) for k,v in charts.items()},directed_graphs=[len(s) for s in sets],
              sha256={str(p):h(p) for p in sorted(frozen)})
    path=Path("results/phase7/input_audit.json")
    with path.open("x") as f:json.dump(report,f,indent=2)
    print(common,graph,report["panels"])
if __name__=="__main__":main()
