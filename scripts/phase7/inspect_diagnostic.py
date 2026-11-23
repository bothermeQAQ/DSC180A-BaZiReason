"""Localize diagnostic failures from fixed prompts; never change labels or primary outcomes."""
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
from analyze import load, OUT
from graphs import parse,calc,derived
from bazireason.p2_prompts import extract_final_json

def semantic(raw,gold):
    ans=extract_final_json(raw)
    if not isinstance(ans,dict) or "answer" not in ans:return False
    v=ans["answer"]
    if type(v)==type(gold) and v==gold:return True
    if isinstance(gold,int) and isinstance(v,str) and v.strip().isdigit():return int(v.strip())==gold
    if isinstance(gold,list) and isinstance(v,str):
        if not gold:return v.strip() in ("无","没有","无匹配","[]")
        if gold and all(x.endswith("柱") for x in gold):
            return re.fullmatch(r"(?:(?:年|月|日|时)柱[、,， ]*)+",v.strip()) is not None and re.findall(r"[年月日时]柱",v)==gold
        if all(len(x)==1 for x in gold):return v.replace("、","").replace("，","").strip()=="".join(gold)
    return False

def main():
    rs=load("diagnostic","u8")
    groups=defaultdict(dict)
    for r in rs:groups[r["group"]][r["condition"]]=r
    tot=Counter()
    for rec in groups.values():
        st={k:v["observation"]["correct"] for k,v in rec.items()}
        sem={k:semantic(v["observation"]["raw"],v["answer"]) for k,v in rec.items()}
        x=rec["full"]; g=x["answer"]; raw=x["observation"]["raw"]
        if not st["full"]:
            tot["strict_misses"]+=1
            tot["schema_only_detected"]+=sem["full"]
            tot["producer_only_failed"]+=not st["producer"] and st["consumer"]
            tot["consumer_only_failed"]+=st["producer"] and not st["consumer"]
            tot["both_failed"]+=not st["producer"] and not st["consumer"]
            tot["both_passed"]+=st["producer"] and st["consumer"]
        if not sem["full"]:
            tot["semantic_misses"]+=1
            tot["semantic_prerequisite_failure"]+=not (sem["producer"] and sem["consumer"])
    (OUT/"diagnostic_localization.json").write_text(json.dumps(tot,indent=2))
    print(tot)
if __name__=="__main__":main()
