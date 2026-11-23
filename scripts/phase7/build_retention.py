"""Familiar rule and calendar panels on fresh inputs, grouped and frozen before arm evaluation."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from bazireason import train_data as TD, train_data6 as T6
from bazireason.intent import plan_for
from bazireason.p5_eval import make_item
from bazireason.prompts6 import apply
from bazireason.workflow import prompt_messages
from graphs import chart, historical_keys, rows, write_rows
from build_panel import years

def historical_calendar():
    prompts=set()
    dates=set()
    for path in Path("data").glob("phase[23456]/**/*.jsonl"):
        for r in rows(path):
            if r.get("task")=="chart":
                user=next((m["content"] for m in r["messages"] if m["role"]=="user"),None)
                prompts.add(user)
                dt=r.get("meta",{}).get("inputs",{}).get("datetime")
                if isinstance(dt,str):
                    dates.add(dt[:10])
    return prompts,dates

def main():
    rng=random.Random(710211)
    excluded=historical_keys()
    for path in Path("data/phase7").glob("*.jsonl"):
        for r in rows(path):
            if "chart" in r:
                excluded.add("".join(r["chart"].values()))
    result=[]
    for head in T6.WEIGHTS:
        for i in range(12):
            p=chart(rng,years("dev"),excluded)
            gender=rng.choice(["male","female"])
            intent=T6.concrete_intent(head,p,rng)
            q=T6.template_question(intent,p,gender,rng)
            plan=plan_for(intent,p,gender)
            group=f"retain-rule-{head}-{i}"
            item=make_item(group,"p7dev","template",p,gender,q,intent,plan,group)
            result.append(dict(id=group,group=group,condition=head,track="rules",item=item,
                               messages=apply("p6u8",[prompt_messages(item)])[0],cap=1100))
    prompts,dates=historical_calendar()
    original=TD.split_years()
    new_years = set(years("dev"))
    with patch.object(TD,"split_years",lambda:dict(original,valid=new_years)):
        cal=TD.generate_split("valid",710217,scale=12,version=2)
    groups=defaultdict(list)
    for r in cal:
        if r["task"]!="chart":
            continue
        dt=r["meta"].get("inputs",{}).get("datetime")
        user=prompt_messages(r)[-1]["content"]
        if user in prompts or (isinstance(dt,str) and dt[:10] in dates):
            continue
        prompts.add(user)
        groups[(r["family"],r["group_id"])].append(r)
    cells=defaultdict(list)
    for (family,g),rs in groups.items():
        cells[family].append((g,rs))
    for cs in cells.values():
        rng.shuffle(cs)
    n=0
    while n<240 and any(cells.values()):
        for family,cs in sorted(cells.items()):
            if not cs or n>=240:
                continue
            g,rs=cs.pop()
            for r in rs:
                group=f"retain-cal-{family}-{g}"
                result.append(dict(id="retain-"+r["id"],group=group,condition=family,track="calendar",item=r,
                                   messages=apply("p6u8",[prompt_messages(r)])[0],cap=700))
                n+=1
    assert n>=220,n
    write_rows("data/phase7/retention.jsonl",result)
    print(len(result),"total rows",n,"calendar",flush=True)
if __name__=="__main__":
    main()
