"""Read back frozen training streams and validate counterfactuals, labels and restrictions."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/phase7"))
from graphs import independent, parse, PILLARS, SOURCES, OPS, pool, rows

DATA=ROOT/"data/phase9"
OUT=ROOT/"results/phase9/training_audit.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m=json.loads((DATA/"training_manifest.json").read_text())
    meta={}
    for a in ("s","x","p"):
        assert digest(DATA/a/"train.jsonl")==m["hashes"][a]["train"]
        assert digest(DATA/a/"valid.jsonl")==m["hashes"][a]["valid"]
        assert digest(ROOT/f"configs/phase9/{a}.yaml")==m["hashes"][a]["config"]
    cs=Counter()
    features={a:Counter() for a in ("s","x","p")}
    edges={a:Counter() for a in features}
    support={a:defaultdict(set) for a in features}
    chart_groups={a:defaultdict(dict) for a in features}
    new_ids=set();user_prompts={a:set() for a in features};fresh_keys=set()
    for batch in zip(*(rows(DATA/a/"train.jsonl") for a in ("s","x","p")),strict=True):
        s,x,p=batch
        assert len({r["phase9_sort_length"] for r in batch})==1
        if s["family"]!="p9_graph":
            assert s==x==p
            cs["common_rows"]+=1
            if s["family"]=="p7_graph":
                assert (s["source"],s["op"]) in pool("train")
                cs["shared_existing_graph_replays"]+=1
            continue
        cs["fresh_rows"]+=1
        assert s["id"] not in new_ids;new_ids.add(s["id"])
        assert all(s[k]==x[k]==p[k] for k in ("id","chart","chart_group","source","pos","branch","layout","register"))
        assert x["op"]==p["op"] and x["messages"][:2]==p["messages"][:2]
        cs["s_x_changed_op"]+=s["op"]!=x["op"]
        for a,r in zip(("s","x","p"),batch,strict=True):
            src,op,chart,b,position,group=(r[k] for k in ("source","op","chart","branch","pos","chart_group"))
            key="".join(chart[k] for k in PILLARS);fresh_keys.add(key)
            ref=independent(src,op,chart,position)
            assert ref==(b,r["answer"])
            predicted=parse(r["messages"][-1]["content"])
            assert type(predicted) is type(r["answer"]) and predicted==r["answer"]
            user=r["messages"][1]["content"]
            assert user not in user_prompts[a] and user.endswith("结论JSON 需含 answer。")
            user_prompts[a].add(user)
            assert src+"→"+op not in m["protected_edges"]
            features[a][(chart["day"][0],b,op)]+=1
            edges[a][(src,op)]+=1
            support[a][op].add(b)
            chart_groups[a][(group,position)][op]=(src,b,key,r["layout"],r["register"])
    assert features["s"]==features["x"]==features["p"]
    assert fresh_keys==set(json.loads((DATA/"train_chart_keys.json").read_text()))
    assert cs["common_rows"]==m["common_rows"] and cs["fresh_rows"]==m["fresh_graph_rows"]
    for a in features:
        for pairs in chart_groups[a].values():
            assert len(pairs)==2 and len(set(pairs.values()))==1
        assert len(chart_groups[a])==2*len(fresh_keys)
    assert set(edges["s"])==set(pool("train"))
    unseen={(s,o) for s in SOURCES for o in OPS}-set(edges["x"])
    assert len(unseen)==5
    meta.update(counts=cs, added_vs_sparse=len(set(edges["x"])-set(edges["s"])),
                excluded_edges_expanded=sorted(s+"→"+o for s,o in unseen),
                input_branch_cardinality_by_operation={o:len(b) for o,b in sorted(support["s"].items())},
                observed_joint_day_branch_operation_cells=len(features["x"]),
                total_possible_joint_cells=10*12*8,
                counterfactual_pairs_per_arm=len(chart_groups["x"]),
                dataset_checks=["Pre-existing graph replay within original training edges",
                                "Identical row, source, natal chart, anchor and realized intermediate across all arms",
                                "No repeated fresh prompt in any arm",
                                "Exact matched (day stem, derived branch, operation) histogram",
                                "Each paired question changes the demanded operation with fixed source, anchor, chart, branch and template style",
                                "Independent library/table reconstruction agrees with every strict typed final answer"])
    OUT.parent.mkdir(exist_ok=True,parents=True);OUT.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print(OUT.read_text())

if __name__=="__main__":
    main()
