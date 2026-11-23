"""Create sparse, expanded, and plan-first graph continuations with matched operands."""
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import random
import sys

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from graphs import BRANCHES, OPS, PILLARS, R, SOURCES, calc, chart, compact, derived
from graphs import independent, pool, rows, source_name, write_rows
from bazireason.finetune import PromptCompletionDataset
from bazireason.p5_eval import head_line
from bazireason.prompts6 import SHORT_SYSTEM

def archived_module(name, path):
    spec = spec_from_file_location(name, ROOT / path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

p8 = archived_module("phase8_curriculum", "scripts/phase8/build_training.py")
panel8 = archived_module("phase8_key_tools", "scripts/phase8/panel.py")
OUT = Path("data/phase9")
SEED = 910217
HOLD = {("害","stage"), ("破","god"), ("驿马","positions"), ("桃花","element")}
N_GROUPS = 72 * len(SOURCES)
TERMS = {
    "element": ("地支的五行","它在木火土金水中属哪一行","论此支本身的五行属性"),
    "main": ("藏干中的本气","这个支以哪个天干为主气","支中占本气的那一干"),
    "hidden": ("其中所藏各干，按本气在前列出","全套藏干，请依本中余气次序","该支的藏干清单，主气居首"),
    "number": ("藏干的数目","此支共藏几个天干","所藏天干合计几项"),
    "stage": ("日主临这一支的十二长生阶段","按日干取十二运，此支为哪一宫","以日干排长生十二宫，所临的阶段名称"),
    "god": ("以日干为我，此支本气的十神","日主与此支所藏主气构成哪一种十神关系","取该支本气藏干，问对日主的十神名称"),
    "positions": ("四柱中地支与它同五行的柱名，按年月日时列出","按年月日时顺序列出原局地支与此支五行相同的各柱","在原局寻找同五行的地支，按四柱原顺序给出柱名"),
    "count": ("四个本命地支中与它五行相同的个数","原局中有几柱的地支与此支同五行","数一数同此支五行的原局地支，一共几柱"),
}
OP_NAME = dict(element="地支五行", main="本气藏干", hidden="藏干列表", number="藏干数量",
               stage="十二长生",god="本气十神",positions="同五行柱位",count="同五行柱数")


def all_archive_keys():
    keys = set()
    paths = list(Path("data").glob("phase[2345678]/**/*.jsonl"))
    paths += list(Path("results/phase6").glob("own_replies*.jsonl"))
    for path in paths:
        for record in rows(path):
            keys.update(panel8.keys_in(record.get("row", record)))
    return keys


def chart_groups(rng, used):
    records = []
    from bazireason.train_data import split_years
    train_years = sorted(split_years()["train"])
    sparse = set(pool("train"))
    for i in range(N_GROUPS):
        source = SOURCES[i%len(SOURCES)]
        ops = [o for o in OPS if (source,o) in sparse]
        while True:
            pillars = chart(rng, train_years, used)
            if derived(source,pillars,"year") != derived(source,pillars,"day"):
                break
        for pos in ("year","day"):
            branch = derived(source,pillars,pos)
            selected = rng.sample(ops,2)
            # Both questions in a given minimal pair use the same layout and register.
            layout, register = rng.randrange(4), rng.randrange(3)
            for op in selected:
                records.append(dict(id=f"p9-{len(records):05d}", chart_group=i, chart=pillars,
                    source=source, pos=pos, branch=branch, sparse_op=op, layout=layout, register=register))
    return records


def expand(records,rng):
    """Swap op labels only within the same (day stem, derived branch) cell.

    This preserves actual joint operand × operation counts. Reject duplicate operations
    within a chart/source question pair and all protected edges. At any accepted move
    the sum of squared edge counts decreases or stays constant.
    """
    labels = [r["sparse_op"] for r in records]
    bins = defaultdict(list)
    edges = Counter()
    for i,r in enumerate(records):
        bins[(r["chart"]["day"][0], r["branch"])].append(i)
        edges[(r["source"],labels[i])]+=1
    bins = [v for v in bins.values() if len(v)>1]
    for _ in range(200_000):
        i,j = rng.sample(rng.choice(bins),2)
        lhs,rhs = records[i]["source"],records[j]["source"]
        a,b = labels[i],labels[j]
        if a==b or lhs==rhs or (lhs,b) in HOLD or (rhs,a) in HOLD:
            continue
        if labels[i^1] == b or labels[j^1] == a:
            continue
        if edges[lhs,a]+edges[rhs,b] < edges[lhs,b]+edges[rhs,a]+2:
            continue
        labels[i],labels[j] = b,a
        edges[lhs,a]-=1;edges[rhs,b]-=1
        edges[lhs,b]+=1;edges[rhs,a]+=1
    return labels


def question(r,op):
    s = source_name(r["source"],r["pos"])
    term = TERMS[op][r["register"]]
    variants = (f"取{s}作目标。请给出{term}。",
                f"我想知道{term}。这里说的支，取{s}。",
                f"先由命盘确定{s}，再按这个结果处理以下所问：{term}。",
                f"本题使用{s}。针对所取之支，所求项目为：{term}。")
    return head_line(r["chart"],None)+variants[r["layout"]]+"\n结论JSON 需含 answer。"


def derivation(r,op):
    p,b = r["chart"],r["branch"]
    d = p["day"][0]
    hidden = R.HIDDEN_STEMS[b]
    elem = R.BRANCH_ELEMENT[b]
    if op=="stage":
        start = R._STAGE_START[d]
        sign = 1 if R.is_yang(d) else -1
        steps = sign*(BRANCHES.index(b)-BRANCHES.index(start)) % 12
        stage = R.LIFE_STAGES[steps]
        assert stage==R.life_stage(d,b)
        return f"{d}干长生在{start}，{'顺' if sign==1 else '逆'}行至{b}共{steps}步。以长生为第0位，第{steps}位是{stage}。"
    if op=="god":
        return f"{b}藏{'、'.join(hidden)}，本气为{hidden[0]}。日干{d}，{R.ten_god_reason(d,hidden[0])}。"
    if op=="element":
        return f"地支五行表中，{b}对应{elem}，故取{elem}。"
    if op in ("main","hidden","number"):
        last = {"main":f"本气排在第一，是{hidden[0]}","hidden":"按本气在前保留整张清单","number":f"逐干计数，共{len(hidden)}项"}[op]
        return f"{b}的藏干依次为{'、'.join(hidden)}。{last}。"
    values = "、".join(R.POSITION_ZH[k]+p[k][1]+"属"+R.BRANCH_ELEMENT[p[k][1]] for k in PILLARS)
    picked = [R.PILLAR_ZH[k] for k in PILLARS if R.BRANCH_ELEMENT[p[k][1]]==elem]
    tail = f"因此共{len(picked)}柱" if op=="count" else f"依序取{'、'.join(picked) if picked else '空列表'}"
    return f"{b}属{elem}。原局逐柱查得{values}。{tail}。"


def example(r,op,plan):
    p,src,b,pos = r["chart"],r["source"],r["branch"],r["pos"]
    ref,ans = independent(src,op,p,pos)
    assert ref == b and type(ans) is type(calc(op,p,b)) and ans == calc(op,p,b)
    pos_text = R.POSITION_ZH[pos]+"支"
    extract = f"{pos_text}是{p[pos][1]}，查{source_name(src,pos)}得{b}。"
    step = derivation(r,op)
    if plan:
        extra = "，日干="+p["day"][0] if op in ("god","stage") else ""
        body = f"计划：先取{src}支，再求{OP_NAME[op]}。\n1. {extract}\n2. 执行{OP_NAME[op]}，输入支={b}{extra}。{step}"
    else:
        body = f"1. 依命盘核对题目所指的地支。{extract}\n2. 接下来用刚才所得的{b}支计算。{step}"
    final = body+"\n结论JSON: "+compact({"answer":ans})
    return dict(id=r["id"], family="p9_graph", task="p9_graph", chart_group=r["chart_group"],
                source=src,op=op,pos=pos,chart=p,branch=b,answer=ans,layout=r["layout"],register=r["register"],
                messages=[dict(role="system",content=SHORT_SYSTEM),dict(role="user",content=question(r,op)),dict(role="assistant",content=final)])


def edge_stats(group):
    counts=Counter((r["source"],r["op"]) for r,_,_ in group)
    producers=Counter(); consumers=Counter()
    for (s,o),n in counts.items():
        producers[s]+=n;consumers[o]+=n
    n=len(group)
    return dict(edges={s+"→"+o:num for (s,o),num in sorted(counts.items())}, num_edges=len(counts),
                producer_counts=producers,consumer_counts=consumers,
                mutual_information_bits=sum((num/n)*math.log2(n*num/producers[s]/consumers[o]) for (s,o),num in counts.items()),
                new_edges=sorted(s+"→"+o for s,o in set(counts)-set(pool("train"))))


def main():
    assert p8.sha(p8.PARENT)==p8.PARENT_SHA
    rng = random.Random(SEED)
    cfg = yaml.safe_load(Path("configs/phase7/g.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"],local_files_only=True)
    ds = PromptCompletionDataset([],tok)
    def pack(row):
        ids, offset = ds.process(row)
        assert len(ids)<=2048
        return row,len(ids),len(ids)-offset+1

    common, common_tokens = p8.common_replay(rng,pack)
    used = all_archive_keys()
    n_old=len(used)
    records = chart_groups(rng, used)
    expanded = expand(records,rng)
    graphs = {a:[] for a in "sxp"}
    for r, new_op in zip(records,expanded,strict=True):
        for a in graphs:
            graphs[a].append(pack(example(r,r["sparse_op"] if a=="s" else new_op,a=="p")))
    def operand_counts(graph):
        return Counter((r["chart"]["day"][0],r["branch"],r["op"]) for r,_,_ in graph)
    assert operand_counts(graphs["s"])==operand_counts(graphs["x"])
    assert all((r["source"],r["op"]) not in HOLD for graph in graphs.values() for r,_,_ in graph)
    for x,p in zip(graphs["x"],graphs["p"],strict=True):
        assert x[0]["messages"][:2]==p[0]["messages"][:2] and x[0]["answer"]==p[0]["answer"]
    while (len(common)+len(records))%4:
        i = next(i for i,r in enumerate(common) if p8.category(r[0])=="p7_graph")
        row=common.pop(i)
        common_tokens["p7_graph"]-=row[1]
    info={a:edge_stats(group) for a,group in graphs.items()}
    assert info["s"]["num_edges"]==22 and info["x"]["num_edges"]>=49
    full = {a: common+graph for a,graph in graphs.items()}
    n=len(full["s"])
    order=list(range(n));rng.shuffle(order)
    shared = [round(sum(group[i][1] for group in full.values())/3) for i in range(n)]
    batches=sorted(order,key=shared.__getitem__)
    stats={}
    for a,ls in full.items():
        sizes=[ls[i][1] for i in batches]
        stats[a]=dict(rows=n,updates=n//4,sequence=sum(sizes),supervised_logged=sum(x[2] for x in ls),
                      padded=sum(2*min(2048,1+32*math.ceil(max(sizes[k:k+2])/32)) for k in range(0,n,2)))
    print(stats, flush=True)
    for key in ("rows","updates","sequence","supervised_logged","padded"):
        vals=[s[key] for s in stats.values()]
        assert max(vals)/min(vals)<=1.03,(key,vals)
    OUT.mkdir(exist_ok=True, parents=True)
    valid=list(rows("data/phase7/g/valid.jsonl"))
    hashes={}
    for a,ls in full.items():
        folder=OUT/a;folder.mkdir(exist_ok=False)
        write_rows(folder/"train.jsonl",(dict(ls[i][0],phase9_sort_length=shared[i]) for i in order))
        write_rows(folder/"valid.jsonl",valid)
        run="bazireason-phase9-"+a
        cfg_path=Path("configs/phase9")/(a+".yaml")
        cfg_path.parent.mkdir(exist_ok=True)
        settings=dict(cfg,run_name=run,seed=SEED,init_adapter=str(p8.PARENT),adapter_dir=f"artifacts/adapters/{run}",
                      log_dir=f"results/phase9/training/{a}",learning_rate=5e-5,end_learning_rate=5e-6,
                      data=dict(train=str(folder/"train.jsonl"),valid=str(folder/"valid.jsonl"),valid_rows=len(valid)),
                      steps_per_eval=n//2,steps_per_save=n//2)
        with cfg_path.open("x") as f: yaml.safe_dump(settings,f,sort_keys=False,allow_unicode=True)
        hashes[a]=dict(train=p8.sha(folder/"train.jsonl"),valid=p8.sha(folder/"valid.jsonl"),config=p8.sha(cfg_path))
    keyfile=OUT/"train_chart_keys.json"
    with keyfile.open("x") as f:
        json.dump(sorted({"".join(r["chart"][k] for k in PILLARS) for r in records}),f,ensure_ascii=False)
    manifest=dict(seed=SEED,parent=str(p8.PARENT),parent_sha256=p8.PARENT_SHA,stats=stats,hashes=hashes,graph_stats=info,
                  common_rows=len(common),common_tokens=common_tokens,fresh_graph_rows=len(records),
                  fresh_chart_count=N_GROUPS,unique_prior_chart_keys=n_old,keys_sha256=p8.sha(keyfile),
                  protected_edges=sorted(s+"→"+o for s,o in HOLD),
                  condition="Exact shared replay and per-row chart, source, anchor, branch and prompt style. S/X have identical op × day stem × derived branch histograms. X/P share full questions, independent labels and calculation text.",
                  training_script_sha256=p8.sha(Path(__file__)))
    with (OUT/"training_manifest.json").open("x") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2);f.write("\n")
    print("Training data frozen",flush=True)

if __name__=="__main__":
    main()
