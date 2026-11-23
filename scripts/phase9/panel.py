"""Build Phase 9 transfer panels from frozen input identities only."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/phase7"))
from graphs import (BRANCHES, PILLARS, SHORT_SYSTEM, SOURCES, TERMS, R, calc, derived, independent, rows, write_rows)
from bazireason.indep_rules import eight_char
from bazireason.p5_eval import head_line
from bazireason.train_data import split_years

OUT = ROOT / "data/phase9"
SEED = 910409
BRIDGE_SEED = 910421
PRODUCERS = ("三合生", "三合旺", "三会孟", "三会季")
PAIRS = (("stage", "god"), ("stage", "element"), ("hidden", "god"), ("hidden", "element"))
DEEP = (("冲", "华盖"), ("破", "驿马"), ("驿马", "害"), ("六合", "破"))
BRIDGE = (("害", "stage"), ("破", "god"), ("驿马", "positions"), ("桃花", "element"))
INDEX = {"三合生": 0, "三合旺": 1, "三会孟": 0, "三会季": 2}
FOUR = re.compile("[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]")
REFERENCE = ("查支资料（不要求原局凑齐三支）：三合每组的生、旺、墓顺序为申子辰、亥卯未、寅午戌、巳酉丑；"
             "三会每组的孟、仲、季顺序为寅卯辰、巳午未、申酉戌、亥子丑。")
FOOTER = "\n结论JSON 仅含 answer 这一项。"
REQUEST = {
    "stage": ("日元落在此支，按十二长生应记作哪一宫？", "从日干看，到这一支时，长生十二运走到了哪一段？"),
    "god": ("把此支的本气当作对照天干，依盘中日元判它的十神称呼。", "日干是我，以这支所藏的主气为对方，十神该如何定名？"),
    "element": ("请报此支本身的五行归属。", "在木火土金水这五类之中，这一支应归哪类？"),
    "hidden": ("请从本气起按通行表依序列出此支的藏干，填为一张列表。", "这个支里面藏了哪几个天干？按藏干表的次序写成列表，本气放在首位。"),
    "positions": ("查原盘年、月、日、时四支，按原顺序列出地支五行同此支的柱名。", "原盘哪些柱的地支与这一支同属一个五行？按年月日时把柱名写成列表。")
}
METHOD = {"stage": "按日元查十二长生", "god": "以本气与日元定十神", "hidden": "列所藏天干", "element": "查地支五行"}
ROLES = {"三合生": ("生支", "生位"), "三合旺": ("旺支", "旺位"), "三会孟": ("孟支", "孟位"), "三会季": ("季支", "季位")}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def chart_key(chart):
    return "".join(chart[k] for k in PILLARS)


def keys_in(record):
    """Extract declared charts and explicit pillar mentions in user-side messages."""
    row = record.get("row", record)
    item = row.get("item", row)
    keys = set()
    for c in (row.get("chart"), row.get("pillars"), item.get("pillars"), item.get("meta", {}).get("pillars")):
        if isinstance(c, dict) and all(isinstance(c.get(k),str) for k in PILLARS):
            keys.add(chart_key(c))
    for raw in (item, row):
        for m in raw.get("messages", []):
            if m["role"] == "user":
                for line in m["content"].splitlines():
                    found = FOUR.findall(line)
                    if len(found) >= 4:
                        keys.add("".join(found[:4]))
    return keys


def exclusion_files():
    training = sorted((OUT).glob("*/train.jsonl"))
    if {p.parent.name for p in training} != {"s","x","p"}:
        raise ValueError("Build the realized Phase 9 training sets before generating evaluation charts")
    sources = sorted((ROOT / "data").glob("phase[2345678]/**/*.jsonl"))
    sources += sorted((ROOT / "results/phase6").glob("own_replies*.jsonl"))
    live = training + sorted((OUT).glob("*/valid.jsonl"))
    if (OUT / "retention.jsonl").is_file():
        live.append(OUT / "retention.jsonl")
    return sources, live


def identities(paths):
    used = set()
    for path in paths:
        for record in rows(path):
            used.update(keys_in(record))
    return used


def sample_chart(rng, used):
    years = sorted(split_years()["test"])
    while True:
        y = rng.choice(years)
        start = datetime(y, 1, 1)
        moment = start + timedelta(minutes=rng.randrange((datetime(y+1,1,1) - start).days*1440))
        ec = eight_char(moment)
        c = dict(zip(PILLARS, (ec.getYear(), ec.getMonth(), ec.getDay(), ec.getTime())))
        key = chart_key(c)
        if key not in used:
            used.add(key)
            return y, c


def project_branch(source, branch):
    if source in PRODUCERS:
        table = R.TRIPLE_HARMONIES if source.startswith("三合") else R.DIRECTIONAL_TRIPLES
        group, = (g for g in table if branch in g)
        return group[INDEX[source]]
    temp = {k:"甲"+branch for k in PILLARS}
    return derived(source, temp, "year")


def reference_branch(source, branch):
    # Cyclic index arithmetic independently implements the printed triad table.
    i = BRANCHES.index(branch)
    if source == "三合旺" or source == "三合生":
        peak = (9 * i) % 12
        return BRANCHES[(peak if source == "三合旺" else peak - 4) % 12]
    if source == "三会孟" or source == "三会季":
        first = (i - ((i+1) % 3)) % 12
        return BRANCHES[(first if source == "三会孟" else first + 2) % 12]
    temp = {k:"甲"+branch for k in PILLARS}
    return independent(source,"element",temp,"year")[0]


def checked_answer(stages, op, chart, pos):
    b = chart[pos][1]
    independent_b = b
    path = []
    for source in stages:
        b = project_branch(source,b)
        independent_b = reference_branch(source,independent_b)
        assert b == independent_b
        path.append(b)
    # Use the independent consumer tables, passing a synthetic clash partner so that
    # its resulting target branch is b. Retain the original day stem and four branches
    # for position-dependent answers: its temporary input is outside the four pillars.
    temp = dict(chart, synthetic="甲"+BRANCHES[(BRANCHES.index(b)+6) % 12])
    check_b, gold = independent("冲",op,temp,"synthetic")
    assert check_b == b and gold == calc(op,chart,b)
    return path, gold


def source_phrase(source, pos, style):
    poszh = {"year":"年支", "day":"日支"}[pos]
    if source in PRODUCERS:
        group = source[:2]
        name, role = ROLES[source]
        return (f"先找出{poszh}所属的{group}组，取组内的{name}。", f"按{group}分组，找本命{poszh}所在的组，并定位该组{role}所指的那一支。")[style]
    if source in ("华盖","桃花","驿马"):
        return (f"由本命{poszh}定{source}，取{source}所落的支。", f"以原盘{poszh}为依据，查得的{source}支作以下问题的对象。")[style]
    name = {"冲":"六冲","害":"六害","破":"六破","六合":"六合"}[source]
    return (f"取与本命{poszh}构成{name}的另一支。", f"沿地支{name}的配对关系，从盘内{poszh}找出配对的那一支。")[style]


def record(chart, year, pos, assigned, source, op, condition, group, prefix, style, gold, text, path, legend):
    user = head_line(chart,None) + (REFERENCE+"\n" if legend else "") + text + FOOTER
    return dict(id=f"{group}-{condition}-{op}", group=group, condition=condition, source=source,
                family=f"{source}→{op}", op=op, pos=pos, branch=path[-1], intermediates=path,
                chart=chart, sampled_year=year, answer=gold, writer_form=style, assigned_source=assigned,
                source_request=prefix, reference=legend,
                messages=[{"role":"system","content":SHORT_SYSTEM},{"role":"user","content":user}])


def build_transfer(used):
    rng = random.Random(SEED)
    records = []
    letter_order = list(range(4))*24
    random.Random(SEED+2).shuffle(letter_order)
    for si, source in enumerate(PRODUCERS):
        # The four pair recipes occur six times in each producer. Two different
        # output operations on a given chart always share the whole producer clause.
        layout = [(pi, rep) for pi in range(4) for rep in range(6)]
        rng.shuffle(layout)
        for ix,(pi,rep) in enumerate(layout):
            group = f"p9-transfer-{si}-{ix:02}"
            ops = PAIRS[pi]
            pos = ("year","day")[(rep + ((pi+1)//2))%2]
            style = (rep + rep//2 + ((pi+1)//2) + si) % 2
            chosen = ops[rep%2]
            year, chart = sample_chart(rng,used)
            src = source_phrase(source,pos,style)
            for op in ops:
                path,ans = checked_answer([source],op,chart,pos)
                body = src + REQUEST[op][style]
                fixed = f"请取{('年支' if pos=='year' else '日支')}所在{source[:2]}组的{ROLES[source][0]}，然后求它的{TERMS[op]}。"
                if op=="hidden":
                    fixed = fixed.removesuffix("。")+"，从本气开始按表序写成列表。"
                for cond, gold, text, legend in (
                        ("novel",ans,body,True), ("familiar",ans,fixed,True),
                        ("consumer",ans,f"以下所用的地支已经确定为{path[-1]}。"+REQUEST[op][style],True)):
                    records.append(record(chart,year,pos,source,source,op,cond,group,src,style,gold,text,path,legend))
                if op != chosen:
                    continue
                records.append(record(chart,year,pos,source,source,op,"closed",group,src,style,ans,body,path,False))
                labels = [name for name in METHOD if name != op]
                rng.shuffle(labels)
                labels.insert(letter_order[si*24+ix],op)
                menu = "，".join(letter+"="+METHOD[operation] for letter,operation in zip("ABCD",labels))
                meta = f"待判断的请求是：“{body}”\n本次只需要选择它最后要做哪种运算。{menu}。请填对应的一个大写字母。"
                rec = record(chart,year,pos,source,source,op,"operation",group,src,style,"ABCD"[labels.index(op)],meta,path,True)
                rec["options"] = dict(zip("ABCD",labels))
                records.append(rec)
                records.append(record(chart,year,pos,source,source,"branch","producer",group,src,style,path[-1],src+"这一步取出的地支是哪一支？",path,True))
                a,b = DEEP[(si+pi)%4]
                path3, gold3 = checked_answer([a,b],op,chart,pos)
                first = source_phrase(a,pos,style)
                if b in ("华盖","驿马","桃花"):
                    follow = f"再以上一步所得的支为起点，按其三合组查它的{b}支。"
                else:
                    follow = f"接下来，按地支{b}的配对规则取上一步所得地支的配对支。"
                rec = record(chart,year,pos,source,a+"→"+b,op,"deep",group,first+follow,style,gold3,
                             first+follow+REQUEST[op][style],path3,False)
                rec["stages"] = [a,b]
                records.append(rec)
    return records


def build_bridge(used):
    rng = random.Random(BRIDGE_SEED)
    result = []
    for si,(src, op) in enumerate(BRIDGE):
        layout = [(pos,style) for pos in ("year","day") for style in (0,1)]*10
        rng.shuffle(layout)
        for ix,(pos,style) in enumerate(layout):
            year,chart = sample_chart(rng,used)
            path,gold = checked_answer([src],op,chart,pos)
            text = source_phrase(src,pos,style)
            result.append(record(chart,year,pos,src,src,op,"bridge",f"p9-bridge-{si}-{ix:02}",
                                 text,style,gold,text+REQUEST[op][style],path,False))
    return result


def verify_functions():
    def sig(stages):
        v = list(BRANCHES)
        for stage in stages:
            for b in BRANCHES:
                assert project_branch(stage,b) == reference_branch(stage,b)
            v = [project_branch(stage,b) for b in v]
        return "".join(v)
    base = {sig([s]) for s in SOURCES} | {"".join(BRANCHES)}
    new = {sig([s]) for s in PRODUCERS}
    deep = {sig(stages) for stages in DEEP}
    assert base.isdisjoint(new | deep) and new.isdisjoint(deep) and len(new)==len(deep)==4


def check(records, other, name):
    assert len(records) == len({r["id"] for r in records})
    assert len(records) == len({r["messages"][-1]["content"] for r in records})
    groups = {}
    for r in records:
        k = chart_key(r["chart"])
        assert k not in other
        groups.setdefault(r["group"],set()).add(k)
        op = r["op"]
        stages = r.get("stages",[r["source"]])
        path, gold = checked_answer(stages,op if op!="branch" else "element",r["chart"],r["pos"])
        assert r["branch"] == path[-1] and r["intermediates"] == path
        if op == "branch":
            gold = path[-1]
        elif r["condition"] == "operation":
            gold, = [k for k,v in r["options"].items() if v == op]
        assert type(r["answer"]) is type(gold) and r["answer"] == gold
    unique = set.union(*groups.values())
    assert all(len(v)==1 for v in groups.values()) and len(unique)==len(groups)
    if name == "transfer":
        assert len(unique)==96 and Counter(r["condition"] for r in records) == dict(
            novel=192,familiar=192,consumer=192,operation=96,producer=96,closed=96,deep=96)
        cells = Counter((r["source"],r["op"]) for r in records if r["condition"]=="novel")
        assert len(cells)==16 and set(cells.values())=={12}
        for group in groups:
            pairs = [r for r in records if r["group"]==group]
            novel = [r for r in pairs if r["condition"]=="novel"]
            assert len(novel)==2 and novel[0]["source_request"]==novel[1]["source_request"]
            assert novel[0]["answer"]!=novel[1]["answer"]
            closed, = [r for r in pairs if r["condition"]=="closed"]
            match, = [r for r in novel if r["op"]==closed["op"]]
            assert match["messages"][1]["content"].replace(REFERENCE+"\n","")==closed["messages"][1]["content"]
    else:
        assert len(unique)==160 and set(Counter(r["family"] for r in records).values())=={40}
    return unique


def main(action):
    verify_functions()
    past, live = exclusion_files()
    train_index = json.loads((OUT / "training_manifest.json").read_text())
    for path in live:
        if path.parent.name not in ("s","x","p"):
            continue
        assert digest(path) == train_index["hashes"][path.parent.name][path.stem]
        for row in rows(path):
            if "source" in row and "op" in row:
                assert row["source"] in SOURCES
                assert (row["source"],row["op"]) not in BRIDGE
    past_keys = identities(past)
    used = past_keys | identities(live)
    live_hashes = {str(p.relative_to(ROOT)):digest(p) for p in live}
    for panel,fn,seed in (("transfer",build_transfer,SEED),("bridge",build_bridge,BRIDGE_SEED)):
        path = OUT / (panel+".jsonl")
        manifest = OUT / (panel+"_manifest.json")
        if action == "build":
            if path.exists() or manifest.exists():
                raise FileExistsError(f"Already generated {panel}")
            data = fn(used.copy())
        else:
            lock = json.loads(manifest.read_text())
            assert lock["sha256"] == digest(path) and lock["generator_sha256"] == digest(Path(__file__))
            for name,h in lock["training_inputs"].items():
                assert digest(ROOT/name)==h
            data = list(rows(path))
        keys = check(data,used,panel)
        used |= keys
        if action == "build":
            write_rows(path, data)
            manifest.write_text(json.dumps(dict(panel=panel, sha256=digest(path), generator_sha256=digest(Path(__file__)),
                seed=seed, training_inputs=live_hashes, historical_files=len(past), historical_keys=len(past_keys),
                groups=len(keys), rows=len(data), conditions=dict(Counter(r["condition"] for r in data)),
                families=dict(Counter(r["family"] for r in data if r["condition"] in ("novel","bridge"))),
                protocol="Group = chart. All primary producer×consumer cells have equal weight. Pair operations within each chart. Primary includes complete source lookup table. Closed removes only that table. Canonical wording uses familiar consumer vocabulary with the same answer schema. Operation classification uses exact single-letter answers. Depth-three and bridge are unaided. Outcomes were not consulted."), indent=2, ensure_ascii=False)+"\n")
        print(f"{panel} {len(data)} rows, {len(keys)} charts, {digest(path)}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("build","verify"))
    main(parser.parse_args().action)
