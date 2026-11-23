"""Frozen directed BaZi graph families and a strictly typed Phase 7 outcome."""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from lunar_python import EightChar
from lunar_python.util import LunarUtil
from bazireason import deep_rules as D, rules as R
from bazireason.conventions import BRANCHES, PILLARS
from bazireason.indep_rules import eight_char
from bazireason.p5_eval import head_line
from bazireason.prompts6 import SHORT_SYSTEM

SOURCES = ("冲", "六合", "害", "破", "驿马", "桃花", "华盖")
OPS = ("element", "main", "hidden", "number", "stage", "god", "positions", "count")
TERMS = dict(element="五行", main="本气藏干", hidden="全部藏干", number="藏干个数",
             stage="日主临此支的十二长生阶段", god="本气藏干相对日主的十神",
             positions="原局四柱中地支五行与此支相同的柱名", count="原局四个地支中五行与此支相同的个数")
HINTS = dict(element="五行名", main="一个天干", hidden="按本气在前的天干列表", number="整数",
             stage="阶段名", god="十神名", positions="按年月日时顺序的柱名列表，无则 []", count="整数")
# Offsets partition exact source->consumer edges; exclude historically adjacent clash/hidden operations.
def pool(part):
    offsets = {"train": (0, 1, 2), "diagnostic": (3,), "dev": (4,), "reserve": (5, 6)}[part]
    return [(src, op) for i, src in enumerate(SOURCES) for j, op in enumerate(OPS)
            if (i - j) % 7 in offsets and not (src == "冲" and op in ("main","hidden","number","god"))
            and not (src == "华盖" and op in ("number","element","count","positions"))]

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def write_rows(path, records):
    with Path(path).open("x") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")

def historical_keys():
    """Read only identifiers/input metadata for deduplication, never historical test scores."""
    keys = set()
    four = re.compile(r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]")
    paths = list(Path("data").glob("phase[23456]/**/*.jsonl")) + list(Path("results/phase6").glob("own_replies*.jsonl"))
    for path in paths:
        for record in rows(path):
            r = record.get("row", record)
            m = r.get("meta") or {}
            p = m.get("pillars") or r.get("pillars")
            if isinstance(p,dict) and all(k in p and isinstance(p[k],str) for k in PILLARS):
                keys.add("".join(p[k] for k in PILLARS))
            for msg in r.get("messages",[]):
                if msg["role"]=="user":
                    found = four.findall(msg["content"].split("\n")[0])
                    if len(found)>=4:
                        keys.add("".join(found[:4]))
                    break
    return keys

def chart(rng, years, excluded):
    while True:
        dt = datetime(rng.choice(years),1,1) + timedelta(days=rng.randrange(365),hours=rng.randrange(24),minutes=rng.randrange(60))
        ec = eight_char(dt)
        p = dict(zip(PILLARS,(ec.getYear(),ec.getMonth(),ec.getDay(),ec.getTime())))
        key = "".join(p.values())
        if key not in excluded:
            excluded.add(key)
            return p

def derived(src,p,pos):
    b = p[pos][1]
    if src in D.TRIPLE_STARS:
        return D.star_branches(src,p,pos)
    return D.branch_card(b)[src]

def calc(op,p,b):
    h = R.HIDDEN_STEMS[b]
    positions = [R.PILLAR_ZH[x] for x in PILLARS if R.BRANCH_ELEMENT[p[x][1]] == R.BRANCH_ELEMENT[b]]
    values = dict(element=R.BRANCH_ELEMENT[b], main=h[0], hidden=list(h), number=len(h),
                  stage=R.life_stage(p["day"][0],b),god=R.ten_god(p["day"][0],h[0]),positions=positions,count=len(positions))
    return values[op]

def independent(src,op,p,pos):
    natal = p[pos][1]
    partners = {"冲":"午未申酉戌亥子丑寅卯辰巳","六合":"丑子亥戌酉申未午巳辰卯寅",
                "害":"未午巳辰卯寅丑子亥戌酉申","破":"酉辰亥午丑申卯戌巳子未寅"}
    if src in partners:
        b = partners[src][BRANCHES.index(natal)]
    else:
        triples = ("申子辰","寅午戌","巳酉丑","亥卯未")
        b = {"驿马":"寅申亥巳","桃花":"酉卯午子","华盖":"辰戌丑未"}[src][next(i for i,t in enumerate(triples) if natal in t)]
    h = LunarUtil.ZHI_HIDE_GAN[b]
    e = LunarUtil.WU_XING_ZHI[b]
    day = p["day"][0]
    ix = BRANCHES.index(b)
    sign = 1 if "甲丙戊庚壬".find(day)>=0 else -1
    stage = EightChar.CHANG_SHENG[(EightChar._EightChar__CHANG_SHENG_OFFSET[day]+sign*ix)%12]
    positions = [R.PILLAR_ZH[k] for k in PILLARS if LunarUtil.WU_XING_ZHI[p[k][1]]==e]
    ans = dict(element=e,main=h[0],hidden=list(h),number=len(h),stage=stage,god=LunarUtil.SHI_SHEN[day+h[0]],positions=positions,count=len(positions))
    return b, ans[op]

def source_name(src,pos):
    n=R.POSITION_ZH[pos]+"支"
    if src in D.TRIPLE_STARS:
        return f"以{n}查出的{src}支"
    return f"与{n}成{ {'冲':'六冲','六合':'六合','害':'六害','破':'六破'}[src] }的地支"

def compact(x):
    return json.dumps(x,ensure_ascii=False,separators=(",",":"))

def case(p,src,op,pos,render,variant="full"):
    b,ans = independent(src,op,p,pos)
    assert (b,ans)==(derived(src,p,pos),calc(op,p,b))
    srcname = source_name(src,pos)
    hint = HINTS[op]
    question = [f"请取{srcname}，然后求它的{TERMS[op]}。",
                f"我想查{srcname}。用这个地支来算，{TERMS[op]}是什么？",
                f"针对{srcname}，我需要知道{TERMS[op]}。请按这张盘回答。"][render]
    if variant=="producer":
        question,ans,hint = f"{srcname}是哪一个字？",b,"一个地支"
    elif variant=="consumer":
        question = f"指定地支为{b}。请给出它的{TERMS[op]}。"
    elif variant=="oracle":
        question = f"已核对，所要使用的地支是{b}。"+question
    elif variant=="distractor":
        question = f"旁注，原局{R.POSITION_ZH[pos]}支是{p[pos][1]}。"+question
    elif variant=="cue":
        question = f"请先确定所查地支，再做{TERMS[op]}这一步。"+question
    if op in ("positions","count"):
        question=question.replace("它的原局","原局")  # 此支 refers to the derived branch
    user = head_line(p,None)+question+f"\n结论JSON 需含 answer（{hint}）。"
    return dict(messages=[dict(role="system",content=SHORT_SYSTEM),dict(role="user",content=user)],
                answer=ans,branch=b,chart=p,source=src,op=op,pos=pos,variant=variant,render=render)
    
def states(r):
    p,src,op,pos,b = (r[k] for k in ("chart","source","op","pos","branch"))
    target_text = source_name(src,pos)
    if op in ("positions","count"):
        values = "、".join(R.POSITION_ZH[k]+p[k][1]+R.BRANCH_ELEMENT[p[k][1]]+("（取）" if R.BRANCH_ELEMENT[p[k][1]]==R.BRANCH_ELEMENT[b] else "（略）") for k in PILLARS)
        lookup=f"{b}属{R.BRANCH_ELEMENT[b]}，逐柱比较：{values}"
    elif op in ("main","number","hidden","god"):
        lookup=f"{b}藏{'、'.join(R.HIDDEN_STEMS[b])}，本气为{R.HIDDEN_STEMS[b][0]}"
        if op=="god":
            lookup+=f"，以日干{p['day'][0]}查其本气十神"
    elif op=="stage":
        lookup=f"日干{p['day'][0]}，从{R._STAGE_START[p['day'][0]]}长生起{'顺' if R.is_yang(p['day'][0]) else '逆'}数至{b}"
    else:
        lookup=f"查地支五行表：{b}属{R.BRANCH_ELEMENT[b]}"
    answer=compact(r["answer"])
    final='\n结论JSON: '+compact({"answer":r["answer"]})
    prose=f"1. 原局{R.POSITION_ZH[pos]}支为{p[pos][1]}，查{target_text}可得{b}。\n2. 将{b}用于下步，{lookup}。因此{TERMS[op]}为{answer}。"
    structured=f"1. 来源：{target_text}；原局输入={p[pos][1]}；所得地支={b}。\n2. 本步输入={b}；核对：{lookup}；{TERMS[op]}={answer}。"
    return prose+final,structured+final

def parse(text):
    matches=re.findall(r"^\s*结论\s*JSON\s*[:：]\s*(.*?)\s*$",text,re.M|re.I)
    if len(matches)!=1:
        return None
    def unique(items):
        if len(items)!=len(dict(items)):
            raise ValueError("duplicate key")
        return dict(items)
    try:
        d=json.loads(matches[0],object_pairs_hook=unique)
    except (TypeError,ValueError):
        return None
    return d["answer"] if isinstance(d,dict) and set(d)=={"answer"} else None

def correct(raw,gold):
    value=parse(raw)
    return type(value) is type(gold) and value==gold
