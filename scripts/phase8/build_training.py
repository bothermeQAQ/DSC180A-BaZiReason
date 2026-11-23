"""Build paired isolated-consumer curricula with common Phase 7 G replay."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import yaml
from lunar_python import EightChar
from lunar_python.util import LunarUtil
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase7"))
from graphs import chart, compact, historical_keys, rows, write_rows
from bazireason import rules as R
from bazireason.conventions import BRANCHES, PILLARS, STEMS
from bazireason.finetune import PromptCompletionDataset
from bazireason.p5_eval import head_line
from bazireason.prompts6 import SHORT_SYSTEM

OUT = Path("data/phase8")
SEED = 810117
PARENT = Path("artifacts/adapters/bazireason-phase7-g/adapters.safetensors")
PARENT_SHA = "790f87c137011c24e4cdf21dfe9dfabc7936612f0b19d23b9092cdae8e6778f9"
CHART_YEARS = tuple(range(1870, 1890))
FOUR = re.compile(r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]")
CAP = 2048

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def chart_key(r):
    meta = r.get("item", r).get("meta", {})
    p = r.get("chart") or meta.get("pillars")
    if p and all(k in p for k in PILLARS):
        return "".join(p[k] for k in PILLARS)
    msg = next((m["content"] for m in r.get("messages", []) if m["role"] == "user"), "")
    pillars = FOUR.findall(msg.partition("\n")[0])
    return "".join(pillars[:4]) if len(pillars) >= 4 else None

def category(r):
    if r["task"] == "chart":
        return "calendar"
    return r["family"] if r["family"] in ("p7_graph", "procedure", "contrast") else "other"

def common_replay(rng, pack):
    buckets = defaultdict(list)
    for r in rows("data/phase7/g/train.jsonl"):
        buckets[category(r)].append(r)
    chosen, tokens = [], {}
    budgets = dict(calendar=450_000, procedure=350_000, contrast=100_000, other=50_000, p7_graph=250_000)
    for cat, target in budgets.items():
        rng.shuffle(buckets[cat])
        bins = defaultdict(list)
        for r in buckets[cat]:
            if cat == "calendar":
                stratum = (r["family"], r["meta"]["expected"]["status"])
            elif cat == "p7_graph":
                stratum = (r["source"], r["op"])
            else:
                stratum = r.get("meta", {}).get("intent", r["task"])
            bins[stratum].append(r)
        keys = list(bins)
        rng.shuffle(keys)
        n = 0
        while n < target:
            assert keys, cat
            for k in keys:
                item = pack(bins[k].pop())
                chosen.append(item)
                n += item[1]
                if n >= target:
                    break
            keys = [k for k in keys if bins[k]]
        tokens[cat] = n
    return chosen, tokens

def targets(p, b, op):
    """Return the question target, expected type, gold answer and worked result."""
    d = p["day"][0]
    if op == "stage":
        start = R._STAGE_START[d]
        sign = 1 if R.is_yang(d) else -1
        offset = sign * (BRANCHES.index(b) - BRANCHES.index(start)) % 12
        ans = R.life_stage(d, b)
        assert ans == EightChar.CHANG_SHENG[(EightChar._EightChar__CHANG_SHENG_OFFSET[d] + sign*BRANCHES.index(b)) % 12]
        route = "顺" if sign == 1 else "逆"
        body = f"日干为{d}，其长生在{start}，按{route}行排十二宫。\n2. 以长生为第0位，{start}到{b}{route}数{offset}步，第{offset}位为{ans}。"
        return "日主在此支的十二长生阶段", "阶段名", ans, body
    if op == "god":
        stem = R.HIDDEN_STEMS[b][0]
        ans = R.ten_god(d, stem)
        assert stem == LunarUtil.ZHI_HIDE_GAN[b][0] and ans == LunarUtil.SHI_SHEN[d+stem]
        body = f"{b}支藏{'、'.join(R.HIDDEN_STEMS[b])}，取本气{stem}为对照干。\n2. 日主是{d}，{R.ten_god_reason(d,stem)}。"
        return "此支本气藏干相对日主的十神", "十神名", ans, body
    if op in ("positions", "count"):
        elem = R.BRANCH_ELEMENT[b]
        selected = [k for k in PILLARS if R.BRANCH_ELEMENT[p[k][1]] == elem]
        assert selected == [k for k in PILLARS if LunarUtil.WU_XING_ZHI[p[k][1]] == LunarUtil.WU_XING_ZHI[b]]
        terms = "、".join(f"{R.POSITION_ZH[k]}支{p[k][1]}属{R.BRANCH_ELEMENT[p[k][1]]}" for k in PILLARS)
        body = f"{b}的地支五行属{elem}。\n2. 逐柱核对：{terms}。"
        if op == "positions":
            return "原局中地支与此支五行相同的柱名", "年月日时顺序的柱名列表", [R.PILLAR_ZH[k] for k in selected], body
        return "原局四个地支与此支五行相同的个数", "整数", len(selected), body
    hidden = R.HIDDEN_STEMS[b]
    assert list(hidden) == list(LunarUtil.ZHI_HIDE_GAN[b])
    body = f"这一步使用指定地支{b}。\n2. 按藏干表核对，{b}藏{'、'.join(hidden)}，本气排在首位，为{hidden[0]}。"
    if op == "main":
        return "此支的本气藏干", "一个天干", hidden[0], body
    return "此支的全部藏干", "本气在前的天干列表", list(hidden), body

def atomic(p, b, op, variant, ident):
    term, hint, ans, body = targets(p, b, op)
    q = [
        f"这一步已取到地支{b}。请据此求{term}。",
        f"下步计算用{b}支，{term}是什么？",
        f"把地支{b}看作本题所用的支，求{term}。",
        f"查完前一项后得到{b}。请继续：{term}是什么？",
        f"这里要用的地支是{b}，请算出{term}。",
        f"已知目标地支为{b}。现在只计算{term}。"
    ][variant]
    user = head_line(p, None) + q + f"\n结论JSON 需含 answer（{hint}）。"
    reply = "1. " + body + "\n结论JSON: " + compact({"answer": ans})
    return dict(id=ident, family="p8_atomic", op=op, branch=b, chart=p, variant=variant,
                messages=[dict(role="system", content=SHORT_SYSTEM), dict(role="user", content=user),
                          dict(role="assistant", content=reply)])

def main():
    assert sha(PARENT) == PARENT_SHA
    rng = random.Random(SEED)
    cfg = yaml.safe_load(Path("configs/phase7/g.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    ds = PromptCompletionDataset([], tok)
    def pack(row):
        ids, start = ds.process(row)
        assert len(ids) <= CAP
        # MLX-LM compares positions >= prompt offset; the historical trainer logs
        # one more effective item than len(ids)-offset per untruncated example.
        return row, len(ids), len(ids)-start+1
    common, common_tokens = common_replay(rng, pack)
    excluded = historical_keys()
    for path in Path("data/phase7").rglob("*.jsonl"):
        excluded.update(k for r in rows(path) if (k := chart_key(r)))
    existing = len(excluded)
    schedule = defaultdict(list)
    for day in STEMS:
        for branch in BRANCHES:
            for rep in range(8):
                for i, op in enumerate(("stage","god")):
                    control = ("positions","count","hidden","main")[(rep+i) % 4]
                    schedule[day].append((branch,op,control,rep%6))
        rng.shuffle(schedule[day])
    arm_a, arm_b, train_keys = [], [], set()
    matches = Counter()
    while any(schedule.values()):
        p = chart(rng, CHART_YEARS, excluded)
        day = p["day"][0]
        if not schedule[day]:
            continue
        branch, op, control, version = schedule[day].pop()
        ident = f"p8-isolated-{len(arm_a):04d}"
        arm_a.append(pack(atomic(p,branch,op,version,ident)))
        arm_b.append(pack(atomic(p,branch,control,version,ident)))
        train_keys.add("".join(p[k] for k in PILLARS))
        matches["same_as_day_branch"] += branch == p["day"][1]
        matches["present_in_chart"] += any(branch == p[k][1] for k in PILLARS)
    assert len(train_keys) == len(arm_a) == 1920
    # At most three common rows are dropped to finish with complete accumulated batches.
    while (len(common) + len(arm_a)) % 4:
        i = next(i for i, r in enumerate(common) if category(r[0]) == "p7_graph")
        item = common.pop(i)
        common_tokens["p7_graph"] -= item[1]
    rng_order = list(range(len(common)+len(arm_a)))
    rng.shuffle(rng_order)
    arms = {"a": common+arm_a, "b": common+arm_b}
    shared_length = [round((x[1]+y[1])/2) for x,y in zip(arms["a"],arms["b"],strict=True)]
    length_order = sorted(rng_order,key=shared_length.__getitem__)
    stats = {}
    for arm, recs in arms.items():
        sizes = [recs[i][1] for i in length_order]
        stats[arm] = dict(rows=len(recs), updates=len(recs)//4, sequence=sum(sizes),
                          supervised_logged=sum(r[2] for r in recs),
                          padded=sum(2*min(CAP,1+32*math.ceil(max(sizes[i:i+2])/32)) for i in range(0,len(sizes),2)),
                          atomic_counts=dict(Counter(r[0]["op"] for r in recs[len(common):])))
    print("Preflight", stats, flush=True)
    for metric in ("updates","sequence","supervised_logged","padded"):
        vals = [x[metric] for x in stats.values()]
        assert max(vals)/min(vals) <= 1.03,(metric,vals)
    OUT.mkdir(exist_ok=True, parents=True)
    valid = list(rows("data/phase7/g/valid.jsonl"))
    hashes = {}
    for arm, recs in arms.items():
        folder = OUT/arm
        folder.mkdir(exist_ok=False)
        write_rows(folder/"train.jsonl",(dict(recs[i][0],phase8_sort_length=shared_length[i]) for i in rng_order))
        write_rows(folder/"valid.jsonl",valid)
        run = f"bazireason-phase8-{arm}"
        arm_cfg = dict(cfg,run_name=run,seed=SEED,init_adapter=str(PARENT),adapter_dir=f"artifacts/adapters/{run}",
                       log_dir=f"results/phase8/training/{arm}",learning_rate=5e-5,end_learning_rate=5e-6,
                       data=dict(train=str(folder/"train.jsonl"),valid=str(folder/"valid.jsonl"),valid_rows=len(valid)),
                       steps_per_eval=len(recs)//2,steps_per_save=len(recs)//2)
        config_file = Path("configs/phase8")/(arm+".yaml")
        config_file.parent.mkdir(exist_ok=True)
        with config_file.open("x") as f:
            yaml.safe_dump(arm_cfg,f,allow_unicode=True,sort_keys=False)
        hashes[arm] = dict(train=sha(folder/"train.jsonl"),valid=sha(folder/"valid.jsonl"),config=sha(config_file))
    keyfile = OUT/"train_chart_keys.json"
    with keyfile.open("x") as f:
        json.dump(sorted(train_keys),f,ensure_ascii=False)
    with (OUT/"training_manifest.json").open("x") as f:
        json.dump(dict(seed=SEED,parent=str(PARENT),parent_sha256=PARENT_SHA,hashes=hashes,stats=stats,
                       common_rows=len(common),common_tokens=common_tokens,fresh_chart_years=CHART_YEARS,
                       fresh_chart_keys_sha256=sha(keyfile),legacy_exclusion_keys=existing,
                       fresh_chart_count=len(train_keys),branch_overlap=matches,
                       training_script_sha256=sha(Path(__file__)),
                       matched_unit="identical chart, branch, template version and classroom format",
                       stem_branch_reps_per_target_consumer=8),f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()
