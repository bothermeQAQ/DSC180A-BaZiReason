"""Build a new fixed retention panel with complete calendar event groups."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

from bazireason import train_data as TD, train_data6 as T6
from bazireason.intent import plan_for
from bazireason.p5_eval import make_item
from bazireason.prompts6 import apply
from bazireason.workflow import prompt_messages
from graphs import chart, rows, write_rows
from audit_retention import input_files, identities

OUT = Path("data/phase9")


def build():
    rng = random.Random(910037)
    excluded, prompts, dates = set(), set(), set()
    for path in input_files():
        for record in rows(path):
            ids = identities(record)
            excluded.update(ids["chart"])
            prompts.update(sha256(q.encode()).digest() for q in ids["prompt"])
            dates.update(ids["calendar_date"])
    ret = []
    for head in T6.WEIGHTS:
        for i in range(20):
            p = chart(rng, sorted(TD.split_years()["valid"]), excluded)
            gender = rng.choice(["male", "female"])
            intent = T6.concrete_intent(head, p, rng)
            q = T6.template_question(intent, p, gender, rng)
            group = f"p9-rule-{head}-{i}"
            item = make_item(group, "p9", "template", p, gender, q, intent, plan_for(intent, p, gender), group)
            ret.append(dict(id=group, group=group, condition=head, track="rules", item=item,
                            messages=apply("p6u8", [prompt_messages(item)])[0], cap=1100))
    groups = defaultdict(list)
    for r in TD.generate_split("valid", 910043, scale=22, version=2):
        if r["task"] == "chart":
            groups[(r["family"], r["group_id"])].append(r)
    families = defaultdict(list)
    for (family, group), batch in groups.items():
        families[family].append((group, batch))
    for candidates in families.values():
        rng.shuffle(candidates)
    n = 0
    while n < 400 and any(families.values()):
        for family, candidates in sorted(families.items()):
            while candidates and n < 400:
                group, batch = candidates.pop()
                user_lists = [[prompt_messages(r)[-1]["content"],
                               apply("p6u8", [prompt_messages(r)])[0][-1]["content"]] for r in batch]
                users = {sha256(u.encode()).digest() for qs in user_lists for u in qs}
                days = {dt[:10] for r in batch if isinstance(dt := r["meta"]["inputs"].get("datetime"), str)}
                if days & dates or users & prompts or len({qs[-1] for qs in user_lists}) != len(batch):
                    continue
                dates.update(days)
                prompts.update(users)
                for r in batch:
                    ret.append(dict(id="p9-"+r["id"], group=f"p9-cal-{family}-{group}", condition=family,
                                    track="calendar", item=r, messages=apply("p6u8", [prompt_messages(r)])[0], cap=700))
                n += len(batch)
                break
    assert n >= 400 and len(ret) == len({r["id"] for r in ret})
    path = OUT / "retention.jsonl"
    write_rows(path, ret)
    manifest = dict(sha256=sha256(path.read_bytes()).hexdigest(), rules=len(ret)-n, calendar=n,
                    groups=len({r["group"] for r in ret}), calendar_seed=910043, rule_seed=910037)
    (OUT / "retention_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(manifest, flush=True)


if __name__ == "__main__":
    build()
