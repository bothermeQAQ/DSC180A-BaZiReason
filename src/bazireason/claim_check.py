"""Check fixed-form chart and rule claims in free text, independently of any language model.

Every checker reads a statement of a fixed shape ("午未六合", "月支巳藏丙庚戊", "旬空申酉",
"2027年为丁未年", "按月令透干取格为偏印格", "木1、火2、土3、金1、水1"…) and recomputes it with
the calendar-derived pillars and the project's rule code. A statement the code cannot
decide is not counted. The chatbot shows each result next to the reply, and Phase 4
workflow W1 uses the same code. Nothing here judges interpretation or fortune.
"""

import re

from bazireason import deep_rules as D
from bazireason import rules as R
from bazireason.baziqa import audit_details, audit_explanation

S = "[甲乙丙丁戊己庚辛壬癸]"
B = "[子丑寅卯辰巳午未申酉戌亥]"
NEG = re.compile(r"(?:不|无|没有|非|未|并非|既不)[^，。；\n]{0,3}$")
RELATION_WORDS = {"六合": "六合", "相冲": "冲", "冲": "冲", "相刑": "刑", "刑": "刑", "自刑": "自刑",
                  "相害": "害", "害": "害", "相破": "破", "破": "破", "半合": "半合"}  # fmt: skip
BRANCH_REL = re.compile(rf"({B})({B})(六合|相冲|相刑|自刑|相害|相破|半合)")
BRANCH_REL_WITH = re.compile(rf"({B})与({B})(六合|相冲|相刑|相害|相破|半合)")
BRANCH_REL_VERB = re.compile(rf"({B})(冲|刑|害|破)({B})")
STEM_REL = re.compile(rf"({S})({S})(相合|五合|相冲)")
HIDDEN = re.compile(rf"({B})(?:支)?(?:中)?藏(?:干)?(?:[：:]\s*)?((?:{S}[、，, ]?){{1,3}})")
COUNTS = re.compile(r"木\s*(\d)\D{0,3}火\s*(\d)\D{0,3}土\s*(\d)\D{0,3}金\s*(\d)\D{0,3}水\s*(\d)")
XUN = re.compile(rf"(?:旬空|空亡)(?:在|为|是|落在|[：:])?\s*({B})[、，, 和]?({B})")
STAGE = re.compile(
    rf"({S})(?:在|临|于)({B})(?:支)?(?:为|是|处于|属|处)?(长生|沐浴|冠带|临官|帝旺|衰|病|死|墓|绝|胎|养)"
)
STAR = re.compile(
    rf"(禄|阳刃|天乙贵人|驿马|桃花|华盖)(?:在|落在|为|位于)((?:{B}[、，, 和]?){{1,3}})"
)
STRENGTH = re.compile(r"(?:记为|计为|属于|判为|算作)(偏强|偏弱)")
# A pattern stated as this chart's pattern ("记作建禄格", "取格为偏印格"); names inside an
# explanation of the naming rule ("劫财在阳干记阳刃格") are not claims about the chart.
PATTERN = re.compile(
    r"(?:记作|记为|定为|取为|判为|属于|属|为|是|成)"
    r"(正官|七杀|偏官|正财|偏财|食神|伤官|正印|偏印|枭神|建禄|阳刃|月刃|月劫)格"
)
PATTERN_SYNONYMS = {"偏官": "七杀", "枭神": "偏印", "月刃": "阳刃"}
SPOUSE_RULE = re.compile(r"([男女])命?以(?:正|偏|七|财|官)")
TITLE = re.compile(r"《([^》]{1,12})》")
LOOP = re.compile(r"(.{20,}?)\1\1", re.S)
CARD_WORKS = {"渊海子平", "三命通会", "滴天髓", "穷通宝鉴", "子平真诠"}
UNVERIFIABLE = re.compile(
    r"(难以成功|注定|必定|必然会|一定会|肯定会|绝对会|克夫|克妻|克父|克母|短命|必有大|必遭|大富大贵)"
)


def _negated(text, start):
    return bool(NEG.search(text[max(0, start - 6) : start]))


def _result(kind, said, ok, engine):
    return {"kind": kind, "said": said, "ok": ok, "engine": engine}


def check_claims(text: str, pillars: dict, gender: str | None = None) -> list:
    """Every checkable claim in ``text``: kind, the words, ok (True/False) and the engine value."""
    if not isinstance(text, str) or not pillars:
        return []
    out = []
    # Pillars, day master, pillar of a year, Ten God of a named stem (Phase 2 checks).
    wrong = audit_details(text, pillars)
    for kind, ok in audit_explanation(text, pillars):
        out.append(_result(kind, "", ok, None))
    out += [_result("detail", w, False, None) for w in wrong]
    for regex in (BRANCH_REL, BRANCH_REL_WITH):
        for m in regex.finditer(text):
            if _negated(text, m.start()):
                continue
            a, b, word = m.groups()
            kinds = [k.split(":")[0] for k in R.branch_pair_relations(a, b)]
            want = RELATION_WORDS[word]
            out.append(_result("relation", m.group(0), want in kinds, kinds or ["无"]))
    for m in BRANCH_REL_VERB.finditer(text):
        a, word, b = m.groups()
        if _negated(text, m.start()) or text[m.end() : m.end() + 2] in ("太岁",):
            continue
        kinds = [k.split(":")[0] for k in R.branch_pair_relations(a, b)]
        out.append(_result("relation", m.group(0), RELATION_WORDS[word] in kinds, kinds or ["无"]))
    for m in STEM_REL.finditer(text):
        if _negated(text, m.start()):
            continue
        a, b, word = m.groups()
        kinds = [k.split(":")[0] for k in R.stem_pair_relations(a, b)]
        want = "干冲" if word == "相冲" else "干合"
        out.append(_result("stem_relation", m.group(0), want in kinds, kinds or ["无"]))
    for m in HIDDEN.finditer(text):
        # Wrong only if a named stem is not hidden in the branch: replies often name just
        # the main qi, or follow each stem with its Ten God, which the pattern stops at.
        branch, stems = m.group(1), re.findall(S, m.group(2))
        truth = list(R.HIDDEN_STEMS[branch])
        out.append(_result("hidden", m.group(0), set(stems) <= set(truth), "".join(truth)))
    for m in COUNTS.finditer(text):
        said = dict(zip(R.ELEMENTS, map(int, m.groups()), strict=True))
        truth = R.element_counts(pillars)
        out.append(_result("element_count", m.group(0), said == truth, truth))
    for m in XUN.finditer(text):
        truth = R.xun_kong(pillars["day"])
        out.append(_result("xun_kong", m.group(0), set(m.groups()) == set(truth), truth))
    for m in STAGE.finditer(text):
        stem, branch, stage = m.groups()
        truth = R.life_stage(stem, branch)
        out.append(_result("life_stage", m.group(0), stage == truth, truth))
    for m in STAR.finditer(text):
        star, said = m.group(1), set(re.findall(B, m.group(2)))
        options = {D.star_branches(star, pillars, basis) for basis in ("day", "year")}
        ok = any(set(o) == said for o in options if o)
        out.append(
            _result("star", m.group(0), ok, " 或 ".join(sorted(o for o in options if o)) or "无")
        )
    for m in STRENGTH.finditer(text):
        truth = R.strength_indicators(pillars)["summary"]
        out.append(_result("strength", m.group(0), m.group(1) == truth, truth))
    claims = {PATTERN_SYNONYMS.get(c, c) + "格" for c in PATTERN.findall(text)}
    if claims:
        allowed = {D.revealed_pattern(pillars)["pattern"], R.month_pattern(pillars)}
        for claim in sorted(claims):
            out.append(_result("pattern", claim, claim in allowed, " 或 ".join(sorted(allowed))))
    genders = set(SPOUSE_RULE.findall(text))
    if len(genders) == 1:
        used = genders.pop()
        entered = {"male": "男", "female": "女"}.get(gender)
        out.append(
            _result("spouse_star", f"按{used}命取配偶星", used == entered, entered or "未填性别")
        )
    for title in TITLE.findall(text):
        out.append(_result("citation", f"《{title}》", title in CARD_WORKS, "规则卡引用的著作"))
    claim = UNVERIFIABLE.search(text)
    if claim and "不能" not in text[max(0, claim.start() - 12) : claim.start()]:
        out.append(
            _result("unverifiable", claim.group(1), False, "无法核验的断语，没有可验证的依据")
        )
    if LOOP.search(text):
        out.append(_result("repetition", "同一段文字连续重复三次", False, "回答出现循环"))
    return out


def problems(results: list) -> list:
    """Readable lines for the claims that failed."""
    lines = []
    for r in results:
        if r["ok"]:
            continue
        if r["kind"] == "detail":
            lines.append(r["said"])
        elif r["kind"] in ("ten_god", "pillar", "day_master", "flow_year"):
            continue  # covered by the detail lines
        elif r["kind"] == "citation":
            lines.append(f"回答引用了{r['said']}，规则卡没有引用这部书，引文无法核对。")
        elif r["kind"] in ("unverifiable", "repetition"):
            lines.append(f"{r['said']}：{r['engine']}。")
        elif r["kind"] == "spouse_star" and r["engine"] == "未填性别":
            lines.append(
                f"没有填写性别，回答却{r['said']}；男命看财、女命看官杀，性别不同结论不同。"
            )
        elif r["kind"] == "spouse_star":
            lines.append(f"填写的是{r['engine']}命，回答却{r['said']}。")
        else:
            lines.append(f"“{r['said']}”与规则代码不符，规则代码给出：{_show(r['engine'])}。")
    return list(dict.fromkeys(lines))


def _show(value):
    if isinstance(value, dict):
        return "、".join(f"{k}{v}" for k, v in value.items())
    if isinstance(value, list):
        return "、".join(value)
    return str(value)
