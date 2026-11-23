"""Render the verified comparisons for the finite Phase 9 campaign."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

O = Path("results/phase9")
IDS = ("g","a","s","x","p")
HEAD = ["G · parent","A · isolated","S · sparse","X · crossed","P · planned"]


def read(p):
    return json.loads(p.read_text())


def table(cols, lines):
    return "\n".join("| "+" | ".join(map(str,row))+" |" for row in [cols,["---"]*len(cols),*lines])


def frac(d):
    return f"{d['right']}/{d['n']}"


def ci(v):
    return "["+", ".join(f"{x:+.1f}" for x in v)+"]"


def scored(label, counts):
    return [label,*[frac(counts[m]) for m in IDS]]


def main():
    comparison = read(O/"transfer_comparison.json")
    preserve = read(O/"preservation.json")
    err = read(O/"error_audit.json")["counts"]
    tr = comparison["endpoints"]
    primary = tr["novel"]
    ret = preserve["retention"]
    bridge = preserve["bridge"]
    checks = {}
    for m in ("s","x","p"):
        checks[m] = {
            "Primary above G": primary["counts"][m]["right"] > primary["counts"]["g"]["right"],
            "Unaided bridge above G": bridge["counts"][m]["right"] > bridge["counts"]["g"]["right"],
            **{f"{k} vs {ref.upper()}":ret[k]["paired"][f"{m}-{ref}"]["pp"]>=margin
               for k, margin in [("rules",-1.5),("calendar",-2)] for ref in ("g","u8")}
        }
    (O/"selection.json").write_text(json.dumps(checks,indent=2)+"\n")
    train = {m:read(O/f"training/{m}/training_summary.json") for m in ("s","x","p")}
    spec = read(Path("data/phase9/training_manifest.json"))
    winners = [m.upper() for m,v in primary["counts"].items() if v["right"]==max(d["right"] for d in primary["counts"].values())]
    dx = primary["paired"]["x-s"]
    dp = primary["paired"]["p-x"]
    j = comparison["joint"]
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    start = datetime(2026,11,22,12,37,tzinfo=now.tzinfo)
    doc = ["# Phase 9 · Can a 4B model carry the right rule through a complete question?",
           f"The highest fresh reference-assisted primary score belonged to **{' and '.join(winners)}**. Crossing source rules with more downstream operations changed accuracy by **{dx['pp']:+.1f} percentage points** relative to the sparse training control (paired 95% chart interval {ci(dx['chart_ci'])}). Adding an explicit plan on the identical crossed questions changed accuracy by **{dp['pp']:+.1f} points** ({ci(dp['chart_ci'])}). The predeclared X-versus-S screen **{'passed' if comparison['primary_screen']['passes'] else 'did not pass'}**.",
           "## What Phase 8 established",
           "On familiar wording, all 54 wrong G replies named the correct first branch and returned a wrong value from the requested answer class. Under new wording G missed 89 questions. In 57 of those replies the first written branch was recognizably correct, in 15 it was wrong, and in 17 the extraction rule could not classify it. Only five final values exactly matched the result of another tested operation on the correct branch. A’s isolated practice raised single-step accuracy to 73/80, yet on 52 of those 80 charts A answered the isolated question correctly and the complete one incorrectly. These observations support several failure modes, including selecting an operation, binding it to the right input, performing the lookup, and honoring the output contract. [Independent archive diagnosis and methods review](phase9_diagnosis.md)",
           "G’s original worked graph questions associated different first rules with different sets of allowable next steps. Ten-God examples occurred under two star rules, reaching only eight branches. This made broader training coverage a testable alternative to explicit planning. Controlled studies have found that example diversity can improve compositional generalization and that supervision of intermediate calculations can help algorithmic tasks. Those results motivated the comparisons without supplying an expected BaZi effect size. [Zhou, Jiang & Bansal](https://aclanthology.org/2023.emnlp-main.898/) · [Nye et al.](https://arxiv.org/abs/2112.00114)",
           "## Controlled training",
           "G is the Phase 7 conventional worked-answer adapter. A is Phase 8’s isolated-practice model. S, X and P each continued directly from G with seed 910217. All three used the same 3,036 replay examples and 2,016 new worked questions on 504 charts. Two questions shared a chart and branch but requested different operations, and each chart supplied two different source branches. Every arm saw varied instruction wording, a common answer schema and complete downstream arithmetic or table work.",
           "S retained the original 22 source–operation combinations. X expanded to 51 through operation swaps within matched day-stem and derived-branch cells. Its source and operation marginals and its operation-by-day-stem-by-branch counts matched S exactly. In the new questions, knowing the source yielded 1.354 bits of information about the requested operation for S and 0.143 bits for X. P received X’s exact questions, labels and calculations with an explicit plan and operand bindings in its worked targets. The X−S contrast tests the distribution of training combinations. P−X tests the response format and supervision, including its modest token cost.",
           table(["Arm","Rows","Updates","Sequence tokens","Supervised tokens","Padded tokens (planned)","Minutes","Peak GB","Final loss"],
                 [[m.upper(),s["train_rows"],s["optimizer_updates"],f"{s['sequence_tokens_processed']:,}",f"{s['loss_tokens_total']:,}",f"{spec['stats'][m]['padded']:,}",f"{s['duration_seconds']/60:.1f}",f"{s['peak_memory_gb']:.2f}",f"{s['final_train_loss']:.4f}"] for m,s in train.items()]),
           "All arms used the fixed final checkpoint. The equalized operand exposure preserved a limitation across arms. Ten-God targets covered only eight derived branches. Four formerly evaluated edges were deliberately excluded from every arm. Exposure differences on the other 29 added edges are part of the X/P intervention. Prior published scores on those edges can no longer serve as unseen-edge tests for X or P. [Training manifest](../data/phase9/training_manifest.json) · [Readback audit](../results/phase9/training_audit.json)",
           "## Transfer results",
           "The primary test used a compact table defining four previously untrained source lookups drawn from the 三合 and 三会 branch groups. For each chart the model had to read a named year/day branch, find the requested member of its group, then use that branch for a life stage, Ten God, element or hidden-stem answer. Each of 16 source–operation combinations contributed 12 questions. Two requests used the same chart and source. The unaided bridge panel used four familiar source–operation combinations excluded from all three arms, independently written questions and 160 fresh charts. The three-step panel used only familiar rule primitives and no source table.",
           table(["Exact answers",*HEAD],
                 [scored(label,tr[key]["counts"]) for label,key in [
                      ("Reference-assisted, new wording · primary","novel"),("Reference-assisted, canonical wording","familiar"),
                      ("Day-dependent operations · primary","day_dependent"),("Other operations · primary","other"),
                      ("Both requests on a chart correct","both_operations_correct"),("New sources, without the reference","closed"),
                      ("Source lookup in isolation","producer"),("Correct branch supplied, downstream only","consumer"),
                      ("Identify the requested operation, four choices","operation"),("Unaided, three-step questions","deep")]]
                 +[scored("Unaided bridge, familiar sources",bridge["counts"]),scored("Fresh familiar rules",ret["rules"]["counts"]),
                   scored("Fresh calendar questions",ret["calendar"]["counts"])]),
           table(["Contrast","Primary Δ pp","95% chart interval","95% producer + chart interval","Leave-one-producer-out range","Wins / losses on same questions"],
                 [[key.upper(),f"{v['pp']:+.1f}",ci(v["chart_ci"]),ci(v["family_and_chart_ci"]),ci(v["leave_one_producer_out_range"]),f"{v['left_only']} / {v['right_only']}"] for key in ["x-s","x-g","p-x","p-g","s-g","a-g"] for v in [primary["paired"][key]]]),
           "The primary intervals use 10,000 paired bootstrap draws of complete charts, stratified by producer and requested operation pair. The broader sensitivity interval also resamples producers. With four producers and a single training seed, generalizing these intervals to new rule families or independent retraining would overstate the evidence.",
           "## Where answers break down",
           table(["Matched-question check",*HEAD],[
               ["With reference, on the exact 96 closed-book questions",*[f"{j[m]['reference']['reference']}/96" for m in IDS]],
               ["Complete answer right given both isolated answers right",*[frac(j[m]['full_given_both_prereqs']) for m in IDS]],
               ["Both isolated answers right and complete answer wrong",*[j[m]['prerequisites']['both_prereqs_but_full_wrong'] for m in IDS]],
               ["Primary when the natal-branch shortcut gives a wrong answer",*[f"{err[m].get('novel_right_when_natal_answer_differs',0)}/{err[m]['novel_natal_answer_differs']}" for m in IDS]],
               ["Three-step when dropping either or both source steps gives a wrong answer",*[f"{err[m].get('deep_all_shortcuts_wrong_right',0)}/{err[m]['deep_all_shortcuts_wrong_n']}" for m in IDS]]]),
           "Using the natal branch directly solved 71/192 primary questions. The shortcut rows above restrict the comparison to cases where correct execution must produce a different final value. The operation-choice probe adds a four-option letter response, a format absent from the graph curriculum. Its score reflects both instruction interpretation and compliance with that format. Conditional rates use each model’s own prerequisite successes, so their denominators vary.",
           table(["Primary operation",*HEAD],[scored(k,v) for k,v in primary["by_op"].items()]),
           table(["Unaided bridge edge",*HEAD],[scored(k,v) for k,v in preserve["bridge_by_family"].items()]),
           "## Preservation and selection",
           f"On the fresh retention panels, U8 scored {frac(ret['rules']['counts']['u8'])} on familiar rules and {frac(ret['calendar']['counts']['u8'])} on calendar questions. The selection rule requires primary and unaided bridge point improvements over G, plus rule and calendar losses no larger than 1.5 and 2 points respectively against both G and U8. Paired retention intervals resample whole events within task category.",
           table(["Candidate","Primary above G","Bridge above G","Rule Δ vs G / U8 (pp)","Calendar Δ vs G / U8 (pp)","All point criteria"],
                 [[m.upper(),checks[m]["Primary above G"],checks[m]["Unaided bridge above G"],*[ " / ".join(f"{ret[k]['paired'][m+'-'+ref]['pp']:+.2f}" for ref in ("g","u8")) for k in ("rules","calendar")],all(checks[m].values())] for m in ("s","x","p")]),
           table(["Calendar behavior","U8",*HEAD],[[label,*[f"{ret['calendar_behavior'][m][flag]}/{ret['calendar_behavior'][m][den]}" for m in ("u8",*IDS)]]
                 for label,flag,den in [("False refusal on answerable","false_refusal","answerable"),("Definite chart on nonanswerable","definitive_on_nonanswer","nonanswerable")]]),
           "## Boundaries and verification",
           "Separate agents authored the training prompts and evaluation wording. Labels were cross-checked using independent arithmetic and table lookups. The input audits covered historical and realized training charts and retention dates, and every saved response was re-scored and matched to its input and adapter hashes. All data are synthetic, the system prompt and output format are shared, and the reference-assisted test includes a rule table. The unaided panels measure different forms of transfer. No frontier model was evaluated, so these results cannot quantify a gap to a frontier system.",
           f"Started at 12:37 Pacific on November 22. The final report was rendered at {now:%H:%M} Pacific, {(now-start).total_seconds()/3600:.2f} elapsed hours. The three training jobs together used {sum(t['duration_seconds'] for t in train.values())/3600:.2f} hours. All 10,400 planned replies were collected (five adapters × 1,920 questions, plus 800 U8 retention replies).",
           "[Frozen protocol](phase9_protocol.md) · [Transfer results and paired intervals](../results/phase9/transfer_comparison.json) · [Bridge and retention results](../results/phase9/preservation.json) · [Error examples](../results/phase9/error_audit.json) · [Selection checks](../results/phase9/selection.json) · [Raw replies and training records](../results/phase9)"]
    assert [primary["counts"][m]["right"] for m in IDS] == [37,22,39,36,43]
    doc.insert(2, "P is the leading research checkpoint for unaided combinations of familiar rules. It improved from G’s 56/160 to 102/160 on the four held-out familiar-source pairs, a gain of 28.8 points (paired chart interval +20.0 to +37.5). On the primary test with newly supplied rules, P solved only 43/192 (22.4%), compared with G’s 37/192 (19.3%). That six-answer difference remains uncertain. The evidence supports a narrower gain in applying familiar rules across combinations. [P adapter and configuration](../artifacts/adapters/bazireason-phase9-p)")
    ix = doc.index("## Where answers break down")
    doc[ix:ix] = [
        "![Paired accuracy changes across the three evaluation panels](../results/phase9/paired_effects.png)",
        "On the unaided familiar-source panel, the sparse continuation alone improved over G by 18.1 points (95% chart interval +9.4 to +26.9). Expanding combinations added 6.9 points to S (−1.3 to +15.0), and adding plans to X added 3.8 (−4.4 to +11.3). All four bridge families improved from G to P. Since all three continuations shared contrastive questions, varied wording, worked calculations and replay, the gain over G cannot be assigned entirely to planning or to any one shared change.",
        "On unaided three-step questions P scored 26/96, compared with X’s 10/96 and G’s 14/96. The secondary P−X contrast was +16.7 points (95% chart interval +7.3 to +25.0). P still answered just 18/71 correctly when deleting either or both source steps changed the answer. This panel tests four paths and several secondary contrasts were examined. It merits independent replication."
    ]
    ix = doc.index("## Preservation and selection")
    doc[ix:ix] = [
        "The new-source lookup is itself a bottleneck. P solved only 19/96 isolated source questions with the printed reference, including 0/24 requests for the peak member of a three-harmony group. In p9-transfer-1-05-producer-branch, P found the correct group 亥卯未 but answered 衰 (a stage) instead of selecting 卯, its middle branch. Removing the reference from the matched 96 primary questions reduced P from 26 to 15 correct. The table helped on balance, even though P often misread the requested position.",
        "Giving P the correct branch raised downstream accuracy to 100/192. Both isolated prerequisites were right for only 20 complete questions, and P answered 10 of those correctly. On p9-transfer-0-01-novel-stage it derived 巳 and then began calculating luck-cycle direction, while its isolated downstream reply correctly returned 绝. Across all 48 primary life-stage questions P got two right and returned another life stage on 34. Both selection and execution remain plausible failure modes. Declared plans do not ensure either."
    ]
    ix = doc.index("## Boundaries and verification")
    doc[ix:ix] = [
        "P met the point retention margins. Relative to G, familiar-rule accuracy fell 1.0 point (paired 95% interval −2.0 to −0.25) and calendar accuracy fell 1.25 points (−3.25 to +0.5). Both intervals extend past the respective allowable loss. P also issued nine more definite answers than G on 151 calendar requests without a determinate chart. Keep G and U8 as reference checkpoints and retain S as the simpler scientific control.",
        "## Recommended next step",
        "Before scaling training, test reference reading on a small fresh panel. Hold each chart, table and requested operation fixed, and compare the original compact table with separately labeled rows and columns, then with the correct source branch supplied. Include counterfactual tables in which a row or column changes and require the answer to change accordingly. Pair each case with a direct lookup and a complete question. This distinguishes trouble finding the referenced cell from trouble applying a correctly selected value inside a chain.",
        "If explicit table layout or a supplied branch rescues complete answers, test a short curriculum that gradually removes that support while using unfamiliar table names and independently authored instructions. If P instead reproduces its familiar-rule gains across a second seed and fresh families, it would justify a larger study of explicit planning. A blinded stronger-model reference on the same inputs would quantify the remaining frontier gap. None of these follow-on experiments has been launched."
    ]
    output = Path("docs/phase9_results.md")
    output.write_text("\n\n".join(doc)+"\n")


if __name__ == "__main__":
    main()
