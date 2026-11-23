"""Render verified numeric tables for the Phase 7 one-seed comparison."""
import json
from pathlib import Path
from collections import Counter
from analyze import MODELS,OUT,DATA,load
from graphs import parse

def read(path):return json.loads(Path(path).read_text())
def pct(n,d):return f"{n}/{d} ({n/d*100:.1f}%)"
def ci(v):return "["+", ".join(f"{x:+.1f}" for x in v)+"]"
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|",
                      *["| "+" | ".join(map(str,r))+" |" for r in rows]])
def main():
    result=read(OUT/"comparison.json")
    end=result["endpoints"]
    diag=read(OUT/"diagnostic_summary.json")
    loc=read(OUT/"diagnostic_localization.json")
    manifest=read(DATA/"training_manifest.json")
    summaries={m:read(OUT/"training"/m/"training_summary.json") for m in ["r","g","t"]}
    doc=["# Phase 7 one-seed pilot results","**Development experiment · Qwen3-4B-Instruct-2507 · Apple M5, 32 GB**"]
    d=end["full"]["paired"]["t-r"]["pp"]
    dr=end["rules"]["paired"]["t-u8"]["pp"];dc=end["calendar"]["paired"]["t-u8"]["pp"]
    passed=d>=4 and dr>=-1.5 and dc>=-2
    doc+=["## Decision",f"Explicit-state training (T) scored **{pct(*end['full']['counts']['t'])}** on fresh held-out graph questions. Its advantage over ordinary replay (R) was **{d:+.1f} percentage points**. Relative to the original U8, familiar-rule accuracy changed by **{dr:+.1f}** and calendar accuracy by **{dc:+.1f}** points. The predeclared pilot screen **{'passed' if passed else 'did not pass'}**. The thresholds were +4 over replay and retention losses no greater than 1.5 and 2 points, respectively. This experiment used one training seed. The reserved graph families were not scored."]
    doc+=["## Prerequisite diagnostic",table(["U8 condition","Correct /120"],[(label,pct(diag["counts"][key],120)) for key,label in [
        ("full","Ordinary question"),("twin","Same chart, changed source"),("both_twins","Both members right"),
        ("producer","Isolated source lookup"),("consumer","Isolated downstream lookup"),
        ("both_primitives","Both isolated lookups"),("oracle","Correct intermediate supplied"),
        ("distractor","True distractor supplied"),("cue","Intent cue supplied")]]),
        f"Among {diag['counts']['miss']} ordinary misses, {loc['both_passed']} occurred with both isolated prerequisites correct. On the others, {loc['producer_only_failed']} missed only the producer, {loc['consumer_only_failed']} only the consumer, and {loc['both_failed']} both. Providing the branch fixed {diag['counts']['oracle_rescue']} initially wrong answers and broke {diag['counts']['oracle_break']} initially right ones. A secondary content check found {loc['schema_only_detected']} strict misses with a recoverable correct final value. An oracle cue can simplify routing as well as supply the missing value.",
        table(["Diagnostic family","Full","Producer","Consumer","With gold branch","Misses with a failed prerequisite"],[(f,*[pct(sc[k],sc['groups']) for k in ('full','producer','consumer','oracle')],f"{sc['miss_bad_pre']}/{sc['miss']}") for f,sc in sorted(diag["by_family"].items())])]
    doc+=["## What was trained","R continued on familiar procedures. G trained on new graph questions with numbered conventional derivations. T used exactly G’s questions, gold values and final-answer schema, exposing the source and branch consumed by each step in labeled fields. All arms started from the identical U8 adapter and shared 4,683 replay rows. The remaining 3,221 rows differed by arm. Each trained once with seed 710223. Rows were paired across G and T, with matched batches, fixed endpoints and the historical optimizer settings.",
          table(["Arm","Rows","Updates","Sequence tokens","Loss-bearing tokens (logged)","Padded tokens","Train hours","Peak GB","Final iteration"],[(m.upper(),manifest["stats"][m]["rows"],s["optimizer_updates"],f"{s['sequence_tokens_processed']:,}",f"{s['loss_tokens_total']:,}",f"{manifest['stats'][m]['padded']:,}",f"{s['duration_seconds']/3600:.2f}",f"{s['peak_memory_gb']:.2f}",s['selected_checkpoint_iteration']) for m,s in summaries.items()]),
          "The shared token allocations were 30% calendar, 38% familiar procedures, 8% contrast and 4% other replay. The experimental substitution was approximately 20%. The maximum pairwise discrepancies were 0.50% for sequence tokens, 0.84% for loss-bearing tokens and 0.63% for padded tokens. Optimizer updates matched exactly. Reference labels were obtained independently for each graph instance and compared to the project’s own rule code. Legacy calendar answers use the existing calendar scorer."]
    labels={"full":"Held-out graphs, primary","twin":"Held-out graphs, changed source","alternate":"Held-out graphs, new wording","seen0":"Natural draws, trained graph / familiar wording","seen2":"Natural draws, trained graph / new wording","unseen0":"Natural draws, unseen graph / familiar wording","unseen2":"Natural draws, unseen graph / new wording","rules":"Familiar rules","calendar":"Calendar"}
    doc+=["## Controlled comparison",table(["Panel","U8","R","G","T"],[(v,*[pct(*end[k]['counts'][m]) for m in MODELS]) for k,v in labels.items()]),
          table(["Held-out primary contrast","Difference (pp)","95% paired chart interval","95% family + chart interval","Leave-one-family-out range"],[(k.upper(),f"{v['pp']:+.1f}",ci(v["fixed_ci"]),ci(v["family_ci"]),ci(v["lofo"])) for k,v in end["full"]["paired"].items()]),
          "The primary panel weights seven families equally. Each contributes 20 independent chart groups. Intervals use 4,000 paired bootstrap draws. Family resampling shows sensitivity to which operations are represented. It does not quantify training-seed variation. G−R tests the substitution of the synthetic graph data. T−G isolates the target rendering within the tested graph set. U8 is a frozen reference. Secondary contrasts are descriptive."]
    fams=sorted(result["by_family"]["u8"])
    doc +=["## Source switches and error patterns",table(["Held-out edge","U8 full / both","R full / both","G full / both","T full / both"],[(f,*[f"{r['right']}/{r['n']} · {r['both']}/{r['n']}" for m in MODELS for r in [result['by_family'][m][f]]]) for f in fams]),
           table(["Baseline or diagnostic","Primary challenge panel","Natural unseen panel"],[(k,pct(result['baselines']['dev'][k],140),pct(result['baselines']['mix_unseen'][k],70)) for k in ["mode","natal"]]),
           "The mode baseline uses the most frequent training answer for each downstream operation. The natal baseline feeds the selected natal branch directly to that operation, skipping the requested producer. Challenge twins were sampled to have different gold values. The natural draw had no such filter. The alternate Chinese wording comes from a held-out template, with the same author as the training templates."]
    doc+=["## Retention",table(["Arm","Familiar-rule Δ vs U8, 95%","Calendar Δ vs U8, 95%","False refusals /150","Definite charts on nonanswerable /92"],[(m.upper(),*(f"{v['pp']:+.1f} {ci(v['row_ci'])}" for k in ("rules","calendar") for v in [end[k]["paired"][m+"-u8"]]),result["calendar"][m]["refusal"],result["calendar"][m]["unwarranted"]) for m in ("r","g","t")]),
           "The rule panel sampled 12 fresh questions per familiar task type (240 total). The calendar panel retained 242 rows, clustered by generator family and event/twin ID. Deltas use row accuracy with a paired group bootstrap. A small point loss may be within sampling error. Demonstrating the specified non-inferiority margins requires a larger independently held-out study."]
    doc+=["## Reproducibility and boundaries","The development evaluation used greedy generation, a single model and its adapter, the U8 system prompt, and no answer replacement. The primary scorer requires one labeled JSON answer, the specified type, and exact value. Historical Phase 6 scores were not used as Phase 7 outcomes. The seven development edges and seven diagnostic edges are disjoint from the 22 training edges. Twelve other edges remain reserved. This tests a finite family of symbolic transformations on synthetic charts. It does not establish transfer across arbitrary BaZi reasoning, independent human phrasing, distribution shift in deployment or another training seed.",
          "The protocol, data hashes, raw replies, fitted-adapter summaries, bootstrap results and family counts are available locally. [Protocol](phase7_pilot_protocol.md) · [Input audit](../results/phase7/input_audit.json) · [Training budgets and hashes](../data/phase7/training_manifest.json) · [Diagnostic counts and examples](../results/phase7/diagnostic_summary.json) · [Full comparison](../results/phase7/comparison.json) · [Raw replies and training records](../results/phase7)"]
    closeout_path = OUT / "closeout.json"
    if closeout_path.exists():
        closeout = read(closeout_path)
        doc.insert(doc.index("## Prerequisite diagnostic"),
                   "Training on worked graph questions accounted for most of the gain. Conventional derivations (G) gained 64.3 points over replay. Explicit source and value fields (T) added five correct answers over G on the primary panel, all in two families. G answered seven more alternate-wording questions and four more changed-source questions than T. Keep U8 as the reference and carry G forward as the simpler research candidate. The evidence for preferring T depends on the panel.")
        doc.insert(doc.index("## What was trained"),
                   "The diagnostic triggered the prerequisite review. On 101 of 103 ordinary misses, U8 also failed an isolated lookup for the same chart. This prevents attribution to a failure to combine two already-mastered steps. The new prompts also differed from the familiar training tasks, so performance here cannot distinguish incomplete rule knowledge from difficulty applying a known rule to a new instruction. After inspecting the replies, we kept the three approved arms and their frozen evaluation intact to test end-to-end transfer. [Prerequisite decision](../results/phase7/prerequisite_review.json)")
        i = doc.index("## Retention")
        trace = closeout["trace_on_wrong_primary"]
        doc[i:i] = [
            "G and T solved both members of every pair in five of the seven families. Across all seven, G solved both on 102/140 charts and T on 104/140, compared with 2/140 for R and 5/140 for U8. The difficult families asked for the day master’s life stage at the harmony partner branch (六合→stage) and the Ten God of the harm partner’s main hidden stem (害→god). Both depend on the day stem as well as the derived branch.",
            f"On the ordinary primary misses, the first written step named the correct derived branch on {trace['g']['correct_first_branch']}/{trace['g']['misses']} G replies and {trace['t']['correct_first_branch']}/{trace['t']['misses']} T replies. The first-step extraction pattern matched {trace['g']['readable']} and {trace['t']['readable']} replies, respectively. This is a post-hoc check of the written work. It localizes observable errors without assuming the text reveals the model’s internal computation.",
            "## Sensitivity to supplied context",
            "The same 56 fresh charts appeared in each condition, eight per development family. L1 supplied chart facts, L2 supplied a rule explanation, and L3 supplied the correct final answer under a verified-result heading. L3w used the same heading with a plausible wrong answer. Results are descriptive. The pilot decision used the unaided primary panel.",
            table(["Context condition, out of 56", "U8", "R", "G", "T"],
                  [(label, *[closeout["context"][m][cond][metric] for m in MODELS])
                   for label, cond, metric in [
                       ("No context, correct", "L0", "right"), ("Raw chart facts, correct", "L1", "right"),
                       ("Rule explanation, correct", "L2", "right"), ("Verified correct answer, correct", "L3", "right"),
                       ("False verified answer, still correct", "L3w", "right"),
                       ("False verified answer, copied its value", "L3w", "wrong_hint")]])]
        doc.insert(doc.index("## Retention"), "Rule explanations produced the largest context difference. On the same 56 charts they moved G from 46 to 22 correct and T from 42 to 46. G and T each answered 43 correctly under a false verified heading. These findings qualify the choice of G as the simpler unaided candidate when the intended application also supplies rules.")
        i = doc.index("## Reproducibility and boundaries")
        doc[i:i] = [
            "## Recommendation",
            "1. Keep U8 as the reference. Both graph-trained adapters warrant further study. G is the simpler candidate and led on new wording. T passed the stated point thresholds, but its family-resampled advantage over G includes zero, and both retention intervals cross their respective loss margins.",
            "2. Prioritize the two day-stem-dependent operations. A follow-up diagnostic should pair each complete question with isolated source and downstream lookups using the same operand and independently written instructions. This would test whether a model that names the correct branch can reliably apply the downstream rule.",
            "3. If more research time is approved, compare G and T over multiple seeds with independent wording and additional held-out families. A matched control with new single-step examples would help separate the value of multi-step training from extra exposure to the lookup tables. G versus R here changes the example mix, wording and worked answers together. No replication or extended training was part of this pilot."]
        finish = closeout["last_reply_time"]
        doc += ["## Completion and verification",
                f"The 24-hour budget began at 12:19 p.m. Pacific on November 21. The final model reply was recorded at {finish[11:16]} Pacific that evening. The three training runs used about 4 hours 26 minutes in total. Closeout re-scored all {closeout['replies_verified']:,} archived replies, checked one reply per input ID per model and panel, and verified the input, configuration and adapter hashes. [Closeout counts and hashes check](../results/phase7/closeout.json)"]
    path=Path("docs/phase7_one_seed_pilot_results.md")
    with path.open("w") as f:f.write("\n\n".join(doc)+"\n")
if __name__=="__main__":main()
