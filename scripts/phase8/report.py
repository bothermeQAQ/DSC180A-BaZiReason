"""Render the fixed experiment comparison from re-scored replies and training logs."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

RESULTS = Path("results/phase8")
MODELS = ("g","t","b","a")


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str,row)) + " |" for row in [headers, ["---"]*len(headers), *rows])


def score(x):
    return f"{x['right']}/{x['n']} ({x['pct']:.1f}%)"


def interval(x):
    return f"[{x[0]:+.1f}, {x[1]:+.1f}]"


def report():
    r = read(RESULTS / "comparison.json")
    t, old, ret = r["transfer"], r["dev"], r["retention"]
    train = {m:read(RESULTS / f"training/{m}/training_summary.json") for m in ("a","b")}
    budget = read(Path("data/phase8/training_manifest.json"))
    primary = t["novel"]
    winners = [m.upper() for m in MODELS if primary["counts"][m]["right"] == max(x["right"] for x in primary["counts"].values())]
    delta = primary["paired"]["a-b"]
    screen = r["screening"]
    time = datetime.now(ZoneInfo("America/Los_Angeles"))
    start = datetime(2026,11,22,1,12,tzinfo=time.tzinfo)
    elapsed = (time - start).total_seconds()/3600
    sections = ["# Phase 8 · Transfer to unseen BaZi rule combinations",
                f"**Primary leader · {' and '.join(winners)}.** The targeted continuation (A) solved {score(primary['counts']['a'])} independently worded questions, compared with {score(primary['counts']['b'])} for its matched active control and {score(primary['counts']['g'])} for its parent. A−B was **{delta['pp']:+.1f} percentage points**, with a paired chart 95% interval of {interval(delta['chart_ci'])} and a family-and-chart interval of {interval(delta['family_and_chart_ci'])}. The predeclared screen **{'passed' if screen['passes'] else 'did not pass'}**.",
                "## Why this experiment",
                "Phase 7’s conventional worked examples (G) raised primary accuracy from 17/140 under ordinary replay to 107/140. Explicit operands (T) reached 112/140. Every G and T primary miss named the correct intermediate branch. On all 26 life-stage errors across the two adapters the text also gave the right start, direction and target. On all 35 Ten-God errors it named the correct hidden stem. This motivated a direct test of whether strengthening the last operation would carry through to a fresh multi-step question.",
                "G was the parent because its unaided transfer held up across wording changes. T was evaluated as a second frozen contender. A practiced extracting the day stem from the chart and applying it to an explicitly supplied branch. B practiced main or all hidden stems, matching element counts and positions on the same charts and branch operands. B controls for continued optimization, new chart exposure and worked lookups. Its practice can also improve performance on the eight other families.",
                "## Training and timing",
                "Each arm trained once, with seed 810117, from the exact Phase 7 G weights. Each used 3,040 common replay examples plus 1,920 fresh single-step examples. Each of the 120 day-stem × branch combinations appeared eight times in A for each target operation. Neither arm trained a new two-step graph family. Both used rank-16 LoRA, a fresh AdamW optimizer, learning rate 0.00005 with cosine decay to 0.000005 and the fixed final checkpoint. No evaluation answer selected a checkpoint.",
                table(["Arm","Rows","Updates","Sequence tokens","Supervised tokens","Padded tokens (plan)","Minutes","Peak GB","Final train loss"],
                      [[m.upper(),s["train_rows"],s["optimizer_updates"],f"{s['sequence_tokens_processed']:,}",f"{s['loss_tokens_total']:,}",f"{budget['stats'][m]['padded']:,}",f"{s['duration_seconds']/60:.1f}",f"{s['peak_memory_gb']:.2f}",f"{s['final_train_loss']:.4f}"] for m,s in train.items()]),
                f"The budget started at 01:12 Pacific on November 22. GPU training began at 02:19. The report was finalized at {time:%H:%M} Pacific, {elapsed:.2f} hours after the start, including preparation, idle time, inference, verification and reporting.",
                "## Transfer and retention",
                "Each new primary question combines two familiar operations along a directed edge held outside Phase 7 training, diagnostics and development. The panel covers all 12 reserved edges, with 20 independently drawn chart groups per edge. Year/day source position and two new wordings were crossed within edge. A separate evaluation agent wrote the new Chinese wording. All models used the same unaided greedy decoder and answer contract. Each model generated one reply per prompt.",
                table(["Panel","G · parent","T · explicit fields","B · control","A · targeted"],
                      [[label,*[score(panel["counts"][m]) for m in MODELS]] for label,panel in [
                          ("New wording, primary",primary),("Familiar wording, same charts",t["familiar"]),
                          ("New wording, four day-dependent families",t["day_dependent"]),("New wording, eight other families",t["other"]),
                          ("Changed source, four targeted families",t["switch"]),("Correct branch given, downstream only",t["consumer"]),
                          ("Familiar rules, fresh",ret["rules"]),("Calendar, fresh",ret["calendar"]),
                          ("Phase 7 development, original wording",old["full"]),("Phase 7 development, source switched",old["twin"]),
                          ("Phase 7 development, alternate wording",old["alternate"])]]),
                table(["Primary contrast","Δ pp","95% chart interval","95% family + chart interval","Removing one family, range","Left only / right only"],
                      [[k.upper(),f"{v['pp']:+.1f}",interval(v["chart_ci"]),interval(v["family_and_chart_ci"]),interval(v["leave_one_family_out_range"]),f"{v['left_only']} / {v['right_only']}"] for k,v in primary["paired"].items()]),
                "The paired intervals use 10,000 bootstrap draws. The chart interval conditions on the 12 observed families. The broader interval also resamples families. Each model contributed a single fixed training realization. Phase 7 development outcomes are descriptive because they informed the intervention.",
                "## Which operations transferred",
                table(["Source → downstream lookup","G","T","B","A"],
                      [[family,*[f"{v[m]['right']}/{v[m]['n']}" for m in MODELS]] for family,v in primary["by_family"].items()]),
                table(["Matched-chart diagnostic","G","T","B","A"],
                      [["Both wordings correct",*[f"{t['matched_wordings'][m]['both']}/240" for m in MODELS]],
                       ["Both source choices correct when gold changes",*[f"{t['matched_hard_charts'][m]['differing_switch']['both']}/{t['matched_hard_charts'][m]['differing_switch']['n']}" for m in MODELS]],
                       ["Isolated downstream correct, full question wrong",*[f"{t['matched_hard_charts'][m]['hard']['isolated_only']}/80" for m in MODELS]],
                       ["Full question correct, isolated downstream wrong",*[f"{t['matched_hard_charts'][m]['hard']['novel_only']}/80" for m in MODELS]],
                       ["Novel wording form 1",*[score(t["novel_by_writer_form"]["0"][m]) for m in MODELS]],
                       ["Novel wording form 2",*[score(t["novel_by_writer_form"]["1"][m]) for m in MODELS]]]),
                f"A shortcut that applies the downstream operation directly to the natal branch scored {score(t['novel_natal_shortcut'])}. The different-answer source switches test whether the model changes its answer when the chart stays fixed. The isolated probe supplies the correct derived branch. A high isolated score alongside a much lower full score suggests a gap in using the operation inside a chain.",
                "## Preservation of rule and calendar capabilities",
                f"U8 scored {score(ret['rules']['counts']['u8'])} on the new rule questions and {score(ret['calendar']['counts']['u8'])} on calendar questions. Retention intervals resample whole event groups within task type. All contrasts below are percentage points.",
                table(["Track","Contrast","Δ","Paired 95% interval"],
                      [[k,label.upper(),f"{ret[k]['paired'][label]['pp']:+.2f}",interval(ret[k]['paired'][label]['chart_ci'])] for k in ("rules","calendar") for label in ("a-g","a-u8","b-g","t-u8")]),
                table(["Calendar behavior","U8","G","T","B","A"],
                      [[label,*[f"{ret['calendar_behavior'][m][flag]}/{ret['calendar_behavior'][m][den]}" for m in ("u8",*MODELS)]] for label,flag,den in [
                           ("Refused an answerable chart","false_refusal","answerable"),("Gave a definite chart on a nonanswerable case","definitive_on_nonanswer","nonanswerable")]]),
                "## Decision and limits",
                "The screen requires A−B of at least +4 points with its chart-paired interval above zero, improvement over G, and point losses no larger than 1.5 familiar-rule points or 2 calendar points against both G and U8. " + ("All criteria passed." if screen["passes"] else "A met all four retention point criteria. It missed the required primary advantage over B, the chart interval crossed zero, and its primary score fell below G."),
                "The 12 families are a finite set of synthetic two-step rules. The two formulations vary the language while sharing the chart header, system prompt and answer schema. Broad real-user wording, deeper chains, base-model pretraining exposure and seed-to-seed variation remain unmeasured. These reserves have now been evaluated. Any subsequent confirmation needs new families or a new task construction.",
                "## Integrity and reproducibility",
                "Before any transfer inference, input-only checks found chart tuples that repeated across date blocks and calendar dates or prompts shared with the archive. We regenerated the panels from their original seeds with explicit exclusions. The retention audit checks chart identities, exact prompts and calendar input dates. The evaluator stored raw replies and input and adapter hashes. The analysis re-scored every reply, checked identities and paired chart groups, and reused the archived Phase 7 G/T replies only on their original development inputs.",
                "[Frozen protocol](phase8_protocol.md) · [Training inputs and hashes](../data/phase8/training_manifest.json) · [Transfer panel](../data/phase8/transfer_manifest.json) · [Retention audit](../results/phase8/retention_audit.json) · [Full numerical comparison and provenance](../results/phase8/comparison.json) · [Raw replies, timings and training logs](../results/phase8)"]
    # This interpretation is specific to this completed, locked experiment.
    assert [primary["counts"][m]["right"] for m in MODELS] == [151,145,150,144]
    idx = sections.index("## Preservation of rule and calendar capabilities")
    sections[idx:idx] = [
        "A gained 26.3 percentage points over B on isolated downstream probes (paired chart 95% interval +13.8 to +38.8). On full questions that used those same operands, its gain was 6.3 points (−3.8 to +16.2). Across the eight other families, A lost 6.9 points relative to B. The gain on the earlier Phase 7 primary (+11 answers over G) and the familiar wording of the new panel (+7) did not carry over to the new primary wording (−7).",
        "A post-hoc output check found that A answered only 2/40 new life-stage questions correctly. Among the 38 misses, six final answers named the wrong life stage, 14 named a Ten God and 18 returned some other value or failed to parse. The same model could often finish the lookup when given the branch. On transfer-03-00, A correctly derived 申, then answered 比肩, a Ten God. Given the same 申 directly, it used day stem 辛 to count four places backward from 子 and correctly answered 帝旺. On transfer-05-01, A derived 酉 correctly but reversed the elemental relationship in its Ten-God calculation. Its isolated answer correctly identified 正财. These paired replies show two different failure modes, choosing the requested operation and executing it within the longer question."
    ]
    idx = sections.index("## Integrity and reproducibility")
    sections[idx:idx] = [
        "## Model choice and next experiment",
        "Keep Phase 7 G as the generalization reference. It led the primary panel by one answer over B and seven over A, and no primary pairwise interval excluded zero. U8 remains the retention reference. Despite G and A having the same aggregate calendar score, A issued six definite charts on nonanswerable cases against G’s two. The preservation thresholds passed on point estimates for A, while the differences versus U8 remain uncertain at the specified margins.",
        "The next experiment should test selection of the downstream operation in a complete question. Using training-only combinations, pair the same chart, source and derived branch with two different requests, such as life stage versus Ten God. Also pair the same requested operation with different source branches. Vary the wording independently of the operation and require the normal final answer. Compare with an equally sized ordinary worked-answer continuation from G at the same token budget. Hold out entire source–operation pairs and use independently written questions plus the paired isolated probes. A gain on composed questions while retaining isolated accuracy would directly address the failure observed here.",
        "A second seed would check training variability after a method clears a fresh end-to-end screen. Reusing this reserve panel for future tuning would turn it into development data. No follow-on training or deployment was launched."
    ]
    output = Path("docs/phase8_transfer_experiment.md")
    output.write_text("\n\n".join(sections)+"\n")


if __name__ == "__main__":
    report()
