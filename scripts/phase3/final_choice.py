"""Apply the calendar constraint of the Phase 3 brief to the round chosen on validation.

The brief says: do not sacrifice the Phase 2 calendar skills to improve interpretation
scores. ``select_final.py`` (fixed before round 3 was scored) tried to encode this by
adding Phase 2 validation answers to Phase 3 ones, and chose round 2 by 357 to 353. The
calendar check split, generated for exactly this comparison and never inspected, then
showed round 2 clearly below Phase 2 on calendar rows and round 3 not.

This script records the decision taken after that test result, with its evidence:
a round is eligible only if its calendar rows on the check split are not significantly
below Phase 2 (upper end of the 95% paired interval at or above zero). The rule's choice
stands if it is eligible; otherwise the eligible round is taken. BaziQA is not consulted.
Run after ``make_report.py`` has written ``summary.json``; rerun ``make_report.py`` after.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results" / "phase3"
CHECK = "Never-inspected check split"
ADAPTER = "artifacts/adapters/bazireason-qwen3-4b-lora-{}"


def main():
    rule = json.loads((ROOT / "selection_final.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    split = next(v for k, v in summary["phase2_skills"].items() if k.startswith(CHECK))
    paired = {k.replace(" (final)", ""): v for k, v in split["paired_vs_phase2"].items()}
    evidence = {}
    for label, stem in (("Round 2", "p3-r2"), ("Round 3", "p3-r3")):
        cal = paired[label]["calendar"]
        evidence[stem] = {
            "calendar_rows": cal["n"],
            "difference_vs_phase2": round(cal["difference"], 4),
            "interval95": [round(x, 4) for x in cal["interval95"]],
            "all_rows_difference_vs_phase2": round(paired[label]["all"]["difference"], 4),
            "eligible": cal["interval95"][1] >= 0,
        }
    eligible = [stem for stem, e in evidence.items() if e["eligible"]]
    if rule["selected"] in eligible or not eligible:
        chosen = rule["selected"]
    else:
        chosen = eligible[0]
    record = {
        "decided_after_test": True,
        "constraint": (
            "From the Phase 3 brief: do not sacrifice the Phase 2 calendar skills. A round is "
            "eligible only if its calendar rows on data/phase3/p2check_v1/test.jsonl are not "
            "significantly below the Phase 2 adapter (upper end of the 95% paired interval >= 0)."
        ),
        "validation_rule_choice": rule["selected"],
        "evidence_check_split": evidence,
        "selected": chosen,
        "selected_adapter": ADAPTER.format(chosen),
        "overrides_validation_rule": chosen != rule["selected"],
        "caveat": (
            "Decided after the test splits were scored, so the selected round's test numbers are "
            "post-selection. Two candidates only; BaziQA was not consulted."
        ),
    }
    (ROOT / "final_choice.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(record["selected_adapter"])


if __name__ == "__main__":
    main()
