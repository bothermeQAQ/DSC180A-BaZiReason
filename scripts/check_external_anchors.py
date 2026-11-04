"""Reproduce the small manually curated HKO spot checks (no network required)."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from lunar_python import Solar

from bazireason.calendar_engine import compute_chart
from bazireason.conventions import Convention


def main():
    source_path = Path("docs/sources/external_anchors.json")
    source = json.loads(source_path.read_text())
    checks = []
    for anchor in source["day_anchors"]:
        computed = compute_chart(
            {"datetime": anchor["date"] + "T12:00:00", "timezone": "Asia/Shanghai"},
            Convention().to_dict(),
        )["pillars"]["day"]
        checks.append({**anchor, "computed": computed, "agrees": computed == anchor["day"]})
    term = Solar.fromYmdHms(2025, 2, 1, 12, 0, 0).getLunar().getJieQiTable()["立春"]
    expected = datetime.fromisoformat(source["term_anchor"]["datetime_utc8_minute"])
    computed = datetime.fromisoformat(term.toYmdHms())
    delta = abs((computed - expected).total_seconds())
    result = {
        "kind": "partial_independent_published_anchors_not_full_chart_validation",
        "source": source["url"],
        "anchor_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "day_checks": checks,
        "term_check": {
            "published_minute": expected.isoformat(),
            "computed": computed.isoformat(),
            "absolute_difference_seconds": delta,
            "within_60_second_publication_resolution": delta < 60,
        },
        "limits": source["limits"],
    }
    Path("results/external_anchor_checks.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    assert all(c["agrees"] for c in checks)
    assert delta < 60
    print("PASS: 2 published day anchors and 1 minute-resolution Lichun check")


if __name__ == "__main__":
    main()
