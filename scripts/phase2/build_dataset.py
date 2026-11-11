"""Build or verify the Phase 2 fine-tuning dataset (``data/phase2/train_v1``).

    python scripts/phase2/build_dataset.py            # write the three splits and a manifest
    python scripts/phase2/build_dataset.py --check    # regenerate in memory and compare hashes

Generation is deterministic for a given seed, engine version and exclusion
lists. The manifest records those inputs, every file hash, and the counts by
family, task, tool mode and expected status.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from bazireason import train_data as D


def summarize(rows):
    def status(row):
        expected = row["meta"].get("expected") or {}
        return expected.get("status") or expected.get("verdict") or "-"

    return {
        "rows": len(rows),
        "groups": len({(r["family"], r["group_id"]) for r in rows}),
        "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
        "by_task": dict(sorted(Counter(r["task"] for r in rows).items())),
        "by_mode": dict(sorted(Counter(r["mode"] for r in rows).items())),
        "calendar_status": dict(
            sorted(Counter(status(r) for r in rows if r["task"] in ("chart", "verify")).items())
        ),
        "characters": sum(len(m["content"]) for r in rows for m in r["messages"]),
    }


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=sorted(D.VERSIONS), default=1)
    parser.add_argument("--output-dir", help="Default: data/phase2/train_v<dataset>")
    parser.add_argument("--seed", type=int, default=D.SEED)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir or f"data/phase2/train_v{args.dataset}")
    blocked = D.benchmark_exclusions() | D.contest_exclusions()
    years = D.split_years()
    manifest = {
        "dataset_version": D.VERSIONS[args.dataset][0],
        "wording_version": args.dataset,
        "seed": args.seed,
        "generator": "bazireason.train_data",
        "engine": {
            "lunar-python": version("lunar-python"),
            "tzdata": version("tzdata"),
        },
        "split_years": {k: sorted(v) for k, v in years.items()},
        "exclusions": {
            "benchmark_files": {
                p.name: file_sha(p) for p in sorted(Path("data/synthetic").glob("*.jsonl"))
            },
            "benchmark_margin_days": 7,
            "contest_birth_dates_sha256": file_sha(
                "data/phase2/exclusions/contest_birth_dates.json"
            ),
            "contest_margin_days": 1,
            "blocked_dates": len(blocked),
        },
        "contest_questions_used": False,
        "files": {},
    }
    failed = False
    for split in D.SPLITS:
        rows = D.generate_split(split, args.seed, blocked, version=args.dataset)
        payload = D.dataset_bytes(rows)
        digest = D.sha256_bytes(payload)
        path = out / f"{split}.jsonl"
        manifest["files"][split] = {
            "path": str(path),
            "sha256": digest,
            **summarize(rows),
        }
        if args.check:
            same = path.exists() and file_sha(path) == digest
            print(f"{'PASS' if same else 'FAIL'}: {split} {len(rows)} rows sha256 {digest}")
            failed |= not same
            continue
        if path.exists() and not args.force:
            parser.error(f"{path} exists; pass --force to replace it")
        out.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"Wrote {len(rows)} rows to {path}")
    if args.check:
        sys.exit(1 if failed else 0)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
