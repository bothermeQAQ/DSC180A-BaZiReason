"""Generate, verify or rewrite a Phase 3 dataset.

python scripts/phase3/build_dataset.py --check          # regenerate in memory, compare hashes
python scripts/phase3/build_dataset.py --force          # rewrite data/phase3/train_v1
python scripts/phase3/build_dataset.py --dataset 2 ...  # the second-round dataset
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from bazireason import train_data as TD
from bazireason import train_data3 as P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1, choices=sorted(P.VERSIONS))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    name = P.VERSIONS[args.dataset][0]
    folder = Path("data/phase3") / name.removeprefix("p3_")
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    manifest = {
        "dataset_version": name,
        "seed": P.SEED,
        "generator": "bazireason.train_data3.generate_split",
        "split_years": {k: [min(v), max(v), len(v)] for k, v in TD.split_years().items()},
        "blocked_dates": len(blocked),
        "contest_material_used": "birth dates only, as an exclusion list",
        "files": {},
    }
    mismatch = False
    for split in P.SPLITS:
        rows = P.generate_split(split, blocked=blocked, version=args.dataset)
        if not rows:
            continue
        payload = P.dataset_bytes(rows)
        digest = P.sha256_bytes(payload)
        path = folder / f"{split}.jsonl"
        manifest["files"][f"{split}.jsonl"] = {
            "rows": len(rows),
            "sha256": digest,
            "families": dict(sorted(Counter(row["family"] for row in rows).items())),
            "modes": dict(sorted(Counter(row["mode"] for row in rows).items())),
        }
        if args.check:
            same = path.exists() and P.sha256_bytes(path.read_bytes()) == digest
            mismatch |= not same
            print(f"{path}: {'ok' if same else 'MISMATCH'} ({len(rows)} rows)")
            continue
        if path.exists() and not args.force:
            parser.error(f"{path} exists; use --force to rewrite")
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"{path}: {len(rows)} rows sha256 {digest[:16]}")
    if not args.check:
        (folder / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    return int(mismatch)


if __name__ == "__main__":
    raise SystemExit(main())
