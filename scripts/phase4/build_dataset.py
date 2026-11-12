"""Write or verify the Phase 4 approach-A training data (p4_train_v1).

python scripts/phase4/build_dataset.py           # write data/phase4/train_v1 (refuses to overwrite)
python scripts/phase4/build_dataset.py --check   # regenerate in memory and compare hashes
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from bazireason import train_data as TD
from bazireason import train_data4 as D

FOLDER = Path("data/phase4/train_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    blocked = TD.benchmark_exclusions() | TD.contest_exclusions()
    manifest = {
        "dataset_version": D.VERSION,
        "seed": D.SEED,
        "generator": "bazireason.train_data4.generate_split",
        "starts_from_adapter": "bazireason-qwen3-4b-lora-p3-r3",
        "split_years": {k: [min(v), max(v), len(v)] for k, v in TD.split_years().items()},
        "contest_material_used": "birth dates only, as an exclusion list",
        "files": {},
    }
    bad = False
    for split in ("train", "valid"):
        rows = D.generate_split(split, blocked=blocked)
        payload = D.dataset_bytes(rows)
        digest = D.sha256_bytes(payload)
        path = FOLDER / f"{split}.jsonl"
        manifest["files"][f"{split}.jsonl"] = {
            "rows": len(rows),
            "sha256": digest,
            "families": dict(sorted(Counter(r["family"] for r in rows).items())),
            "augmented_questions": sum(bool(r.get("augmented")) for r in rows),
        }
        if args.check:
            same = path.exists() and D.sha256_bytes(path.read_bytes()) == digest
            bad |= not same
            print(f"{path}: {'ok' if same else 'MISMATCH'} ({len(rows)} rows)")
            continue
        if path.exists():
            parser.error(f"{path} exists")
        FOLDER.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"{path}: {len(rows)} rows sha256 {digest[:16]}")
    if not args.check:
        (FOLDER / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
