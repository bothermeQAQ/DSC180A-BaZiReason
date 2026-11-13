"""Write or verify the Phase 5 training data p5_train_v1 (answering from supplied facts).

    python scripts/phase5/build_train.py           # data/phase5/train_v1 (refuses to overwrite)
    python scripts/phase5/build_train.py --check   # regenerate in memory and compare hashes
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from bazireason import train_data5 as D

FOLDER = Path("data/phase5/train_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = {"dataset_version": D.VERSION, "seed": D.SEED, "generator": "bazireason.train_data5.generate_split",
                "starts_from_adapter": "bazireason-qwen3-4b-lora-p4-a1", "prompt_version": 2, "files": {}}  # fmt: skip
    bad = False
    for split in ("train", "valid"):
        rows = D.generate_split(split)
        payload = D.dataset_bytes(rows)
        digest = D.sha256_bytes(payload)
        path = FOLDER / f"{split}.jsonl"
        manifest["files"][f"{split}.jsonl"] = {"rows": len(rows), "sha256": digest,
                                               "families": dict(sorted(Counter(r["family"].split("_replay_")[0] if "_replay_" in r["family"] else r["family"] for r in rows).items()))}  # fmt: skip
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
        (FOLDER / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
