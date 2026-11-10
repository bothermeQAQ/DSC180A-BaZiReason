"""Download pinned open-weight model snapshots for the Phase 2 local MLX experiments.

Every entry names an exact Hugging Face commit, so a later upstream change cannot
silently alter an experiment. Only weight, tokenizer and configuration files are
fetched (no repository code is downloaded or executed), anonymously and without
telemetry. Run:  python scripts/phase2/download_models.py --only selected
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from huggingface_hub import snapshot_download  # noqa: E402

PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.jinja",
    "*.txt",
    "*.model",
    "LICENSE*",
    "README*",
    "NOTICE*",
]
# role: "selected" is the Phase 2 base model; the others are bake-off candidates
# or an optional external comparison.
MODELS = {
    "qwen3.5-4b": {
        "repo": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "role": "candidate",
        "format": "official_bf16",
    },
    "minicpm5-2b": {
        "repo": "openbmb/MiniCPM5-2B",
        "revision": "42433e90dad95123ab87729fd7011e7110629556",
        "role": "candidate",
        "format": "official_bf16",
    },
    "qwen3.5-2b-4bit": {
        "repo": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3",
        "role": "candidate",
        "format": "community_mlx_4bit",
    },
    "gemma-4-e4b-4bit": {
        "repo": "mlx-community/gemma-4-e4b-it-4bit",
        "revision": "475b9088d29754a3379866cf5aeb6b41acd313c2",
        "role": "candidate",
        "format": "community_mlx_4bit",
    },
    "qwen3-4b-2507": {
        "repo": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "role": "selected",
        "format": "official_bf16",
    },
    "qwen3-4b-2507-4bit": {
        "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "revision": "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
        "role": "candidate",
        "format": "community_mlx_4bit",
    },
    "bazi-qwen3-1.7b": {
        "repo": "dclef/Bazi-Qwen3-1.7B",
        "revision": "88945767e1debd762c6be28658d0e3baf720032a",
        "role": "external_specialized_comparison",
        "format": "community_finetune_bf16",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(MODELS), help="Subset of model keys")
    parser.add_argument("--manifest", default="artifacts/models/download_manifest.json")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in args.only or sorted(MODELS):
        spec = MODELS[key]
        start = time.perf_counter()
        path = snapshot_download(
            spec["repo"], revision=spec["revision"], allow_patterns=PATTERNS, max_workers=8
        )
        files = sorted(p for p in Path(path).rglob("*") if p.is_file())
        manifest[key] = {
            **spec,
            "local_path": path,
            "n_files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "download_seconds": round(time.perf_counter() - start, 1),
        }
        print(f"{key}: {manifest[key]['bytes'] / 1e9:.2f} GB in {path}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
