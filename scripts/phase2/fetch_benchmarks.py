"""Fetch the external contest benchmarks at pinned commits into ``external/`` (git-ignored).

    python scripts/phase2/fetch_benchmarks.py

The benchmarks are evaluation data only. They are never copied into the
repository and never read by the training-data generator, except for the list of
birth dates in ``data/phase2/exclusions`` that keeps those dates out of training.
The repositories' own scripts are not executed.
"""

import subprocess
from pathlib import Path

PINNED = {
    "BaziQA": (
        "https://github.com/ChenJiangxi/BaziQA.git",
        "3bad3deaa5aa3a191c9c68b4d8b17f46decff45e",  # data and charts identical to tag v2.3.0
    ),
    "MingLi-Bench": (
        "https://github.com/DestinyLinker/MingLi-Bench.git",
        "b7433280fd86d7a7c27debbc47d0303c218f0bfd",
    ),
}


def main():
    root = Path("external")
    root.mkdir(exist_ok=True)
    for name, (url, commit) in PINNED.items():
        target = root / name
        if not target.exists():
            subprocess.run(
                ["git", "-c", "credential.helper=", "clone", "--quiet", url, str(target)], check=True
            )
        subprocess.run(["git", "-C", str(target), "checkout", "--quiet", commit], check=True)
        head = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"{name}: {head}")


if __name__ == "__main__":
    main()
