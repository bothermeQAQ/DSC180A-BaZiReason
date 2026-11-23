"""Serial offline Phase 8 GPU campaign, bounded to leave a 55-minute report reserve."""
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase8"
TZ = ZoneInfo("America/Los_Angeles")
STOP = datetime(2026, 11, 22, 10, 17, tzinfo=TZ)


def main():
    env = dict(os.environ, PYTHONPATH="src:scripts/phase7", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    locked = json.loads((OUT / "lock.json").read_text())
    for path, expected in locked.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    jobs = [("train", arm, []) for arm in ("a", "b")]
    jobs += [("evaluate", arm, panels) for arm, panels in [
        ("a", ["transfer","retention","dev"]), ("b", ["transfer","retention","dev"]),
        ("g", ["transfer","retention"]), ("t", ["transfer","retention"]), ("u8", ["retention"])]]
    for task, arm, panels in jobs:
        if task == "train" and (OUT / "training" / arm / "training_summary.json").is_file():
            continue
        if task == "evaluate":
            eval_lock = OUT / "evaluation_lock.json"
            while not eval_lock.exists():
                if datetime.now(TZ) >= STOP:
                    raise TimeoutError("Panel preflight missed model-work deadline")
                time.sleep(10)
            for path, expected in json.loads(eval_lock.read_text()).items():
                assert sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
        now = datetime.now(TZ)
        remaining = (STOP - now).total_seconds()
        if remaining <= 0:
            raise TimeoutError("Model-work budget exhausted")
        print(now.isoformat(), task, arm, flush=True)
        with (OUT / "logs" / f"{arm}_{task}.log").open("a") as log:
            subprocess.run([sys.executable, "-u", f"scripts/phase8/{task}.py", arm, *panels], env=env,
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
    print(datetime.now(TZ).isoformat(), "all planned model jobs complete", flush=True)


if __name__ == "__main__":
    main()
