"""Run only the three registered continuations and fixed evaluation roster."""
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase9"
ENV = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           TOKENIZERS_PARALLELISM="false", PYTHONPATH="src:scripts/phase7")


def verify(name):
    lock = OUT / name
    start = time.monotonic()
    while not lock.exists():
        if time.monotonic() - start > 3600:
            raise TimeoutError(f"{name} has not been created")
        time.sleep(5)
    for name, expected in json.loads(lock.read_text()).items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def job(action, arm, *panels):
    with (OUT / "logs" / f"{arm}_{action}.log").open("a") as stream:
        print(datetime.now().astimezone().isoformat(), action, arm, flush=True)
        subprocess.run([sys.executable,"-u",f"scripts/phase9/{action}.py",arm,*panels], cwd=ROOT,
                       env=ENV, check=True, stdout=stream, stderr=subprocess.STDOUT)


def main():
    verify("training_lock.json")
    for arm in ("s","x","p"):
        if not (OUT / "training" / arm / "training_summary.json").exists():
            job("train",arm)
    verify("evaluation_lock.json")
    for arm in ("s","x","p","g","a"):
        job("evaluate", arm, "transfer","bridge","retention")
    job("evaluate", "u8", "retention")
    print("All registered model jobs finished", datetime.now().astimezone().isoformat(), flush=True)


if __name__ == "__main__":
    main()
