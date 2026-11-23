"""Analyze and report once all pre-registered responses are available."""
import os
from pathlib import Path
import subprocess
import sys
import time

last = Path("results/phase9/u8_retention.jsonl")
expected = len(Path("data/phase9/retention.jsonl").read_text().splitlines())
while not last.exists() or len(last.read_text().splitlines()) < expected:
    time.sleep(30)
for name in ("analyze_transfer","analyze_preservation","error_audit","report"):
    subprocess.run([sys.executable,"scripts/phase9/"+name+".py"], check=True,
                   env=dict(os.environ,PYTHONPATH="src:scripts/phase7"))
print("All registered comparisons written", flush=True)
