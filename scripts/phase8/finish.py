"""Produce the audited report once the final planned evaluation has been written."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
deadline = datetime(2026, 11, 22, 11, 12, tzinfo=zone).timestamp()
out = Path("results/phase8")
source = Path("data/phase8/retention.jsonl")
target = out / "u8_retention.jsonl"
expected = len(source.read_text().splitlines())
while not target.exists() or len(target.read_text().splitlines()) < expected:
    if time.time() > deadline - 1800:
        sys.exit("Missing planned evaluations at report reserve")
    time.sleep(45)
for script in ("analyze", "report"):
    subprocess.run([sys.executable, f"scripts/phase8/{script}.py"], check=True,
                   timeout=max(1, deadline - time.time()))
print("Final report written", flush=True)
