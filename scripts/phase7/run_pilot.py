"""Serial GPU pilot. Restrict every child to the 24-hour campaign budget; keep the final hour for reporting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

OUT=Path("results/phase7")
END=datetime(2026,11,22,12,19,tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
def main():
    expected=840
    replies=OUT/"u8_diagnostic.jsonl"
    while not replies.exists() or sum(1 for _ in replies.open())<expected:
        if time.time()>END-3600:
            raise TimeoutError("diagnostic did not complete before reporting reserve")
        time.sleep(30)
    subprocess.run([sys.executable,"scripts/phase7/analyze.py","diagnostic"],check=True)
    subprocess.run([sys.executable,"scripts/phase7/inspect_diagnostic.py"],check=True)
    diag=json.loads((OUT/"diagnostic_summary.json").read_text())
    audit=json.loads((OUT/"input_audit.json").read_text())
    for path,h in audit["sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    assert diag["counts"]["groups"]==120
    if diag["prerequisite_flag"]:
        findings=json.loads((OUT/"prerequisite_review.json").read_text())
        assert findings["decision"]=="proceed" and findings["reason"]
    logs=OUT/"logs";logs.mkdir(exist_ok=True)
    for model in ("r","g","t","u8"):
        jobs=[]
        if model!="u8" and not (OUT/"training"/model/"training_summary.json").exists():
            jobs.append(("train",["scripts/phase7/train.py",model]))
        jobs.append(("evaluate",["scripts/phase7/evaluate.py",model,"dev","mix","retention"]))
        for name,args in jobs:
            timeout=END-3600-time.time()
            if timeout<=0:raise TimeoutError("reporting reserve")
            with (logs/f"{model}_{name}.log").open("a") as stream:
                print(datetime.now().isoformat(),"start",model,name,flush=True)
                subprocess.run([sys.executable,"-u",*args],stdout=stream,stderr=subprocess.STDOUT,
                               timeout=timeout,check=True,env=dict(os.environ,HF_HUB_OFFLINE="1"))
            print(datetime.now().isoformat(),"done",model,name,flush=True)
    subprocess.run([sys.executable,"scripts/phase7/analyze.py","final"],check=True)
    if not Path("docs/phase7_one_seed_pilot_results.md").exists():
        subprocess.run([sys.executable,"scripts/phase7/report.py"],check=True)
    print("CORE_PILOT_COMPLETE",flush=True)
    if END-time.time()>4*3600:
        for m in ("u8","r","g","t"):
            with (logs/f"{m}_context.log").open("a") as stream:
                subprocess.run([sys.executable,"-u","scripts/phase7/context_panel.py",m],
                               stdout=stream,stderr=subprocess.STDOUT,timeout=END-3600-time.time(),check=True)
        subprocess.run([sys.executable,"scripts/phase7/closeout.py"],check=True)
        subprocess.run([sys.executable,"scripts/phase7/report.py"],check=True)
    print("PILOT_GPU_COMPLETE",flush=True)
if __name__=="__main__":main()
