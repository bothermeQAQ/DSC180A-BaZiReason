"""Use the historical LoRA runner with paired length bins and a fixed final endpoint."""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import yaml
from mlx_lm.tuner.datasets import CacheDataset

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=("s","x","p"))
    arm=parser.parse_args().arm
    manifest=json.loads(Path("data/phase9/training_manifest.json").read_text())
    cfg_path=Path(f"configs/phase9/{arm}.yaml")
    cfg=yaml.safe_load(cfg_path.read_text())
    def verify(path,expected):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    verify(cfg_path,manifest["hashes"][arm]["config"])
    for f in ("train","valid"): verify(cfg["data"][f],manifest["hashes"][arm][f])
    verify(cfg["init_adapter"],manifest["parent_sha256"])
    if Path(cfg["adapter_dir"]).exists() or Path(cfg["log_dir"]).exists():
        raise FileExistsError("Phase 9 arm output already exists")
    # Installed MLX-LM's default CacheDataset.itemlen returns len(raw_row), which for
    # dict-valued chat records counts keys. Use the paired three-arm length bin instead.
    CacheDataset.itemlen=lambda self,idx: self._data[idx].get("phase9_sort_length",0)
    sys.argv=["train_lora.py","--config",str(cfg_path)]
    runpy.run_path("scripts/phase2/train_lora.py",run_name="__main__")
    summary=json.loads((Path(cfg["log_dir"])/"training_summary.json").read_text())
    assert summary["selected_checkpoint_iteration"]==summary["iterations"]
if __name__=="__main__":
    main()
