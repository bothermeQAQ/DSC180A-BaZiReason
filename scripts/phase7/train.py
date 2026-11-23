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
    parser.add_argument("arm",choices=("r","g","t"))
    arm=parser.parse_args().arm
    manifest=json.loads(Path("data/phase7/training_manifest.json").read_text())
    cfg_path=Path(f"configs/phase7/{arm}.yaml")
    cfg=yaml.safe_load(cfg_path.read_text())
    def verify(path,expected):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    verify(cfg_path,manifest["hashes"][arm]["config"])
    for f in ("train","valid"): verify(cfg["data"][f],manifest["hashes"][arm][f])
    verify(cfg["init_adapter"],"bf4007c5fce9202c7bce7791fc8ad019fee7b266821d50b81408cc74ba8475a8")
    if Path(cfg["adapter_dir"]).exists() or Path(cfg["log_dir"]).exists():
        raise FileExistsError("Pilot arm already exists; refusing to overwrite")
    # Installed MLX-LM's default CacheDataset.itemlen returns len(raw_row), which for
    # dict-valued chat records counts keys. Use the matched three-arm length bin instead.
    CacheDataset.itemlen=lambda self,idx: self._data[idx].get("phase7_sort_length",0)
    sys.argv=["train_lora.py","--config",str(cfg_path)]
    runpy.run_path("scripts/phase2/train_lora.py",run_name="__main__")
    summary=json.loads((Path(cfg["log_dir"])/"training_summary.json").read_text())
    assert summary["selected_checkpoint_iteration"]==summary["iterations"]
if __name__=="__main__":
    main()
