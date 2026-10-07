"""Audited 128/32, 500-update B0 pre-run; original B0 recipe stays unchanged."""
import json
from pathlib import Path
import subprocess
import sys
from prepare.pilot_ready import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "b0/pilot_config.json").read_text(encoding="utf-8"))
    verify(config["data_root"], config["dataset_report"], [config])
    command = [sys.executable, "-u", "-m", "b0.run", "--scope", "development_baseline"]
    for field, flag in (("data_root", "--data-root"), ("train_manifest", "--train-manifest"),
                        ("val_manifest", "--validation-manifest"), ("output", "--output"),
                        ("steps", "--steps"), ("seed", "--seed"), ("batch_size", "--batch-size"),
                        ("lr_horizon_steps", "--lr-horizon-steps"), ("checkpoint_every", "--checkpoint-every")):
        command.extend([flag, str(config[field])])
    command.extend(["--device", "cuda", "--num-workers", "0", "--cpu-threads", "2", "--log-every", "10"])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
