"""Inference of a frozen v2 model; no validation/hidden labels accepted."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from sa0.controller import read_json, sha256, source_hashes
    manifest = read_json(args.frozen / "FROZEN.json")
    if sha256(args.frozen / "checkpoint.pt") != manifest["checkpoint_sha256"] or source_hashes(args.frozen / "source") != manifest["source_hashes"]:
        raise ValueError("冻结内容哈希不一致")
    sys.path.insert(0, str((args.frozen / "source").resolve()))
    import torch
    import numpy as np
    from train.experiment import build_model
    from train.feature_transform import transform
    from prepare.dataset import EXPECTED_FEATURE_CHANNELS
    feature = np.load(args.feature, allow_pickle=False)
    if feature.ndim != 3 or feature.shape[-1] != EXPECTED_FEATURE_CHANNELS or not np.isfinite(feature).all():
        raise ValueError("输入应为有限H/W/24通道特征")
    # Exact current IRDropDataset conversion: sample [1,24,H,W], batch [N,1,24,H,W].
    tensor = torch.from_numpy(np.ascontiguousarray(feature.transpose(2, 0, 1)[None, ...],
                                                 dtype=np.float32)).unsqueeze(0)
    model = build_model().to(args.device)
    checkpoint = torch.load(args.frozen / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["model"].items()}, strict=True)
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(args.device)
        changed = transform(tensor)
        if changed.shape != tensor.shape or not torch.isfinite(changed).all():
            raise ValueError("特征变换不符合冻结shape/有限值约定")
        prediction = model(changed).cpu().numpy()[0]
    if prediction.shape != feature.shape[:2] or not np.isfinite(prediction).all():
        raise ValueError("预测shape或数值无效")
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("xb") as handle:
        np.save(handle, prediction, allow_pickle=False)


if __name__ == "__main__":
    main()
