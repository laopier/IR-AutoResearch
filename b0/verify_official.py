"""Check local B0 against the provided official CircuitNet implementation."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import torch

from b0.run import official_lr, sha256
from train.experiment import build_model, build_optimizer, compute_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference_root = Path(__file__).parent / 'reference'
    model_path = args.official_root / 'models/mavi.py' if args.official_root else reference_root / 'circuitnet_mavi.py'
    train_path = args.official_root / 'train.py' if args.official_root else reference_root / 'circuitnet_train.py'
    init_path = reference_root / 'mmcv_weight_init_v1_6_0.py'
    # Use the two upstream MMCV helpers without installing unrelated MMCV ops.
    namespace = {'nn': torch.nn, '_BatchNorm': torch.nn.modules.batchnorm._BatchNorm}
    init_tree = ast.parse(init_path.read_text())
    helpers = [node for node in init_tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ['constant_init', 'kaiming_init']]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(init_path), 'exec'), namespace)
    model_tree = ast.parse(model_path.read_text())
    model_tree.body = [node for node in model_tree.body
                       if not (isinstance(node, ast.ImportFrom) and (node.module or '').startswith('mmcv'))]
    exec(compile(model_tree, str(model_path), 'exec'), namespace)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    expected = namespace['MAVI']()
    expected.init_weights()
    torch.manual_seed(17)
    actual = build_model()
    assert expected.state_dict().keys() == actual.state_dict().keys()
    assert all(torch.equal(expected.state_dict()[k], v) for k, v in actual.state_dict().items())
    feature = torch.rand(2, 1, 24, 32, 32)
    target = torch.rand(2, 32, 32)
    expected.train(); actual.train()
    expected_prediction = expected(feature)
    actual_prediction = actual(feature)
    assert torch.equal(expected_prediction, actual_prediction)
    expected_loss = 100.0 * torch.nn.functional.l1_loss(expected_prediction, target)
    actual_loss = compute_loss(actual_prediction, target)
    assert torch.equal(expected_loss, actual_loss)
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=2e-4,
                                           betas=(0.9, 0.999), weight_decay=1e-2)
    actual_optimizer = build_optimizer(actual)
    expected_loss.backward(); actual_loss.backward()
    expected_optimizer.step(); actual_optimizer.step()
    assert all(torch.equal(expected.state_dict()[k], v) for k, v in actual.state_dict().items())
    tree = ast.parse(train_path.read_text())
    scheduler_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == 'CosineRestartLr')
    namespace = {'cos': math.cos, 'pi': math.pi}
    exec(compile(ast.Module(body=[scheduler_class], type_ignores=[]), str(train_path), 'exec'), namespace)
    reference_scheduler = namespace['CosineRestartLr'](2e-4, [200000], [1], 1e-7)
    steps = [0, 1, 99, 10000, 100000, 199999]
    lr_results = {str(step): official_lr(step, 200000) for step in steps}
    assert all(math.isclose(official_lr(step, 200000), reference_scheduler.get_lr(step, 2e-4),
                            rel_tol=1e-14, abs_tol=1e-15) for step in steps)
    result = {'initial_state_dict_equal': True, 'forward_equal': True,
              'loss_equal': True, 'one_optimizer_update_equal': True,
              'official_learning_rate_equal': True, 'learning_rates': lr_results,
              'reference_sha256': {str(p): sha256(p) for p in [model_path, train_path, init_path]},
              'reference_dependency_adapter': 'MMCV imports replaced by extracted upstream v1.6.0 initialization helpers and PyTorch _BatchNorm.',
              'notes': 'Same PyTorch environment; not a complete training reproduction or cross-version equivalence proof.'}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
