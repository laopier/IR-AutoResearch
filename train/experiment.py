from __future__ import annotations

import torch
import torch.nn.functional as functional

from train.mavi import MAVI


def build_model() -> torch.nn.Module:
    model = MAVI()
    model.init_weights()
    return model


def build_optimizer(
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7,
    )


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} "
            f"does not match target shape {tuple(target.shape)}"
        )

    return 100.0 * functional.l1_loss(
        prediction,
        target,
    )


def train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    feature: torch.Tensor,
    target: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(feature)
    loss = compute_loss(prediction, target)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Training loss is NaN or Inf")

    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not all(
        parameter.grad is not None
        for parameter in trainable_parameters
    ):
        raise RuntimeError(
            "Some trainable parameters did not receive gradients"
        )

    if not all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in trainable_parameters
    ):
        raise RuntimeError(
            "Some gradients contain NaN or Inf"
        )

    optimizer.step()

    return float(loss.detach().item())