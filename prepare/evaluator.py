from __future__ import annotations

import cv2
import numpy as np
import torch


OFFICIAL_IMAGE_RANGE = 255.0


def _normalize_batch_shape(
    tensor: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B, H, W], "
            f"[B, 1, H, W], or [H, W], got {tuple(tensor.shape)}"
        )

    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or Inf")

    return tensor


def _prepare_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _normalize_batch_shape(
        prediction,
        "prediction",
    )
    target = _normalize_batch_shape(
        target,
        "target",
    )

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} "
            f"does not match target shape {tuple(target.shape)}"
        )

    return prediction, target


def _to_official_uint8(
    tensor: torch.Tensor,
) -> np.ndarray:
    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
    )

    return np.rint(
        array * OFFICIAL_IMAGE_RANGE
    ).astype(np.uint8)


def mae_float(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction, target = _prepare_pair(
        prediction,
        target,
    )

    return torch.mean(
        torch.abs(prediction.float() - target.float())
    ).item()


def _nrms_single(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)

    rmse = np.sqrt(
        np.mean((prediction - target) ** 2)
    )
    dynamic_range = target.max() - target.min()

    if dynamic_range == 0:
        return 0.0 if rmse == 0 else 0.05

    value = rmse / dynamic_range

    if not np.isfinite(value):
        return 0.05

    return float(value)


def nrms_official(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction, target = _prepare_pair(
        prediction,
        target,
    )

    prediction_uint8 = _to_official_uint8(prediction)
    target_uint8 = _to_official_uint8(target)

    values = [
        _nrms_single(prediction_image, target_image)
        for prediction_image, target_image
        in zip(prediction_uint8, target_uint8)
    ]

    return float(np.mean(values))


def _ssim_single(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    if min(prediction.shape) < 11:
        raise ValueError(
            "SSIM requires images at least 11x11"
        )

    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)

    constant_1 = (0.01 * OFFICIAL_IMAGE_RANGE) ** 2
    constant_2 = (0.03 * OFFICIAL_IMAGE_RANGE) ** 2

    gaussian = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(gaussian, gaussian.transpose())

    prediction_mean = cv2.filter2D(
        prediction,
        -1,
        window,
    )[5:-5, 5:-5]
    target_mean = cv2.filter2D(
        target,
        -1,
        window,
    )[5:-5, 5:-5]

    prediction_mean_squared = prediction_mean ** 2
    target_mean_squared = target_mean ** 2
    mean_product = prediction_mean * target_mean

    prediction_variance = (
        cv2.filter2D(prediction ** 2, -1, window)[5:-5, 5:-5]
        - prediction_mean_squared
    )
    target_variance = (
        cv2.filter2D(target ** 2, -1, window)[5:-5, 5:-5]
        - target_mean_squared
    )
    covariance = (
        cv2.filter2D(
            prediction * target,
            -1,
            window,
        )[5:-5, 5:-5]
        - mean_product
    )

    ssim_map = (
        (2 * mean_product + constant_1)
        * (2 * covariance + constant_2)
    ) / (
        (prediction_mean_squared + target_mean_squared + constant_1)
        * (prediction_variance + target_variance + constant_2)
    )

    return float(ssim_map.mean())


def ssim_official(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction, target = _prepare_pair(
        prediction,
        target,
    )

    prediction_uint8 = _to_official_uint8(prediction)
    target_uint8 = _to_official_uint8(target)

    values = [
        _ssim_single(prediction_image, target_image)
        for prediction_image, target_image
        in zip(prediction_uint8, target_uint8)
    ]

    return float(np.mean(values))


def evaluate_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    return {
        "mae_float": mae_float(prediction, target),
        "nrms_official": nrms_official(prediction, target),
        "ssim_official": ssim_official(prediction, target),
    }