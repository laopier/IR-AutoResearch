import math
from b0.run import official_lr
def candidate_lr(step: int, horizon: int,initial_lr: float = 1e-4) -> float:
    # Official train.py sets the LR BEFORE update, with zero-based iter_num.
    if not 0 <= step < horizon:
        raise ValueError('step must be in [0, lr_horizon_steps)')
    if not initial_lr > 1e-7:
        raise ValueError("initial_lr must be higher than 1e-7")
    return 1e-7 + 0.5 * (initial_lr - 1e-7) * (math.cos(math.pi * step / horizon) + 1)
if __name__ == "__main__":
    initial_lr = 1.5e-4
    print(candidate_lr(0,200000,initial_lr))
    print(official_lr(0,200000))
