from b0.run import official_lr
from b0.lr_test import candidate_lr
import os
import torch
from program.runner import set_seed,validate
from train.experiment import build_optimizer,build_model
from torch.utils.data import DataLoader
from prepare.dataset import IRDropDataset
import json
from pathlib import Path
from train.experiment import train_batch
import time
if __name__ == "__main__":
    config_path = Path(os.environ["IR_SA0_CONFIG"])
    with config_path.open("r",encoding="utf-8") as handle:
        config = json.load(handle)
    seed = config["seed"]
    total_step = config["steps"]
    train_batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    lr_horizon_steps = config["lr_horizon_steps"]
    initial_lr = official_lr(0, lr_horizon_steps)
    if config["lr_name"]=="baseline":
        lr_fn = official_lr
    elif config["lr_name"]=="candidate":
        lr_fn = candidate_lr
        initial_lr = config["initial_lr"]
    else:
        raise ValueError("未知的学习率配置")
        
    out_dir = Path(config["out_dir"])
    
    out_dir.mkdir(parents=True,exist_ok=False)
    device = torch.device("cuda")
    set_seed(seed)
    train_dataset = IRDropDataset("/mnt/d/WSL/Project/CircuitNet-main/training_set_full_mini/IR_drop","/mnt/d/Project/IR-AutoResearch/prepare/manifests/smoke_train.csv")
    val_dataset = IRDropDataset("/mnt/d/WSL/Project/CircuitNet-main/training_set_full_mini/IR_drop","/mnt/d/Project/IR-AutoResearch/prepare/manifests/smoke_validation.csv")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True,generator=generator,drop_last=True,num_workers=0)
    val_loader = DataLoader(val_dataset,val_batch_size)
    model = build_model().to(device)
    optimizer = build_optimizer(model)
    init_metrics = validate(model,val_loader,device)
    print(init_metrics)
    print("空闲显存 / 总显存 GiB:", [v / 1024**3 for v in torch.cuda.mem_get_info(device)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    iterator = iter(train_loader)
    for step in range(total_step):
        try:
            feature,target,ids = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            feature,target,ids = next(iterator)
        feature = feature.to(device)
        target = target.to(device)
        if config["lr_name"]=="baseline":
            lr = lr_fn(step,lr_horizon_steps)
        elif config["lr_name"]=="candidate":
            lr = lr_fn(step,lr_horizon_steps,initial_lr)
        for para in optimizer.param_groups:
            para["lr"]=lr
        loss = train_batch(model,optimizer,feature,target)
        print(f"step:{step+1}",
              f"learning rate:{lr}",
              f"loss:{loss}")
    torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/(1024**2)
    final_metrics = validate(model,val_loader,device)
    result = {
        "lr_function":lr_fn.__name__,
        "steps":total_step,
        "seed":seed,
        "val_batch_size": val_batch_size,
        "batch_size":train_batch_size,
        "lr_horizon_steps":lr_horizon_steps,
        "initial_metrics":init_metrics,
        "final_metrics":final_metrics,
        "training_seconds":end_time-start_time,
        "peak_allocated_mib":peak_allocated_mib,
        "initial_lr":initial_lr
    }
    with (out_dir / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(final_metrics)