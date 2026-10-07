import os
from pathlib import Path
import numpy as np
import torch
from b0.run import official_lr
from train.experiment import build_model, build_optimizer, train_batch
from torch.utils.data import DataLoader
import random
from prepare.dataset import IRDropDataset
from program.runner import set_seed
resume_path = None
if resume_path is not None:
    loaded_checkpoint = torch.load(resume_path,map_location="cpu")
else:
    loaded_checkpoint = None
device = torch.device("cuda")
set_seed(0)
model = build_model().to(device=device)
optimizer = build_optimizer(model)
output_dir = Path("/mnt/d/Project/IR-AutoResearch/results/resume_learning_continue")
output_dir.mkdir(parents=True, exist_ok=False)
dataset = IRDropDataset("/mnt/d/WSL/Project/CircuitNet-main/training_set_full_mini/IR_drop","/mnt/d/Project/IR-AutoResearch/prepare/manifests/smoke_train.csv")
if resume_path is not None:
    state_of_model = loaded_checkpoint["model"]
    model.load_state_dict(state_of_model)
    state_of_optimizer = loaded_checkpoint["optimizer"]
    optimizer.load_state_dict(state_of_optimizer)
    start_step = loaded_checkpoint["step"]
    
    
else:
    start_step = 0
checkpoint_every =100
total_step = 4# A number we want  

generator = torch.Generator()
generator.manual_seed(0)
loader = DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=0,generator=generator)
if resume_path is not None:
    epoch_generator_state = loaded_checkpoint["epoch_generator_state"]
    generator.set_state(epoch_generator_state)
    iterator = iter(loader)

    batch_consume = loaded_checkpoint["batch_consume"]

    for i in range(batch_consume):
        next(iterator)
    if not torch.equal(generator.get_state(),loaded_checkpoint["generator_state"]):
        raise RuntimeError("采样状态不同")
    if loaded_checkpoint["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(loaded_checkpoint["cuda_rng_state"])
    random.setstate(loaded_checkpoint["python_rng_state"])
    torch.set_rng_state(loaded_checkpoint["rng_state"])
    saved_numpy_state = loaded_checkpoint["numpy_state"]
    numpy_key = np.array(saved_numpy_state[1],dtype=np.uint32)
    restored_numpy_state = (saved_numpy_state[0],numpy_key,saved_numpy_state[2],saved_numpy_state[3],saved_numpy_state[4])
    np.random.set_state(restored_numpy_state)
else:
    epoch_generator_state = generator.get_state()
    batch_consume = 0
    iterator = iter(loader)
for i in range(start_step,total_step):
    try:
        feature , target ,id = next(iterator)
    except StopIteration:
        epoch_generator_state = generator.get_state()
        iterator = iter(loader)
        batch_consume=0
        feature , target ,id = next(iterator)
    feature = feature.to(device)
    target = target.to(device)
    lr = official_lr(i,200000)
    batch_consume+=1
    for para in optimizer.param_groups:
        para["lr"] = lr
    loss = train_batch(model,optimizer,feature,target)
    print(f"{i+1}:",loss)
    numpy_state = np.random.get_state()
    numpy_key = numpy_state[1].tolist()
    saved_numpy_state = (numpy_state[0],numpy_key,numpy_state[2],numpy_state[3],numpy_state[4])
    checkpoint = {
    "model":model.state_dict(),
    "optimizer":optimizer.state_dict(),
    "step":i+1,
    "generator_state":generator.get_state(),
    "batch_consume":batch_consume,
    "epoch_generator_state":epoch_generator_state,
    "rng_state":torch.get_rng_state(),
    "cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    "python_rng_state":random.getstate(),
    "numpy_state":saved_numpy_state
}
    
    if (i+1)%checkpoint_every == 0 or i+1 == total_step:
        torch.save(checkpoint,Path(output_dir / f"checkpoint_step{i+1}.pt"))
        