import torch
continuous_checkpoint =torch.load("/mnt/d/Project/IR-AutoResearch/results/resume_learning_continue/checkpoint_step4.pt",map_location="cpu")
resumed_checkpoint = torch.load("/mnt/d/Project/IR-AutoResearch/results/resume_learning_resumed/checkpoint_step4.pt",map_location="cpu")
for name in continuous_checkpoint["model"]:
    if (not torch.equal(continuous_checkpoint["model"][name],resumed_checkpoint["model"][name])):
        print(name)