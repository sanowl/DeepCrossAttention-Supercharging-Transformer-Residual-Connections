import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def create_optimizer_and_scheduler(model, warmup_steps=1000, total_steps=500000):
    """Creates optimizer and learning rate scheduler"""
    optimizer = AdamW(
        model.parameters(),
        lr=0.0016,
        betas=(0.9, 0.98),
        weight_decay=0.1
    )
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return (warmup_steps / float(max(1, step))) ** 0.5
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
