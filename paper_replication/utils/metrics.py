import torch

def accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()
