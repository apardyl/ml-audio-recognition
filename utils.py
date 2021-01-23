import torch
from torch import nn


def save_train_state(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, best_score: float,
                     file_path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
    }, file_path)


def load_train_state(file_path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    if 'scheduler' in data:
        scheduler.load_state_dict(data['scheduler'])
    return data['epoch'], data.get('best_score', 0)


def load_model_state(file_path, model: nn.Module):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
