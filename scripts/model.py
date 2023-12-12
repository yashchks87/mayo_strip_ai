import torch
import torch.nn as nn
import torchvision

def get_res50():
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, 1)
    return model

def freeze_layers(model):
    for name, weight in model.named_parameters():
        if 'fc' not in name:
            weight.requires_grad = False
    return model

