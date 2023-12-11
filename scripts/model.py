import torch
import torch.nn as nn
import torchvision

def get_res50():
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, 1)
    return model