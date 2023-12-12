import torch
import torch.nn as nn

def precision(inputs, targets, thres = 0.5):
    sigmoid = nn.Sigmoid()
    inputs = sigmoid(inputs)
    inputs = torch.where(inputs > thres, 1.0, 0.0)
    tp = torch.sum(torch.logical_and(inputs == 1.0, targets == 1.0))
    fp = torch.sum(torch.logical_and(inputs == 0.0, targets == 1.0))
    if tp + fp != 0.0:
        return tp / (tp + fp)
    return torch.Tensor([0.0])

def recall(inputs, targets, thres = 0.5):
    sigmoid = nn.Sigmoid()
    inputs = sigmoid(inputs)
    inputs = torch.where(inputs > thres, 1.0, 0.0)
    tp = torch.sum(torch.logical_and(inputs == 1.0, targets == 1.0))
    fn = torch.sum(torch.logical_and(inputs == 1.0, targets == 0.0))
    if tp + fn != 0.0:
        return tp / (tp + fn)
    return torch.Tensor([0.0])