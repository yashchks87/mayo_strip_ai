import torch
import torch.nn as nn

def bce(inputs, targets):
    sigmoid = nn.Sigmoid()
    inputs = sigmoid(inputs)
    # return torch.sum(- (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))) / inputs.shape[0]
    return -(inputs.log()*targets + (1-targets)*(1-inputs).log()).mean()

