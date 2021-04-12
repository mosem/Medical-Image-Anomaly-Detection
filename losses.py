import torch
import torch.nn as nn


class CompactnessLoss(nn.Module):
    def __init__(self):
        super(CompactnessLoss, self).__init__()

    def forward(self, inputs):
        n = inputs.size(0)  # batch size
        m = inputs.size(1)  # feature size
        repeated_mean = torch.sum(inputs, dim=0).repeat(n, 1)
        vectorized_means = (repeated_mean - inputs) / (n-1)
        variances = (inputs - vectorized_means).norm(dim=1).pow(2) / m
        return variances.mean()