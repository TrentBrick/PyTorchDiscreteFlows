"""
Various helper network modules. Taken from https://github.com/karpathy/pytorch-normalizing-flows
"""

import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
            #nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

