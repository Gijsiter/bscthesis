import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class MappingNet(nn.Module):
    def __init__(self, in_features, hidden_features, dec_hidden=3, nfeatures=256):
        super().__init__()
 
        lyr_fts = [in_features] + hidden_features
        self.net = nn.Sequential(
            *sum(
                [[nn.Linear(lyr_fts[i], lyr_fts[i+1]), nn.ReLU()]
                 for i in range(len(lyr_fts)-1)], []
            )
        )

        self.to_gamma = nn.Linear(lyr_fts[-1], nfeatures)
        self.to_beta = nn.Linear(lyr_fts[-1], nfeatures)



    def forward(self, input):
        out = self.net(input)

        gamma = self.to_gamma(out)
        beta = self.to_beta(out)

        return gamma, beta