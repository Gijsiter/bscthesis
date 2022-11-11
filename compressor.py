import torch
from torch import nn


class EncodingCompressor(nn.Module):
    def __init__(self, in_features, out_features, activation, mode):
        super().__init__()
        self.net = [
            # L1
            nn.Linear(in_features, out_features),
            activation,
            # L2
            nn.Linear(out_features, out_features),
            activation,
            # TL
            nn.Linear(out_features, out_features)
        ]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        out = self.net(input)
        return out