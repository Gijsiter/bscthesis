import torch
from torch import nn


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            activation,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, input):
        out = self.block(input)
        return out
