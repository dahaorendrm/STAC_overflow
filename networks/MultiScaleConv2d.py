import torch.nn as nn
import torch

class MultiScaleConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Iterable[int] = (3, 5, 7)):
        # iterate the list and create a ModuleList of single Conv1d blocks
        self.models = nn.ModuleList()
        for k in kernel_sizes:
            self.models.append(nn.Conv2d(in_channels, out_channels, k, padding=(k-1)/2))

    def forward(self, inputs):
        # now you can build a single output from the list of convs
        out = [module(inputs) for module in self.models]
        # you can return a list, or even build a single tensor, like so
        return torch.cat(out, dim=1)
