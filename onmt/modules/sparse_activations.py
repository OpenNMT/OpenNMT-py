import torch
from entmax import Entmax15, Sparsemax


class LogSparsemax(Sparsemax):
    def forward(self, *args, **kwargs):
        return torch.log(super().forward(*args, **kwargs))


class LogEntmax15(Entmax15):
    def forward(self, *args, **kwargs):
        return torch.log(super().forward(*args, **kwargs))
