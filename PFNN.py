import torch
import torch.nn as nn
import numpy as np


class PFNN(nn.Module):
    def __init__(self, size_input, size_output, size_hidden=512, dropout=0.7):
        super().__init__()
        self.dropout0 = nn.Dropout(p=dropout)

        self.lin0a = nn.Linear(size_input - 1, size_hidden)
        self.lin0b = nn.Linear(size_input - 1, size_hidden)
        self.lin0c = nn.Linear(size_input - 1, size_hidden)
        self.lin0d = nn.Linear(size_input - 1, size_hidden)
        self.lin0 = [self.lin0a, self.lin0b, self.lin0c, self.lin0d]

        self.dropout1 = nn.Dropout(p=dropout)
        self.lin1a = nn.Linear(size_hidden, size_hidden)
        self.lin1b = nn.Linear(size_hidden, size_hidden)
        self.lin1c = nn.Linear(size_hidden, size_hidden)
        self.lin1d = nn.Linear(size_hidden, size_hidden)
        self.lin1 = [self.lin1a, self.lin1b, self.lin1c, self.lin1d]

        self.dropout2 = nn.Dropout(p=dropout)
        self.lin2a = nn.Linear(size_hidden, size_output)
        self.lin2b = nn.Linear(size_hidden, size_output)
        self.lin2c = nn.Linear(size_hidden, size_output)
        self.lin2d = nn.Linear(size_hidden, size_output)
        self.lin2 = [self.lin2a, self.lin2b, self.lin2c, self.lin2d]

    def forward(self, x):
        # phase = x[:, 0] * 4
        # pindex_1 = torch.floor(phase) % 4
        # pindex_0 = (pindex_1 - 1) % 4
        # pindex_2 = (pindex_1 + 1) % 4
        # pindex_3 = (pindex_1 + 2) % 4
        #
        # w = phase % 1.0
        # w = w.view([-1, 1])
        #
        # def cubic(y0, y1, y2, y3, mu):
        #     return (
        #             (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
        #             (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
        #             (-0.5 * y0 + 0.5 * y2) * mu +
        #             y1)
        #
        # wX = -0.5 * w + w * w - 0.5 * w * w * w
        # wY = 1 - 2.5 * w * w + 1.5 * w * w * w
        # wZ = 0.5 * w + 2 * w * w - 1.5 * w * w * w
        # wW = -0.5 * w * w + 0.5 * w * w * w
        #
        # x0 = cubic(self.lin0[pindex_0](x[:, 1:]),
        #            self.lin0[pindex_1](x[:, 1:]),
        #            self.lin0[pindex_2](x[:, 1:]),
        #            self.lin0[pindex_3](x[:, 1:]), w)
        #
        # x0 = self.dropout0(x0)
        # x0 = torch.relu(x0)
        #
        # x1 = cubic(self.lin1[pindex_0](x0),
        #            self.lin1[pindex_1](x0),
        #            self.lin1[pindex_2](x0),
        #            self.lin1[pindex_3](x0), w)
        # x1 = torch.relu(x0)
        # x2 = cubic(self.lin2[pindex_0](x1),
        #            self.lin2[pindex_1](x1),
        #            self.lin2[pindex_2](x1),
        #            self.lin2[pindex_3](x1), w)
        #
        # return x2

        # old way - prof.umetani
        phase = x[:, 0] * 4
        index = torch.floor(phase)
        w = phase - index
        wX = -0.5 * w + w * w - 0.5 * w * w * w
        wY = 1 - 2.5 * w * w + 1.5 * w * w * w
        wZ = 0.5 * w + 2 * w * w - 1.5 * w * w * w
        wW = -0.5 * w * w + 0.5 * w * w * w
        wa = torch.heaviside(-torch.abs(index - 0.), wX) + \
             torch.heaviside(-torch.abs(index - 1.), wY) + \
             torch.heaviside(-torch.abs(index - 2.), wZ) + \
             torch.heaviside(-torch.abs(index - 3.), wW)
        wb = torch.heaviside(-torch.abs(index - 1.), wX) + \
             torch.heaviside(-torch.abs(index - 2.), wY) + \
             torch.heaviside(-torch.abs(index - 3.), wZ) + \
             torch.heaviside(-torch.abs(index - 0.), wW)
        wc = torch.heaviside(-torch.abs(index - 2.), wX) + \
             torch.heaviside(-torch.abs(index - 3.), wY) + \
             torch.heaviside(-torch.abs(index - 0.), wZ) + \
             torch.heaviside(-torch.abs(index - 1.), wW)
        wd = torch.heaviside(-torch.abs(index - 3.), wX) + \
             torch.heaviside(-torch.abs(index - 0.), wY) + \
             torch.heaviside(-torch.abs(index - 1.), wZ) + \
             torch.heaviside(-torch.abs(index - 2.), wW)
        wa = wa.view([-1, 1])
        wb = wb.view([-1, 1])
        wc = wc.view([-1, 1])
        wd = wd.view([-1, 1])
        #        print(wa+wb+wc+wd)
        #
        x0 = self.lin0a(x[:, 1:]) * wa + \
             self.lin0b(x[:, 1:]) * wb + \
             self.lin0c(x[:, 1:]) * wc + \
             self.lin0d(x[:, 1:]) * wd
        x0 = self.dropout0(x0)
        x0 = torch.relu(x0)
        x1 = self.lin1a(x0) * wa + \
             self.lin1b(x0) * wb + \
             self.lin1c(x0) * wc + \
             self.lin1d(x0) * wd
        x1 = torch.relu(x1)
        x2 = self.lin2a(x1) * wa + \
             self.lin2b(x1) * wb + \
             self.lin2c(x1) * wc + \
             self.lin2d(x1) * wd
        return x2


def mse_loss(input, target):
    return ((input - target) ** 2).mean()
