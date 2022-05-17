from copy import deepcopy

from torch import nn
from torch.autograd import Variable


class EWC(object):
    """
    Implementation of EWC based on the origianl paper
            Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks."
            Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.
    """

    def __init__(self, model, criterion):

        self.model = model
        self.criterion = criterion

        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        self.precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.precision_matrices[n] = Variable(p.data).cuda()
        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data).cuda()

    def update(self, model, t, loader):
        self.model = model
        self.model.eval()
        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data).cuda()
        for x, y in loader:
            self.model.zero_grad()
            x = Variable(x).cuda()
            output = self.model(x, t)
            loss = self.criterion(output, y.cuda())
            loss.backward()
            for n, p in self.model.named_parameters():
                self.precision_matrices[n].data = (
                    self.precision_matrices[n].data + (p.grad.data ** 2) * t
                ) / (t + 1)
            break

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
