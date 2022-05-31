import torch
from   torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def build_sequential(*neurons, activate_final = False):
    layers = []
    for n_in, n_out in zip(neurons[:-2], neurons[1:-1]):
        layers.append(nn.Linear(n_in, n_out))
        layers.append(nn.ReLU(True))

    n_in, n_out = neurons[-2:]

    layers.append(nn.Linear(n_in, n_out))

    if activate_final:
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)
