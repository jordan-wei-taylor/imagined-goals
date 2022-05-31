from   rlig.base      import Base
from   rlig.pytorch.base import kaiming_init, build_sequential
from   torch.autograd import Variable
from   torch          import nn
from   torch.nn       import functional as F

import torch


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld# , dimension_wise_kld, mean_kld
    
class BetaVAE(Base, nn.Module):

    def __init__(self, encoder, decoder, beta = 1):
        if not isinstance(encoder, nn.Sequential):
            encoder = build_sequential(*encoder)

        if not isinstance(decoder, nn.Sequential):
            decoder = build_sequential(*decoder)

        Base.__init__(self, locals())
        # nn.Module.__init__(self)

        self.z_dim = encoder[-1].out_features // 2 # mu, logvar

        self.weight_init()

        self.optim = torch.optim.Adam(self.parameters(), lr = 1e-4)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu            = distributions[:, :self.z_dim]
        logvar        = distributions[:, self.z_dim:]
        z             = reparametrize(mu, logvar)
        x_recon       = self._decode(z)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def mean_encode(self, x):
        return self._encode(x)[:,:self.z_dim]

    def _decode(self, z):
        return self.decoder(z)

    def _fit(self, X, distribution = None):

        X_hat, mu, logvar = self(X)

        loss = reconstruction_loss(X, X_hat, distribution) + self.beta * kl_divergence(mu, logvar)

        self.optim.zero_grad()

        loss.backward()

        self.optim.step()

    def fit(self, X, epochs = 1, distribution = None):
        for epoch in range(epochs):
            self._fit(X, distribution)
        return self

