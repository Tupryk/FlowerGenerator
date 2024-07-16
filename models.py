import torch
import torch.nn as nn

IMAGE_DIMS = 64
LATENT_DIM = 128

class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int=LATENT_DIM):
        super().__init__()
        input_size = IMAGE_DIMS**2 * 3
        layers = []
        layers.append(nn.Linear(input_size, 1024))
        layers.append(nn.Linear(1024, 512))
        layers.append(nn.Linear(512, latent_dim))
        layers.append(nn.Linear(latent_dim, 512))
        layers.append(nn.Linear(512, 1024))
        layers.append(nn.Linear(1024, input_size))
        self.h_layer_count = int(len(layers)*.5)
        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.linears[:-1]:
            x = nn.LeakyReLU()(l(x))
        return nn.Sigmoid()(self.linears[-1](x))

    def encode(self, x):
        for i in range(self.h_layer_count):
            x = nn.LeakyReLU()(self.linears[i](x))
        return x

    def dencode(self, x):
        for i in range(self.h_layer_count-1):
            x = nn.LeakyReLU()(self.linears[i+self.h_layer_count](x))
        return nn.Sigmoid()(self.linears[-1](x))


class Diffuser(nn.Module):
    def __init__(self, nhidden: int=1024):
        super().__init__()
        layers = [nn.Linear(LATENT_DIM+1, nhidden)]
        for _ in range(3):
            layers.append(nn.Linear(nhidden, nhidden))
        layers.append(nn.Linear(nhidden, LATENT_DIM))
        self.linears = nn.ModuleList(layers)

    def forward(self, x, t):
        x = torch.concat([x, t], axis=-1)
        for l in self.linears[:-1]:
            x = nn.ReLU()(l(x))
        return self.linears[-1](x)
    
    def genrate():
        return [0]
