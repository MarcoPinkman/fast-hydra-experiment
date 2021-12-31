import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hparams:dict):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hparams['x_dim'] + hparams['attr_dim'], 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)