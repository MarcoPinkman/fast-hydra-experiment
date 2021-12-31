import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hparams: dict):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hparams['z_dim'] + hparams['attr_dim'], 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)