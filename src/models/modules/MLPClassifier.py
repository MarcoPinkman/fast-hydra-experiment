import torch.nn as nn
class MLPClassifier(nn.Module):
    def __init__(self, hparams:dict):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hparams['x_dim'] + hparams['attr_dim'], 2000),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2000, 1200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, 1200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, hparams['out_dim']),
        )

    def forward(self, x):
        return self.model(x)