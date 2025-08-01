import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features, p = .2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=150),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=150, out_features=80),
            nn.Dropout(p=.4),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=4)
        )

    def forward(self, x):
        y = self.model(x)
        return y
