import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features, p = .2):
        super.__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=in_features),
            nn.Dropout(p=p),
            nn.ReLU(), 
            nn.Linear(in_features=in_features, out_features=4)
        )

    def forward(self, x):
        y = self.model(x)
        return y
