import torch
import torch.nn as nn
import torch.nn.functional as F


class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.DEBUG = False
        self.fc = nn.Sequential(
            nn.Conv3d(1, 16, (1,1,8)),
            nn.ReLU(),
            nn.MaxPool3d((1,1,2)),
            nn.Conv3d(16, 32, (2, 2, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 1, 2)),
            nn.Flatten(),
            nn.Linear(27136, 17)
        )

    def forward(self, x):
        x = self.fc(x)
        if self.DEBUG:
            print(x.shape)
            exit(0)
        return F.log_softmax(x, dim=1)
