import torch
import torch.nn as nn
import torch.nn.functional as F


class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv3d(1, 5, (2,2,10)),
            nn.MaxPool3d((1,1,5)),
            nn.Conv3d(5, 25, (2, 2, 5)),
            nn.MaxPool3d((1,1,2)),
            nn.Flatten(),
            nn.Linear(4275, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 17)
        )

    def forward(self, x):
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        #exit(0)
        return F.log_softmax(x, dim=1)
