import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
from patch_dataset import PatchDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train():
    num_epochs = 3
    model = Machine()
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pd = PatchDataset(is_train=True)
    dataloader = DataLoader(pd, batch_size=50, shuffle=True)
    for epoch in range(3):
        for x,y in dataloader:
            optimiser.zero_grad()
            y_pred = model(x)
            loss = F.nll_loss(y_pred, y)
            print(f"Epoch {num_epochs}, Loss {loss.item()}")
            loss.backward()
            optimiser.step()

    torch.save(model, "models/machine.h5")


if __name__ == "__main__":
    train()
