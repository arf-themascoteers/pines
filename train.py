import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
from patch_dataset import PatchDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train(device):
    num_epochs = 10
    batch_size = 100
    model = Machine()
    model.to(device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pd = PatchDataset(is_train=True)
    dataloader = DataLoader(pd, batch_size=batch_size, shuffle=True)
    batch_total =  int(len(pd) / batch_size + 1)
    for epoch in range(num_epochs):
        batch = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimiser.zero_grad()
            y_pred = model(x)
            loss = F.nll_loss(y_pred, y)
            print(f"Epoch {epoch + 1} (of {num_epochs}), Batch {batch + 1} (of {batch_total}), Loss {loss.item()}")
            loss.backward()
            optimiser.step()
            batch = batch+1

    torch.save(model, "models/machine.h5")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)
