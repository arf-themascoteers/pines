import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
import patch_dataset
import os
import train
import plotter
from patch_dataset import PatchDataset
from torch.utils.data import DataLoader


def test(device):
    model = Machine()
    #if not os.path.isfile("models/machine.h5"):
    train.train(device)
    model = torch.load("models/machine.h5")
    model.to(device)
    correct = 0
    total = 0
    pd = PatchDataset(is_train=False)
    dataloader = DataLoader(pd, batch_size=200, shuffle=False)
    for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        pred = torch.argmax(y_pred, dim=1, keepdim=True)
        correct += pred.eq(y.data.view_as(pred)).sum()
        total += x.shape[0]
        print(f'Total:{total}, Correct:{correct}, Accuracy:{correct / total * 100:.2f}')



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
