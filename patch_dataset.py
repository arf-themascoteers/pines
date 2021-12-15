import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from scipy import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PatchDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.WINDOW_SIZE = 5
        self.LENGTH = 145
        self.N_BANDS = 220
        self.N_PATCHES = int(145**2 / 5**2)
        cube = io.loadmat('data/Indian_pines.mat')["indian_pines"].astype(float)
        gt = io.loadmat('data/Indian_pines_gt.mat')["indian_pines_gt"].astype(float)

        mean = np.mean(cube)
        std = np.std(cube)
        cube = (cube-mean)/std

        patches = self.__create_patches__(cube)
        labels = self.__create_labels__(patches, gt)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(patches, labels, test_size=0.2, random_state=11)
        self.x = x_train
        self.y = y_train
        if not self.is_train:
            self.x = x_test
            self.y = y_test

        self.x = self.x.reshape(self.x.shape[0],1,self.x.shape[1],self.x.shape[2],self.x.shape[3])

    def __create_patches__(self, cube):
        patches = torch.zeros(self.N_PATCHES, self.WINDOW_SIZE , self.WINDOW_SIZE, self.N_BANDS)
        patch_no = 0
        for row in range(0,self.LENGTH, self.WINDOW_SIZE):
            for col in range(0, self.LENGTH, self.WINDOW_SIZE):
                patches[patch_no] = torch.tensor(cube[row:row+self.WINDOW_SIZE, col:col+self.WINDOW_SIZE])
                patch_no += 1
        return patches

    def __create_labels__(self, patches, gt):
        y = torch.zeros(len(patches), dtype=torch.int64)
        gt_patches = self.__create_gt_patches__(gt)
        for index, patch in enumerate(patches):
            label = self.__get_dominant_label__(gt_patches[index])
            y[index] = label
        return y

    def __get_dominant_label__(self, gt_patch):
        return gt_patch.reshape(-1).bincount().argmax()

    def __create_gt_patches__(self, gt):
        patches = torch.zeros(self.N_PATCHES, self.WINDOW_SIZE, self.WINDOW_SIZE, dtype=torch.int8)
        patch_no = 0
        for row in range(0,self.LENGTH, self.WINDOW_SIZE):
            for col in range(0, self.LENGTH, self.WINDOW_SIZE):
                patches[patch_no] = torch.tensor(gt[row:row+self.WINDOW_SIZE, col:col+self.WINDOW_SIZE])
                patch_no += 1
        return patches

    def show_patch(self, patch):
        x = patch.mean(dim=2)
        plt.imshow(x)
        plt.show()
        print()

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    pd = PatchDataset()
    dataloader = DataLoader(pd, batch_size=50, shuffle=True)
    for x, y in dataloader:
        pd.show_patch(x[0][0])
        print(y[0])
        exit(0)
