import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class RandomDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):

        patches = []
        letters = []

        for _ in range(11):
            val = torch.randn((3, 25,25))
            patches.append(val)

        for _ in range(111):
            val = torch.randn((3, 25, 25))
            letters.append(val)

        return patches, letters, torch.randint(0,299,(1,))