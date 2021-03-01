import os
import sys
import glob
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CycleGANDataset(Dataset):

    def __init__(self, dataset_name=None, transform=None, unaligned=True, mode='train'):


        self.transform = transform
        self.unaligned = unaligned
        self.mode = mode

        dataset_dir = os.path.join('dataset', dataset_name)

        self.files_A = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}A/*.*")))
        self.files_B = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}B/*.*")))

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index]))


        if self.mode == 'test':
            item_B = self.transform(Image.open(self.files_B[index]))
        else:
            if self.unaligned:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))


        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
