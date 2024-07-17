import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torchvision import transforms


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", augment: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        ##new
        self.augment = augment
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)
    #insert 
    def augment_data(self, data):
        # noise
        noise = torch.randn_like(data) * 0.01
        data += noise
        # shift
        shift = torch.randint(-10, 10, (1,)).item()
        data = torch.roll(data, shifts=shift, dims=1)
        # scaling
        scale = 1.0 + 0.1 * (torch.rand(1).item() - 0.5)
        data *= scale
        # jitter
        jitter = torch.randn_like(data) * 0.01
        data += jitter
        return data
    
    #To training dataset
    def __getitem__(self, i):
        if hasattr(self, "y"):
            data, label, subject_idx = self.X[i], self.y[i], self.subject_idxs[i]
            if self.augment and self.split == 'train':
                data = self.augment_data(data)
            return data, label, subject_idx
        else:
            data, subject_idx = self.X[i], self.subject_idxs[i]
            if self.augment and self.split == 'train':
                data = self.augment_data(data)
            return data, subject_idx
    """
    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
    """
       
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
