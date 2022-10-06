from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiDSprites(Dataset):
    def __init__(self, path_to_dataset: Path):
        data = np.load(path_to_dataset)
        self.masks = data['masks']
        self.images = data['images']
        self.visibility = data['visibility']
        self.image_size = self.images[0].shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image).float() / 255
        return image
