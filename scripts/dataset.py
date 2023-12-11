import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import random

class MayoData(Dataset):
    def __init__(self, csv_file: pd.DataFrame, img_size : int) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.img_paths = csv_file['updated_paths'].values.tolist()
        self.targets = csv_file['target'].values.tolist()
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, index) -> tuple:
        img = torchvision.io.read_file(self.img_paths[index])
        img = torchvision.io.decode_jpeg(img)
        img = torchvision.transforms.functional.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        target = torch.Tensor(np.array(self.targets[index]).astype(np.float32)).float()
        return img, target


def train_loader(csv_file: pd.DataFrame, img_size : int, batch_size : int = 32, shuffle : bool = True, num_workers : int = 8, return_dataset : bool = False) -> DataLoader:
    dataset = MayoData(csv_file, img_size)
    loader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    if return_dataset:
        return dataset, loader
    return loader

def val_loader(csv_file: pd.DataFrame, img_size : int, batch_size : int = 32, shuffle : bool = True, num_workers : int = 8, return_dataset : bool = False) -> DataLoader:
    dataset = MayoData(csv_file, img_size)
    loader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    if return_dataset:
        return dataset, loader
    return loader

def test_loader(csv_file: pd.DataFrame, img_size : int, batch_size : int = 32, shuffle : bool = False, num_workers : int = 8, return_dataset : bool = False) -> DataLoader:
    dataset = MayoData(csv_file, img_size)
    loader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    if return_dataset:
        return dataset, loader
    return loader


def plot_images(dataset: Dataset, thumb_size: int = 64, cols: int = 10, rows : int = 3) -> None:
    mosaic = Image.new(
        mode = 'RGB',
        size = (thumb_size * cols + (cols-1), thumb_size * rows + (rows-1))
    )
    for x in range(30):
        ix = x % cols
        iy = x // cols
        id = random.randint(0, 400)
        img, label = dataset.__getitem__(id)
        img = (np.clip((img.permute(1,2,0) * 255).numpy(), 0, 255)).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=Image.BILINEAR)
        mosaic.paste(img, (ix * thumb_size + ix, iy * thumb_size + iy))
    display(mosaic)