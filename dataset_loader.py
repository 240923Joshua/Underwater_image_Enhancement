import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random


class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir, training=True):

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.training = training

        self.input_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
        ])

        self.target_paths = sorted([
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
        ])

        assert len(self.input_paths) == len(self.target_paths), \
            "Input and target count mismatch!"

        self.to_tensor = T.ToTensor()

    def random_crop(self, inp, tar, size=128):

        w, h = inp.size

        if w < size or h < size:
            inp = inp.resize((size, size))
            tar = tar.resize((size, size))
            return inp, tar

        x = random.randint(0, w - size)
        y = random.randint(0, h - size)

        inp = inp.crop((x, y, x + size, y + size))
        tar = tar.crop((x, y, x + size, y + size))

        return inp, tar

    def augment(self, inp, tar):

        # Random horizontal flip
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
            tar = tar.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() < 0.5:
            inp = inp.transpose(Image.FLIP_TOP_BOTTOM)
            tar = tar.transpose(Image.FLIP_TOP_BOTTOM)

        # Random 90 degree rotation
        if random.random() < 0.5:
            inp = inp.rotate(90)
            tar = tar.rotate(90)

        return inp, tar

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):

        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tar = Image.open(self.target_paths[idx]).convert("RGB")

        if self.training:

            # Augmentations
            inp, tar = self.augment(inp, tar)

            # Random patch cropping
            inp, tar = self.random_crop(inp, tar, size=128)

        inp = self.to_tensor(inp)
        tar = self.to_tensor(tar)

        return inp, tar
