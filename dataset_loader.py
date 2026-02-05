import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from patch_utils import random_crop
from config import Config


class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir=None, training=True):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.training = training

        self.images = sorted(os.listdir(input_dir))

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        inp_path = os.path.join(self.input_dir, name)
        tar_path = os.path.join(self.target_dir, name) if self.target_dir else None

        inp = Image.open(inp_path).convert("RGB")
        inp = self.transform(inp)

        if tar_path:
            tar = Image.open(tar_path).convert("RGB")
            tar = self.transform(tar)

            # ✅ PATCH TRAINING
            if self.training:
                inp, tar = random_crop(inp, tar, Config.PATCH_SIZE)

            return inp, tar

        return inp
