import os
import cv2
import torch
from torch.utils.data import Dataset
from patch_utils import random_crop_pair
from config import Config

class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir=None, training=True):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.training = training

        self.images = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0   # ✅ strict normalization
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __getitem__(self, idx):
        name = self.images[idx]

        inp_path = os.path.join(self.input_dir, name)
        inp = self._load_image(inp_path)

        if self.target_dir is not None:
            tar_path = os.path.join(self.target_dir, name)
            tar = self._load_image(tar_path)
        else:
            tar = inp.clone()

        if self.training and Config.PATCH_SIZE > 0:
            inp, tar = random_crop_pair(inp, tar, Config.PATCH_SIZE)

        return inp, tar
