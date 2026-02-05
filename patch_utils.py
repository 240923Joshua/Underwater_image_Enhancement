import random
import torch

def random_crop(inp, tar, patch_size):
    _, h, w = inp.shape

    if h < patch_size or w < patch_size:
        return inp, tar

    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)

    inp_patch = inp[:, y:y+patch_size, x:x+patch_size]
    tar_patch = tar[:, y:y+patch_size, x:x+patch_size]

    return inp_patch, tar_patch
