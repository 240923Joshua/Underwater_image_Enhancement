import random
import torch

def random_crop_pair(inp, tar, patch_size):
    _, h, w = inp.shape

    if h < patch_size or w < patch_size:
        return inp, tar

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    inp_patch = inp[:, top:top+patch_size, left:left+patch_size]
    tar_patch = tar[:, top:top+patch_size, left:left+patch_size]

    return inp_patch, tar_patch


def center_crop_pair(inp, tar, patch_size):
    _, h, w = inp.shape
    top = max((h - patch_size) // 2, 0)
    left = max((w - patch_size) // 2, 0)

    return inp[:, top:top+patch_size, left:left+patch_size], \
           tar[:, top:top+patch_size, left:left+patch_size]


def clamp_tensor(x):
    return torch.clamp(x, 0.0, 1.0)
