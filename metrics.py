import torch
import torch.nn.functional as F
import math

def psnr(pred, target, eps=1e-8):
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    mse = F.mse_loss(pred, target)
    if mse < eps or torch.isnan(mse):
        return torch.tensor(0.0, device=pred.device)

    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(pred, target, eps=1e-8):
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    mu_x = pred.mean()
    mu_y = target.mean()

    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2 + eps)

    val = num / den

    if torch.isnan(val):
        return torch.tensor(0.0, device=pred.device)

    return val
