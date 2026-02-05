import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from config import Config
from dataset_loader import UnderwaterDataset
from models.restormer import Restormer
from metrics import psnr, ssim
from patch_utils import clamp_tensor
from models.perceptual import PerceptualLoss
from ema import EMA


def log_images(writer, inp, out, tar, epoch):

    grid_inp = vutils.make_grid(inp[:4].cpu(), normalize=True)
    grid_out = vutils.make_grid(out[:4].cpu(), normalize=True)
    grid_tar = vutils.make_grid(tar[:4].cpu(), normalize=True)

    writer.add_image("Input", grid_inp, epoch)
    writer.add_image("Output", grid_out, epoch)
    writer.add_image("Target", grid_tar, epoch)

    # Also log side-by-side comparison
    comparison = torch.cat([inp[:4], out[:4], tar[:4]], dim=3)
    grid_comp = vutils.make_grid(comparison.cpu(), normalize=True)

    writer.add_image("Comparison", grid_comp, epoch)


def main():

    device = Config.DEVICE
    print("🚀 Training on:", device)

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    train_dataset = UnderwaterDataset(
        Config.TRAIN_INPUT,
        Config.TRAIN_TARGET,
        training=True
    )

    val_dataset = UnderwaterDataset(
        Config.VAL_INPUT,
        Config.VAL_TARGET,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True
    )

    model = Restormer().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6
    )

    scaler = GradScaler()

    l1_loss = torch.nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)

    ema = EMA(model, decay=0.999)

    writer = SummaryWriter(Config.LOG_DIR)

    start_epoch = 0
    best_psnr = 0

    latest_ckpt = os.path.join(Config.CHECKPOINT_DIR, "latest.pth")
    best_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best.pth")

    if os.path.exists(latest_ckpt):
        print("♻️ Resuming from checkpoint...")
        ckpt = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if "ema" in ckpt:
            ema.shadow = ckpt["ema"]

        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0)

    for epoch in range(start_epoch, Config.EPOCHS):

        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

        total_loss = 0
        total_psnr = 0
        total_ssim = 0

        for step, (inp, tar) in enumerate(loop):

            inp = inp.to(device)
            tar = tar.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):

                out = model(inp)
                out = clamp_tensor(out)

                loss_l1 = l1_loss(out, tar)
                loss_per = perceptual_loss(out, tar)
                loss_ssim = 1 - ssim(out, tar)

                loss = loss_l1 + 0.1 * loss_per + 0.2 * loss_ssim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update()

            p = psnr(out.detach(), tar.detach())
            s = ssim(out.detach(), tar.detach())

            total_loss += loss.item()
            total_psnr += p.item()
            total_ssim += s.item()

            global_step = epoch * len(train_loader) + step

            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/PSNR", p.item(), global_step)
            writer.add_scalar("Train/SSIM", s.item(), global_step)
            writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]['lr'], global_step)

            loop.set_postfix(loss=loss.item(), psnr=p.item())

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)

        writer.add_scalar("Epoch/Train_Loss", avg_loss, epoch)
        writer.add_scalar("Epoch/Train_PSNR", avg_psnr, epoch)
        writer.add_scalar("Epoch/Train_SSIM", avg_ssim, epoch)

        # -------- VALIDATION --------

        ema.apply_shadow()
        model.eval()

        val_psnr_total = 0
        val_ssim_total = 0

        with torch.no_grad():
            # pick a random batch index to visualize
            random_vis_index = torch.randint(0, len(val_loader), (1,)).item()
            for i, (inp, tar) in enumerate(val_loader):

                inp = inp.to(device)
                tar = tar.to(device)

                out = model(inp)
                out = clamp_tensor(out)

                val_psnr_total += psnr(out, tar).item()
                val_ssim_total += ssim(out, tar).item()

                # Log sample images from first validation batch
                if i < 3:   # log first 3 random batches
                   log_images(writer, inp, out, tar, epoch * 10 + i)

        ema.restore()

        val_psnr_avg = val_psnr_total / len(val_loader)
        val_ssim_avg = val_ssim_total / len(val_loader)

        writer.add_scalar("Val/PSNR", val_psnr_avg, epoch)
        writer.add_scalar("Val/SSIM", val_ssim_avg, epoch)

        print(f"Epoch {epoch+1}: ValPSNR={val_psnr_avg:.2f}, ValSSIM={val_ssim_avg:.4f}")

        if val_psnr_avg > best_psnr:
            best_psnr = val_psnr_avg
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.shadow,
                "best_psnr": best_psnr
            }, best_ckpt)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow,
            "best_psnr": best_psnr
        }, latest_ckpt)

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()
