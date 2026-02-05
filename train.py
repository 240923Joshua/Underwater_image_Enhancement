import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset_loader import UnderwaterDataset
from models.restormer import Restormer
from metrics import psnr, ssim
from patch_utils import clamp_tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Device:", device)

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
        shuffle=False
    )

    model = Restormer().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    criterion = torch.nn.L1Loss()
    scaler = GradScaler()

    writer = SummaryWriter(Config.LOG_DIR)

    start_epoch = 0
    latest_ckpt = os.path.join(Config.CHECKPOINT_DIR, "latest.pth")

    if os.path.exists(latest_ckpt):
        print("♻️ Resuming training...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

        total_loss = 0
        total_psnr = 0

        optimizer.zero_grad()

        for step, (inp, tar) in enumerate(loop):
            inp = inp.to(device)
            tar = tar.to(device)

            with autocast("cuda"):
                out = model(inp)
                out = clamp_tensor(out)   # ✅ prevent NaN explosion
                loss = criterion(out, tar) / Config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            p = psnr(out.detach(), tar.detach())

            total_loss += loss.item()
            total_psnr += p.item()

            global_step = epoch * len(train_loader) + step

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("PSNR/train_step", p.item(), global_step)

            if step == 0:
                writer.add_images("Train/Input", inp[:4].cpu(), epoch)
                writer.add_images("Train/Output", out[:4].cpu(), epoch)
                writer.add_images("Train/Target", tar[:4].cpu(), epoch)

            loop.set_postfix(loss=loss.item(), psnr=p.item())

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)

        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("PSNR/train_epoch", avg_psnr, epoch)

        # ===== VALIDATION (NO AMP → IMPORTANT) =====
        model.eval()
        val_psnr_total = 0
        val_ssim_total = 0

        with torch.no_grad():
            for i, (inp, tar) in enumerate(val_loader):
                inp = inp.to(device)
                tar = tar.to(device)

                out = model(inp)
                out = clamp_tensor(out)

                p = psnr(out, tar)
                s = ssim(out, tar)

                val_psnr_total += p.item()
                val_ssim_total += s.item()

                if i == 0:
                    writer.add_images("Val/Input", inp.cpu(), epoch)
                    writer.add_images("Val/Output", out.cpu(), epoch)
                    writer.add_images("Val/Target", tar.cpu(), epoch)

        val_psnr_avg = val_psnr_total / len(val_loader)
        val_ssim_avg = val_ssim_total / len(val_loader)

        writer.add_scalar("PSNR/val", val_psnr_avg, epoch)
        writer.add_scalar("SSIM/val", val_ssim_avg, epoch)

        print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={avg_psnr:.2f}, ValPSNR={val_psnr_avg:.2f}, ValSSIM={val_ssim_avg:.4f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, latest_ckpt)

    writer.close()

if __name__ == "__main__":
    main()
