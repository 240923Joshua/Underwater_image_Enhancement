import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset_loader import UnderwaterDataset
from models.restormer import Restormer   # ⚠️ make sure class name is Restormer
import torch.nn.functional as F
import math

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # ===== DATASETS =====
    train_dataset = UnderwaterDataset(
        Config.TRAIN_INPUT, Config.TRAIN_TARGET, training=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # ===== MODEL =====
    model = Restormer().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    criterion = torch.nn.L1Loss()

    scaler = GradScaler()

    writer = SummaryWriter(Config.LOG_DIR)

    # ===== RESUME TRAINING =====
    start_epoch = 0
    latest_ckpt = os.path.join(Config.CHECKPOINT_DIR, "latest.pth")

    if os.path.exists(latest_ckpt):
        print("♻️ Resuming training...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    # ===== TRAIN LOOP =====
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

        total_loss = 0

        optimizer.zero_grad()

        for step, (inp, tar) in enumerate(loop):
            inp = inp.to(device)
            tar = tar.to(device)

            # ===== Forward Pass (AMP) =====
            with autocast("cuda"):
                out = model(inp)
                loss = criterion(out, tar) / Config.GRAD_ACCUM_STEPS

            # ===== Backprop =====
            scaler.scale(loss).backward()

            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ===== Metrics =====
            total_loss += loss.item()
            p = psnr(out, tar)

            # ===== TensorBoard Logging =====
            global_step = epoch * len(train_loader) + step

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("PSNR/train_step", p, global_step)

            if step == 0:
                writer.add_images("Input", inp[:4].cpu(), epoch)
                writer.add_images("Output", out[:4].cpu(), epoch)
                writer.add_images("Target", tar[:4].cpu(), epoch)

            writer.flush()  # 🔥 force write logs



        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        print(f"✅ Epoch {epoch+1} Loss: {avg_loss:.6f}")

        # ===== SAVE CHECKPOINT =====
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, latest_ckpt)

        if (epoch + 1) % Config.SAVE_EVERY == 0:
            torch.save(model.state_dict(),
                       os.path.join(Config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))

    writer.close()


if __name__ == "__main__":
    main()
