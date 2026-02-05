import os
import torch
import cv2
from models.restormer import Restormer
from config import Config
from dataset_loader import UnderwaterDataset
from patch_utils import clamp_tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Restormer().to(device)
    ckpt = torch.load(os.path.join(Config.CHECKPOINT_DIR, "latest.pth"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    dataset = UnderwaterDataset(Config.TEST_INPUT, None, training=False)

    with torch.no_grad():
        for i, (inp, _) in enumerate(dataset):
            inp = inp.unsqueeze(0).to(device)

            out = model(inp)
            out = clamp_tensor(out)

            out_img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = (out_img * 255).astype("uint8")

            cv2.imwrite(os.path.join(Config.RESULT_DIR, f"{i}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print("✅ Testing done")

if __name__ == "__main__":
    main()
