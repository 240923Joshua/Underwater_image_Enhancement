import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from dataset_loader import UnderwaterDataset
from models.restormer import Restormer

def main():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on device: {device}")

    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    model = Restormer().to(device)
    ckpt = sorted(os.listdir(Config.CHECKPOINT_DIR))[-1]
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, ckpt)))
    model.eval()

    test_dataset = UnderwaterDataset(Config.TEST_INPUT)
    
    for i, (inp, _) in enumerate(tqdm(test_dataset, desc="Testing")):
        inp = inp.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)

        save_path = os.path.join(Config.RESULT_DIR, f"result_{i}.png")
        save_image(out, save_path)

    print("✅ Testing finished!")

if __name__ == "__main__":
    main()
