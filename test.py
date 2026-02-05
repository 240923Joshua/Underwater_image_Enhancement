import os
import torch
from PIL import Image
import torchvision.transforms as T

from config import Config
from models.restormer import Restormer
from patch_utils import split_patches, merge_patches, clamp_tensor


def load_model():

    device = Config.DEVICE

    model = Restormer().to(device)

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best.pth")

    if not os.path.exists(ckpt_path):
        raise Exception("No trained model found!")

    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, device


def enhance_image(model, img_tensor, device):

    img_tensor = img_tensor.to(device)

    # Test-time self-ensemble
    with torch.no_grad():

        # Normal inference
        out1 = model(img_tensor)

        # Horizontal flip inference
        flipped = torch.flip(img_tensor, [3])
        out2 = model(flipped)
        out2 = torch.flip(out2, [3])

        # Average results
        out = (out1 + out2) / 2.0

    return clamp_tensor(out)


def process_image(path, model, device):

    img = Image.open(path).convert("RGB")

    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    tensor = to_tensor(img).unsqueeze(0)

    # Split into overlapping patches
    patches, info = split_patches(tensor, patch_size=256, overlap=32)

    enhanced = []

    for p in patches:
        out = enhance_image(model, p, device)
        enhanced.append(out.cpu())

    final = merge_patches(enhanced, info)

    final = final.squeeze(0)

    return to_pil(final)


def main():

    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    model, device = load_model()

    input_dir = Config.TEST_INPUT
    output_dir = Config.RESULT_DIR

    print("Starting Enhancement...")

    for img_name in os.listdir(input_dir):

        img_path = os.path.join(input_dir, img_name)

        try:
            result = process_image(img_path, model, device)

            save_path = os.path.join(output_dir, img_name)

            result.save(save_path)

            print(f"Enhanced: {img_name}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")


if __name__ == "__main__":
    main()
