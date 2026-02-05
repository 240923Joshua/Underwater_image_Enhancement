import os
import shutil
import random
import sys


# ---------- USER SETTINGS ----------

# CHANGE THIS to your actual LSUI folder location
RAW_LSUI_PATH = r"C:\underwater_advanced\LSUI"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42

OUTPUT_PATH = "dataset"

# -----------------------------------


def verify_paths():

    print("Checking LSUI paths...")

    if not os.path.exists(RAW_LSUI_PATH):
        print("\nERROR: LSUI folder not found at:")
        print(RAW_LSUI_PATH)
        print("\nPlease edit RAW_LSUI_PATH in the script to match your actual LSUI location.")
        sys.exit(1)

    input_path = os.path.join(RAW_LSUI_PATH, "input")
    gt_path = os.path.join(RAW_LSUI_PATH, "GT")

    if not os.path.exists(input_path):
        print("\nERROR: 'input' folder not found inside LSUI folder.")
        print("Expected:", input_path)
        sys.exit(1)

    if not os.path.exists(gt_path):
        print("\nERROR: 'GT' folder not found inside LSUI folder.")
        print("Expected:", gt_path)
        sys.exit(1)

    return input_path, gt_path


def create_dirs():

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_PATH, split, "input"), exist_ok=True)

        if split != "test":
            os.makedirs(os.path.join(OUTPUT_PATH, split, "target"), exist_ok=True)


def get_valid_pairs(INPUT_PATH, GT_PATH):

    input_images = sorted(os.listdir(INPUT_PATH))
    gt_images = sorted(os.listdir(GT_PATH))

    valid = []

    for img in input_images:
        if img in gt_images:
            valid.append(img)

    return valid


def split_dataset(files):

    random.seed(RANDOM_SEED)
    random.shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    return train, val, test


def copy_files(file_list, split, INPUT_PATH, GT_PATH):

    for f in file_list:

        src_inp = os.path.join(INPUT_PATH, f)
        dst_inp = os.path.join(OUTPUT_PATH, split, "input", f)

        shutil.copy(src_inp, dst_inp)

        if split != "test":
            src_gt = os.path.join(GT_PATH, f)
            dst_gt = os.path.join(OUTPUT_PATH, split, "target", f)

            shutil.copy(src_gt, dst_gt)


def main():

    INPUT_PATH, GT_PATH = verify_paths()

    print("Creating directories...")
    create_dirs()

    print("Collecting valid image pairs...")
    files = get_valid_pairs(INPUT_PATH, GT_PATH)

    print("Total valid pairs found:", len(files))

    if len(files) == 0:
        print("No matching images found between input and GT folders!")
        sys.exit(1)

    train, val, test = split_dataset(files)

    print("Train images:", len(train))
    print("Val images:", len(val))
    print("Test images:", len(test))

    print("Copying training data...")
    copy_files(train, "train", INPUT_PATH, GT_PATH)

    print("Copying validation data...")
    copy_files(val, "val", INPUT_PATH, GT_PATH)

    print("Copying test data...")
    copy_files(test, "test", INPUT_PATH, GT_PATH)

    print("\nDataset successfully arranged!")
    print("Output location:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
