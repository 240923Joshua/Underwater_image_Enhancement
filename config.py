import os
import torch

class Config:
    # ===== ROOT PROJECT DIR =====
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ===== DATASET PATHS =====
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

    TRAIN_INPUT = os.path.join(DATASET_DIR, "train", "input")
    TRAIN_TARGET = os.path.join(DATASET_DIR, "train", "target")

    VAL_INPUT = os.path.join(DATASET_DIR, "val", "input")
    VAL_TARGET = os.path.join(DATASET_DIR, "val", "target")

    TEST_INPUT = os.path.join(DATASET_DIR, "test", "input")

    # ===== OUTPUT DIRS =====
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    RESULT_DIR = os.path.join(ROOT_DIR, "results")

    # ===== TRAINING SETTINGS =====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 150
    BATCH_SIZE = 4          # safe for 8GB GPU
    LR = 2e-4
    NUM_WORKERS = 2

    # ===== PATCH TRAINING =====
    PATCH_SIZE = 128        # 🔥 critical for Restormer memory
    GRAD_ACCUM_STEPS = 2    # effective batch size boost

    # ===== CHECKPOINTING =====
    SAVE_EVERY = 10

    # ===== AMP / MIXED PRECISION =====
    USE_AMP = True

    # ===== DEBUG / SAFETY =====
    CLAMP_OUTPUT = True
