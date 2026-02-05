import os

class Config:
    # ===== PATHS =====
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    TRAIN_INPUT = os.path.join(BASE_DIR, "dataset/train/input")
    TRAIN_TARGET = os.path.join(BASE_DIR, "dataset/train/target")

    VAL_INPUT = os.path.join(BASE_DIR, "dataset/val/input")
    VAL_TARGET = os.path.join(BASE_DIR, "dataset/val/target")

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # ===== TRAINING =====
    EPOCHS = 150
    BATCH_SIZE = 4        # safe for 8GB GPU
    PATCH_SIZE = 128      # ⭐ patch-based training
    LR = 2e-4
    NUM_WORKERS = 2
    SAVE_EVERY = 5

    # ===== MEMORY OPTIMIZATION =====
    GRAD_ACCUM_STEPS = 2   # simulate larger batch
    AMP = True             # mixed precision
