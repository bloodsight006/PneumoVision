"""
train.py — PneumoVision Fast Training
Model  : MobileNetV2 @ 160x160
Target : ~95-97% accuracy in ~35-45 min on Intel i7 CPU
"""

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model, IMG_SIZE, MODEL_PATH

# ── Config ─────────────────────────────────────────────────────
DATASET_DIR = "dataset"
FLAT_DIR    = "dataset_flat"   # we flatten subfolders here
BATCH_SIZE  = 64
EPOCHS      = 12
LR          = 1e-3
SEED        = 42

# Use all CPU threads
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


# ── Step 1: Flatten dataset ────────────────────────────────────
# Handles both:
#   dataset/COVID/*.png          (flat)
#   dataset/COVID/images/*.png   (nested — Kaggle default)
def flatten_dataset():
    if os.path.exists(FLAT_DIR):
        print(f"  '{FLAT_DIR}' already exists — skipping flatten step")
        return

    print(f"  Flattening dataset into '{FLAT_DIR}'...")
    os.makedirs(FLAT_DIR, exist_ok=True)

    class_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
        and d != FLAT_DIR
    ])

    for cls in class_dirs:
        cls_src  = os.path.join(DATASET_DIR, cls)
        cls_dest = os.path.join(FLAT_DIR, cls)
        os.makedirs(cls_dest, exist_ok=True)
        count = 0

        # Walk all subdirectories
        for root, dirs, files in os.walk(cls_src):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src  = os.path.join(root, fname)
                    # Make filename unique to avoid collisions
                    unique_name = f"{os.path.basename(root)}_{fname}"
                    dst  = os.path.join(cls_dest, unique_name)
                    shutil.copy2(src, dst)
                    count += 1

        print(f"  {cls}: {count} images copied")

    print(f"  Flatten complete!\n")


# ── Step 2: Build generators ───────────────────────────────────
def build_generators():
    folders = sorted([
        d for d in os.listdir(FLAT_DIR)
        if os.path.isdir(os.path.join(FLAT_DIR, d))
    ])
    print(f"  Classes found: {folders}")

    for cls in folders:
        n = len(os.listdir(os.path.join(FLAT_DIR, cls)))
        print(f"  {cls}: {n} images")

    train_gen_config = ImageDataGenerator(
        rescale            = 1.0 / 255,
        rotation_range     = 10,
        width_shift_range  = 0.08,
        height_shift_range = 0.08,
        zoom_range         = 0.08,
        brightness_range   = [0.85, 1.15],
        horizontal_flip    = True,
        fill_mode          = "nearest",
        validation_split   = 0.20,
    )
    val_gen_config = ImageDataGenerator(
        rescale          = 1.0 / 255,
        validation_split = 0.20,
    )

    train_gen = train_gen_config.flow_from_directory(
        FLAT_DIR,
        target_size  = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        class_mode   = "categorical",
        subset       = "training",
        classes      = folders,
        seed         = SEED,
        interpolation= "bilinear",
    )
    val_gen = val_gen_config.flow_from_directory(
        FLAT_DIR,
        target_size  = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        class_mode   = "categorical",
        subset       = "validation",
        classes      = folders,
        seed         = SEED,
    )

    print(f"\n  Class mapping: {train_gen.class_indices}")
    return train_gen, val_gen, folders


# ── Step 3: Train ──────────────────────────────────────────────
def train():
    # Delete old model
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"Deleted old {MODEL_PATH} — starting fresh\n")

    # Flatten dataset (handles nested Kaggle folders automatically)
    print("Step 1: Preparing dataset...")
    flatten_dataset()

    print("Step 2: Building data generators...")
    train_gen, val_gen, folders = build_generators()

    # Class weights to fix imbalance
    counts  = np.bincount(train_gen.classes)
    total   = counts.sum()
    weights = total / (len(counts) * counts)
    class_weights = {i: float(w) for i, w in enumerate(weights)}
    print(f"\n  Class weights: { {folders[i]: round(w,2) for i,w in class_weights.items()} }")

    print("\nStep 3: Building MobileNetV2 model...")
    model = create_model(num_classes=len(folders))

    # Freeze backbone
    backbone = model.layers[1]
    for layer in backbone.layers:
        layer.trainable = False

    model.compile(
        optimizer = tf.keras.optimizers.Adam(LR),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    trainable = sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"  Trainable params: {trainable:,}")
    print(f"  Image size      : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Max epochs      : {EPOCHS}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only = True,
            monitor        = "val_accuracy",
            verbose        = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 2,
            verbose  = 1,
            min_lr   = 1e-6,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 4,
            restore_best_weights = True,
            verbose              = 1,
        ),
    ]

    print("\n" + "="*55)
    print("  TRAINING STARTED — DO NOT CLOSE THIS WINDOW")
    print("="*55 + "\n")

    model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = EPOCHS,
        callbacks       = callbacks,
        class_weight    = class_weights,
    )

    # Evaluate best saved model
    print("\nEvaluating best model...")
    best  = tf.keras.models.load_model(MODEL_PATH)
    loss, acc, auc = best.evaluate(val_gen, verbose=1)

    print(f"\n{'='*55}")
    print(f"  Validation Accuracy : {acc*100:.2f}%")
    print(f"  Validation AUC      : {auc:.4f}")
    print(f"  Model saved to      : {MODEL_PATH}")
    print(f"  Class order         : {folders}")
    print(f"{'='*55}")

    if acc >= 0.95:
        print("\n  TARGET REACHED! Model ready for deployment.")
    else:
        print("\n  Run again — model may improve with another pass.")


if __name__ == "__main__":
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    train()
