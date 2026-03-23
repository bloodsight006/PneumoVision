"""
model.py — PneumoVision ML Core
Model   : MobileNetV2 (faster than DenseNet121, same accuracy)
Classes : COVID-19 | Normal | Pneumonia
GradCAM : Bulletproof 3-method fallback
Novelty : Bilateral Lung Asymmetry Analysis (BLAA)
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

CLASSES    = ["COVID-19", "Normal", "Pneumonia"]
IMG_SIZE   = 160        # reduced from 224 — 2x faster, minimal accuracy loss
MODEL_PATH = "xray_model.h5"


# ── Model Architecture ─────────────────────────────────────────
def create_model(num_classes=3):
    base = tf.keras.applications.MobileNetV2(
        weights      = "imagenet",
        include_top  = False,
        input_shape  = (IMG_SIZE, IMG_SIZE, 3),
    )

    # Freeze all except last 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="PneumoVision_MobileNetV2")


def load_xray_model():
    if os.path.exists(MODEL_PATH):
        print(f"  Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded successfully")
    else:
        print(f"  No saved model found — using ImageNet weights only. Run train.py.")
        model = create_model()
    return model


# ── Preprocessing ──────────────────────────────────────────────
def preprocess(image):
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for c in range(img.shape[2]):
        img[:, :, c] = clahe.apply(img[:, :, c])
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# ── Inference ──────────────────────────────────────────────────
def predict_xray(model, image):
    inp   = preprocess(image)
    preds = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(preds))

    # Map index back to class name using model output order
    num_classes = len(preds)
    classes = CLASSES[:num_classes]

    return {
        "class_name"   : classes[idx],
        "class_idx"    : idx,
        "confidence"   : float(preds[idx]) * 100,
        "probabilities": {c: float(p) * 100 for c, p in zip(classes, preds)},
    }


# ── GradCAM ────────────────────────────────────────────────────
def get_all_layers_flat(model):
    layers = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            layers.extend(get_all_layers_flat(layer))
        else:
            layers.append(layer)
    return layers


def generate_gradcam_heatmap(model, image, class_idx):
    img_array = preprocess(image)
    heatmap   = None

    # ── Method 1: Last Conv2D gradients ───────────────────────
    try:
        all_layers   = get_all_layers_flat(model)
        conv_layers  = [l for l in all_layers if isinstance(l, tf.keras.layers.Conv2D)]

        if conv_layers:
            last_conv = conv_layers[-1]
            print(f"  GradCAM: using {last_conv.name}")

            grad_model = tf.keras.models.Model(
                inputs  = model.inputs,
                outputs = [last_conv.output, model.output]
            )

            with tf.GradientTape() as tape:
                inputs_t = tf.cast(img_array, tf.float32)
                conv_out, preds = grad_model(inputs_t)
                tape.watch(conv_out)
                loss = preds[:, class_idx]

            grads = tape.gradient(loss, conv_out)

            if grads is not None:
                pooled   = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
                conv_map = conv_out[0].numpy()
                heatmap  = np.dot(conv_map, pooled)
                print("  GradCAM method 1 success")

    except Exception as e:
        print(f"  GradCAM method 1 failed: {e}")

    # ── Method 2: Input gradient saliency ─────────────────────
    if heatmap is None:
        try:
            print("  GradCAM method 2 (input gradients)...")
            inputs_t = tf.Variable(tf.cast(img_array, tf.float32))
            with tf.GradientTape() as tape:
                tape.watch(inputs_t)
                preds = model(inputs_t, training=False)
                loss  = preds[:, class_idx]
            grads   = tape.gradient(loss, inputs_t)
            heatmap = np.mean(np.abs(grads.numpy()[0]), axis=-1)
            print("  GradCAM method 2 success")
        except Exception as e:
            print(f"  GradCAM method 2 failed: {e}")

    # ── Method 3: Occlusion saliency ──────────────────────────
    if heatmap is None:
        try:
            print("  GradCAM method 3 (occlusion)...")
            img_np    = img_array[0]
            base_pred = model.predict(img_array, verbose=0)[0][class_idx]
            sal_map   = np.zeros((IMG_SIZE, IMG_SIZE))
            step      = 20
            for i in range(0, IMG_SIZE, step):
                for j in range(0, IMG_SIZE, step):
                    masked = img_np.copy()
                    masked[i:i+step, j:j+step, :] = 0
                    p = model.predict(np.expand_dims(masked, 0), verbose=0)[0][class_idx]
                    sal_map[i:i+step, j:j+step] = base_pred - p
            heatmap = sal_map
            print("  GradCAM method 3 success")
        except Exception as e:
            print(f"  GradCAM method 3 failed: {e}")

    # ── All methods failed ─────────────────────────────────────
    if heatmap is None:
        orig     = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        orig_bgr = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return orig_bgr, {"left_pct": 0.0, "right_pct": 0.0,
                           "pattern": "GradCAM unavailable"}

    # ── Normalise ──────────────────────────────────────────────
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # ── BLAA ──────────────────────────────────────────────────
    H, W       = heatmap.shape
    left_half  = heatmap[:, :W // 2]
    right_half = heatmap[:, W // 2:]
    threshold  = 0.45
    left_pct   = round(float(np.mean(left_half  > threshold) * 100), 2)
    right_pct  = round(float(np.mean(right_half > threshold) * 100), 2)
    asym       = abs(left_pct - right_pct)

    if left_pct > 10 and right_pct > 10 and asym < 15:
        pattern = "Bilateral symmetric — consistent with COVID-19 pattern"
    elif asym >= 15:
        dominant = "left" if left_pct > right_pct else "right"
        pattern  = f"Unilateral {dominant}-dominant — consistent with bacterial pneumonia"
    else:
        pattern  = "Minimal / no significant pulmonary involvement detected"

    blaa_result = {"left_pct": left_pct, "right_pct": right_pct, "pattern": pattern}

    # ── Overlay ────────────────────────────────────────────────
    orig_rgb    = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heat_up     = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heat_col    = cv2.applyColorMap(np.uint8(255 * heat_up), cv2.COLORMAP_JET)
    heat_col    = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
    overlay     = (heat_col * 0.45 + orig_rgb * 0.55).clip(0, 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    return overlay_bgr, blaa_result
