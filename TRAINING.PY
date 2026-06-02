"""
train_hybrid.py  —  Hybrid CNN + Landmark Model Trainer (Improved)
Loads both images and landmarks, trains a dual-input model, saves it.
Changes from v1:
  - Data augmentation on images (rotation, brightness, flip)
  - Reduced Dropout so small datasets don't underfit
  - Per-class accuracy report so you can see exactly which gesture is weak
  - Lower learning rate with ReduceLROnPlateau
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 128
DATASET_DIR = "dataset"
CSV_FILE    = "landmarks.csv"
MODEL_OUT   = "hybrid_model.h5"
ENCODER_OUT = "label_encoder.pkl"
BATCH_SIZE  = 16   # smaller batch = better for small datasets (only 3 classes now)
EPOCHS      = 80   # more epochs — EarlyStopping will cut it when ready
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Load landmarks CSV ─────────────────────────────────────────────────────
print("[1/5] Loading landmarks CSV...")
df = pd.read_csv(CSV_FILE)
labels_raw  = df["label"].values
lm_features = df.drop(columns=["label"]).values.astype(np.float32)
print(f"    Samples per gesture:\n{df['label'].value_counts().to_string()}")

# ── 2. Load matching images ───────────────────────────────────────────────────
print("\n[2/5] Loading images...")
images     = []
valid_mask = []

for idx, row in df.iterrows():
    label      = row["label"]
    label_rows = df[df["label"] == label].index.tolist()
    img_idx    = label_rows.index(idx)
    img_path   = os.path.join(DATASET_DIR, label, f"{img_idx}.jpg")

    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        images.append(img_to_array(img) / 255.0)
        valid_mask.append(True)
    else:
        valid_mask.append(False)

valid_mask  = np.array(valid_mask)
X_images    = np.array(images, dtype=np.float32)
X_landmarks = lm_features[valid_mask]
y_raw       = labels_raw[valid_mask]
print(f"    Loaded {len(X_images)} samples across {len(np.unique(y_raw))} classes")

# ── 3. Encode labels ──────────────────────────────────────────────────────────
print("\n[3/5] Encoding labels...")
le          = LabelEncoder()
y_encoded   = le.fit_transform(y_raw)
num_classes = len(le.classes_)
y_cat       = to_categorical(y_encoded, num_classes)

with open(ENCODER_OUT, "wb") as f:
    pickle.dump(le, f)
print(f"    Classes: {list(le.classes_)}")

# ── 4. Train / val split ──────────────────────────────────────────────────────
X_img_tr, X_img_val, X_lm_tr, X_lm_val, y_tr, y_val, y_enc_tr, y_enc_val = \
    train_test_split(X_images, X_landmarks, y_cat, y_encoded,
                     test_size=0.2, random_state=42, stratify=y_encoded)

# ── 5. Image augmentation (applied only on training set) ──────────────────────
print("\n[4/5] Building model with augmentation...")

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="augmentation")

# ── 6. Build Hybrid Model ─────────────────────────────────────────────────────

# --- CNN Branch ---
img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = augment(img_input)                                        # augment train images

x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)                                    # reduced from 0.4
x = layers.Dense(128, activation="relu")(x)

# --- Landmark Branch ---
lm_input = Input(shape=(42,), name="landmarks")
y = layers.Dense(128, activation="relu")(lm_input)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.2)(y)                                    # reduced from 0.3
y = layers.Dense(64,  activation="relu")(y)
y = layers.Dense(64,  activation="relu")(y)

# --- Fusion ---
combined = layers.Concatenate()([x, y])
z = layers.Dense(256, activation="relu")(combined)
z = layers.Dropout(0.3)(z)
z = layers.Dense(128, activation="relu")(z)
output = layers.Dense(num_classes, activation="softmax", name="output")(z)

model = Model(inputs=[img_input, lm_input], outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),   # lower LR
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\n[5/5] Training...")
callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
]

history = model.fit(
    {"image": X_img_tr, "landmarks": X_lm_tr},
    y_tr,
    validation_data=({"image": X_img_val, "landmarks": X_lm_val}, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ── 8. Evaluate + per-class report ───────────────────────────────────────────
loss, acc = model.evaluate(
    {"image": X_img_val, "landmarks": X_lm_val}, y_val, verbose=0)
print(f"\n{'='*50}")
print(f"  Overall Val Accuracy : {acc*100:.2f}%")
print(f"  Overall Val Loss     : {loss:.4f}")
print(f"{'='*50}")

# Per-class breakdown — tells you exactly which gesture is weak
print("\nPer-gesture accuracy:")
preds_raw  = model.predict({"image": X_img_val, "landmarks": X_lm_val}, verbose=0)
preds_cls  = np.argmax(preds_raw, axis=1)
for cls_idx, cls_name in enumerate(le.classes_):
    mask       = y_enc_val == cls_idx
    if mask.sum() == 0: continue
    cls_acc    = (preds_cls[mask] == cls_idx).mean() * 100
    bar        = "█" * int(cls_acc / 5) + "░" * (20 - int(cls_acc / 5))
    status     = "✔" if cls_acc >= 80 else "⚠  needs more data"
    print(f"  {cls_name:<12}  [{bar}]  {cls_acc:5.1f}%  {status}")

print(f"\n[SAVED]  Model   → {MODEL_OUT}")
print(f"[SAVED]  Encoder → {ENCODER_OUT}")
print("\nIf any gesture shows < 80%, collect 100 more samples for it and retrain.")