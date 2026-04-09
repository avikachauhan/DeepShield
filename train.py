"""
DeepShield — train.py
Trains a MobileNetV2-based binary classifier (real vs. fake).
Expects:
    dataset/train/real/   dataset/train/fake/
    dataset/val/real/     dataset/val/fake/
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20
MODEL_OUT  = "model/deepfake_model.h5"
TRAIN_DIR  = "dataset/train"
VAL_DIR    = "dataset/val"

os.makedirs("model", exist_ok=True)

# ── Data generators ───────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["real", "fake"],
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["real", "fake"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False   # freeze feature extractor initially

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="sigmoid")(x)   # 1 = fake, 0 = real

model = Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
]

# ── Phase 1: train head only ──────────────────────────────────────────────────
print("\n[Phase 1] Training classification head …")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Phase 2: fine-tune last 30 layers of base ─────────────────────────────────
print("\n[Phase 2] Fine-tuning top layers of MobileNetV2 …")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks,
)

print(f"\n✅ Model saved → {MODEL_OUT}")
