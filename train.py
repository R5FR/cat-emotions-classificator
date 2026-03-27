"""
Script d'entraînement autonome — Cat Emotions Classifier
Génère best_mnv2_model.keras sans avoir besoin de Jupyter.

Usage :
    python train.py
"""

import os, random, warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS_P1   = 15
EPOCHS_P2   = 20
CLASSES     = ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised']
BASE_DIR    = Path(__file__).parent

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print(f"TensorFlow {tf.__version__} | GPU: {bool(tf.config.list_physical_devices('GPU'))}")

# ── Collecte des chemins ──────────────────────────────────────────────────────
all_paths, all_labels = [], []
for split in ["train", "valid"]:
    for cls_dir in sorted((BASE_DIR / split).iterdir()):
        if cls_dir.is_dir() and cls_dir.name in CLASSES:
            for p in cls_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    all_paths.append(str(p))
                    all_labels.append(cls_dir.name)

print(f"{len(all_paths)} images collectées")

# ── Split 70/15/15 ────────────────────────────────────────────────────────────
p_train, p_tmp, l_train, l_tmp = train_test_split(
    all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels)
p_val, p_test, l_val, l_test = train_test_split(
    p_tmp, l_tmp, test_size=0.50, random_state=SEED, stratify=l_tmp)

print(f"Train: {len(p_train)} | Val: {len(p_val)} | Test: {len(p_test)}")

# ── Chargement images ─────────────────────────────────────────────────────────
le = LabelEncoder(); le.fit(CLASSES)

def load_images(paths, labels):
    X, y = [], []
    for path, label in zip(paths, labels):
        try:
            img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            X.append(arr); y.append(label)
        except Exception:
            pass
    return np.array(X, dtype=np.float32), le.transform(y)

print("Chargement des images...")
X_train, y_train = load_images(p_train, l_train)
X_val,   y_val   = load_images(p_val,   l_val)
X_test,  y_test  = load_images(p_test,  l_test)
print("Chargement terminé.")

# ── Modèle MobileNetV2 ────────────────────────────────────────────────────────
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                   include_top=False, weights="imagenet", pooling=None)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.08)(x)
x = layers.RandomZoom(0.08)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.40)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
model = keras.Model(inputs, outputs)

aug = ImageDataGenerator(horizontal_flip=True, rotation_range=12,
                          zoom_range=0.08, width_shift_range=0.05,
                          height_shift_range=0.05, fill_mode="nearest")
val_gen = ImageDataGenerator()

cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=8,
                             restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                 patience=4, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint("best_mnv2_model.keras", monitor="val_accuracy",
                               save_best_only=True, verbose=0),
]

# ── Phase 1 ───────────────────────────────────────────────────────────────────
print("\n--- Phase 1 : Feature extraction ---")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
          epochs=EPOCHS_P1,
          validation_data=val_gen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False),
          callbacks=cb, verbose=1)

# ── Phase 2 : Fine-tuning ─────────────────────────────────────────────────────
print("\n--- Phase 2 : Fine-tuning ---")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
          epochs=EPOCHS_P2,
          validation_data=val_gen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False),
          callbacks=cb, verbose=1)

# ── Évaluation finale ─────────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="macro")
print(f"\nTest — Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
print("Modele sauvegarde : best_mnv2_model.keras")
