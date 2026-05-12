# train_disease_model.py
# Trains a binary disease detection model
# Healthy vs Diseased using PlantVillage data

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os, shutil, json

# ── BUILD BINARY DATASET ──────────────────────────────────────────────────────
# Reorganize PlantVillage into just 2 classes: Healthy and Diseased
SOURCE = "./combined_dataset"
DEST = "./disease_dataset"

print("Building healthy vs diseased dataset...")
os.makedirs(f"{DEST}/Healthy", exist_ok=True)
os.makedirs(f"{DEST}/Diseased", exist_ok=True)

healthy_count = 0
diseased_count = 0

for folder in os.listdir(SOURCE):
    src = os.path.join(SOURCE, folder)
    if not os.path.isdir(src):
        continue

    is_healthy = "healthy" in folder.lower()
    dst = os.path.join(DEST, "Healthy" if is_healthy else "Diseased")

    images = [f for f in os.listdir(src)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    # Max 300 per folder to keep balance
    images = images[:300]

    for img in images:
        dest_img = os.path.join(dst, f"{folder}_{img}")
        if not os.path.exists(dest_img):
            shutil.copy(os.path.join(src, img), dest_img)

    if is_healthy:
        healthy_count += len(images)
    else:
        diseased_count += len(images)
    print(f"  {'Healthy' if is_healthy else 'Diseased'}: {folder} ({len(images)} images)")

print(f"\nTotal Healthy: {healthy_count}")
print(f"Total Diseased: {diseased_count}")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 20

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    DEST, target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='binary',
    subset='training'
)
val = datagen.flow_from_directory(
    DEST, target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='binary',
    subset='validation'
)

print(f"\nClass mapping: {train.class_indices}")
print(f"Training on {train.samples} images...")

base = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')  # Binary: 0=Diseased, 1=Healthy
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    ModelCheckpoint('disease_model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    train, validation_data=val,
    epochs=EPOCHS, callbacks=callbacks
)

# Save class indices
with open('disease_class_indices.json', 'w') as f:
    json.dump(train.class_indices, f)

final_acc = round(history.history['val_accuracy'][-1] * 100, 1)
print(f"\nDisease model saved as disease_model.h5")
print(f"Final accuracy: {final_acc}%")