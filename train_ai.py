import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------
# Step 1: Organize the 3-Class Data for TensorFlow
# ---------------------------------------------------------
print("Organizing 3-Class dataset...")

base_dir = r"C:\Users\Thabang Moloko\PyCharmMiscProject\Chest Diseases Dataset"

# Removed Pneumonia, kept the remaining 3 classes
source_folders = {
    "0_Healthy": os.path.join(base_dir, "9. Normal", "CSI"),
    "1_Tuberculosis": os.path.join(base_dir, "5. Tuberculosis", "CSI"),
    "2_COVID-19": os.path.join(base_dir, "1. COVID-19", "CSI")
}

# Updated folder name to reflect 3 classes
train_dir = r"C:\Users\Thabang Moloko\PyCharmMiscProject\Training_Data_3Class"
os.makedirs(train_dir, exist_ok=True)

def copy_images(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    if os.path.exists(source_folder):
        count = 0
        for filename in os.listdir(source_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                shutil.copy(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))
                count += 1
        print(f"✅ Copied {count} images from {os.path.basename(os.path.dirname(source_folder))}")
    else:
        print(f"❌ ERROR: Could not find path -> {source_folder}")

for class_name, source_path in source_folders.items():
    dest_path = os.path.join(train_dir, class_name)
    copy_images(source_path, dest_path)

print("Data organization complete!\n")

# ---------------------------------------------------------
# Step 2: Load, Split, and Optimize the Datasets
# ---------------------------------------------------------
print("Loading images into the AI pipeline...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150, 150),
    batch_size=8,
    label_mode='categorical'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=8,
    label_mode='categorical'
)

print(f"Class mapping: {train_dataset.class_names}\n")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------
# Step 3: Build the 3-Class "Brain" (CNN)
# ---------------------------------------------------------
print("Building the Neural Network...")

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomContrast(0.1)
])

model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    data_augmentation,
    layers.Rescaling(1. / 255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # Changed output nodes to 3
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------------------------
# Step 4: Train and Verify the AI
# ---------------------------------------------------------
print("Starting training! Watch the 'val_accuracy' metric...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping]
)

# ---------------------------------------------------------
# Step 5: Save the trained Brain
# ---------------------------------------------------------
model_path = r"C:\Users\Thabang Moloko\PyCharmMiscProject\tb_cough_model.keras"
model.save(model_path)
print(f"\n🎉 Training Complete! The optimized 3-Class AI brain has been saved at: {model_path}")