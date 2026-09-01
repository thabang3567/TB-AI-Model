import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Updated path to 3-Class folder
data_dir = r"C:\Users\Thabang Moloko\PyCharmMiscProject\Training_Data_3Class"
model_path = r"C:\Users\Thabang Moloko\PyCharmMiscProject\tb_cough_model.keras"
output_image = r"C:\Users\Thabang Moloko\PyCharmMiscProject\confusion_matrix.png"

def evaluate_model():
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("Loading validation dataset...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(150, 150),
        batch_size=8,
        label_mode='categorical'
    )

    class_names = val_dataset.class_names
    print(f"Classes detected: {class_names}")

    print("Generating predictions and extracting ground truth...")
    true_labels = []
    predicted_labels = []

    for images, labels in val_dataset:
        true_labels.extend(np.argmax(labels.numpy(), axis=-1))
        batch_preds = model.predict(images, verbose=0)
        predicted_labels.extend(np.argmax(batch_preds, axis=-1))

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False,
                annot_kws={"size": 14})

    plt.title('AI Respiratory Screening: Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Diagnosis (Ground Truth)', fontsize=12, fontweight='bold')
    plt.xlabel('AI Predicted Diagnosis', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_image, dpi=300)
    print(f"\n✅ High-resolution confusion matrix saved as '{output_image}'")

    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    plt.show()

if __name__ == "__main__":
    evaluate_model()