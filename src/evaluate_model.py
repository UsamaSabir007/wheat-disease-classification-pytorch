from pathlib import Path
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from prepare_data import IMAGE_SIZE, class_names, test_ds
from model import WheatDiseaseCNN



# 1. Configuration and model loading

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model_cleaned_384_jitter_epochs30.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    RESULTS_DIR / "evaluation_cleaned_384_jitter_epochs30.txt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}\n"
        "Run python src\\train_model.py first."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

# Verify that the checkpoint matches the current experiment.
if tuple(checkpoint["image_size"]) != tuple(IMAGE_SIZE):
    raise ValueError(
        f"Checkpoint image size {checkpoint['image_size']} "
        f"does not match current IMAGE_SIZE {IMAGE_SIZE}."
    )

if checkpoint["class_names"] != class_names:
    raise ValueError(
        "Checkpoint class names do not match the current dataset."
    )

model = WheatDiseaseCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Checkpoint: {MODEL_PATH.name}")
print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
print(f"Saved validation loss: {checkpoint['val_loss']:.4f}")
print(f"Saved validation accuracy: {checkpoint['val_accuracy']:.2%}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Test images: {len(test_ds.dataset)}")



# 2. Generate predictions on the test set

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_ds:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())



# 3. Overall accuracy

test_accuracy = accuracy_score(all_labels, all_predictions)

print(f"\nTest Accuracy: {test_accuracy:.2%}")


# 4. Per-class precision, recall, and F1-score

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\nClassification Report:\n")
print(report)


# 5. Confusion matrix

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

print("Confusion Matrix:")
print(matrix)

print("\nRows = actual classes")
print("Columns = predicted classes")
print("Class order:", class_names)


# 6. Save evaluation results

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("WheatDiseaseCNN - Evaluation Results\n")
    f.write("=" * 45 + "\n\n")

    f.write(f"Checkpoint: {MODEL_PATH.name}\n")
    f.write(f"Epoch: {checkpoint['epoch']}\n")
    f.write(f"Image size: {IMAGE_SIZE}\n")
    f.write(f"Test images: {len(test_ds.dataset)}\n")
    f.write(f"Saved validation loss: {checkpoint['val_loss']:.4f}\n")
    f.write(
        f"Saved validation accuracy: "
        f"{checkpoint['val_accuracy']:.2%}\n"
    )
    f.write(f"Test accuracy: {test_accuracy:.2%}\n\n")

    f.write("Classification Report:\n\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))
    f.write("\n\nRows = actual classes\n")
    f.write("Columns = predicted classes\n")
    f.write(f"Class order: {class_names}\n")

print(f"\nEvaluation results saved to: {REPORT_PATH}")