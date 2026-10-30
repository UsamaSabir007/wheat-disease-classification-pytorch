from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch
from PIL import Image

from prepare_data import (
    test_ds,
    test_dataset,
    class_names,
    IMAGE_SIZE,
)
from model import WheatDiseaseCNN


# 1. Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model_cleaned_384_jitter_epochs30.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_CLASS = "Black Rust"

# Number of images to show in each window.
IMAGES_PER_WINDOW = 6
COLUMNS = 2

# Inspect every misclassified Black Rust test image.
MAX_IMAGES = None



# 2. Validate and load the checkpoint

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run python src\\train_model.py first."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

if checkpoint["class_names"] != class_names:
    raise ValueError(
        "The class order in the checkpoint does not match "
        "the current dataset."
    )

if tuple(checkpoint["image_size"]) != tuple(IMAGE_SIZE):
    raise ValueError(
        f"Checkpoint image size {checkpoint['image_size']} "
        f"does not match current IMAGE_SIZE {IMAGE_SIZE}."
    )

model = WheatDiseaseCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

target_index = class_names.index(TARGET_CLASS)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Checkpoint: {MODEL_PATH.name}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Saved validation accuracy: {checkpoint['val_accuracy']:.2%}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Test images: {len(test_ds.dataset)}")
print(f"Target class: {TARGET_CLASS}")


# 3. Find misclassified Black Rust images

errors = []
black_rust_total = 0
black_rust_correct = 0

with torch.no_grad():
    for index, (image, label) in enumerate(test_dataset):
        actual = int(label.item())

        # Ignore images that are not actually Black Rust.
        if actual != target_index:
            continue

        black_rust_total += 1

        image_batch = image.unsqueeze(0).to(device)
        outputs = model(image_batch)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)

        predicted = int(prediction.item())

        if predicted == actual:
            black_rust_correct += 1
            continue

        # Retrieve the original image for clear visual inspection.
        image_path = Path(test_dataset.paths[index])

        with Image.open(image_path) as original:
            display_image = original.convert("RGB").copy()

        errors.append({
            "image": display_image,
            "actual": actual,
            "predicted": predicted,
            "confidence": float(confidence.item()),
            "path": str(image_path),
        })


# 4. Summarize Black Rust performance

black_rust_errors = len(errors)

recall = (
    black_rust_correct / black_rust_total
    if black_rust_total else 0.0
)

print(f"\nActual Black Rust images: {black_rust_total}")
print(f"Correctly identified: {black_rust_correct}")
print(f"Misclassified: {black_rust_errors}")
print(f"Black Rust recall: {recall:.2%}")


if not errors:
    print("No misclassified Black Rust images found.")
    raise SystemExit


# 5. Group errors by predicted class

for predicted_index in range(len(class_names)):
    if predicted_index == target_index:
        continue

    group = [
        error for error in errors
        if error["predicted"] == predicted_index
    ]

    if not group:
        continue

    # Highest-confidence mistakes first.
    group.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    if MAX_IMAGES is not None:
        group = group[:MAX_IMAGES]

    predicted_name = class_names[predicted_index]

    print(
        f"\nBlack Rust predicted as {predicted_name}: "
        f"{len(group)} images"
    )

    # Display a few images at a time.
    for start in range(0, len(group), IMAGES_PER_WINDOW):
        batch = group[start:start + IMAGES_PER_WINDOW]
        rows = math.ceil(len(batch) / COLUMNS)

        fig, axes = plt.subplots(
            rows,
            COLUMNS,
            figsize=(14, rows * 5),
            squeeze=False,
        )

        for ax in axes.flat:
            ax.axis("off")

        for i, error in enumerate(batch):
            row = i // COLUMNS
            col = i % COLUMNS
            ax = axes[row, col]

            ax.imshow(error["image"])

            filename = Path(error["path"]).name

            ax.set_title(
                f"{filename}\n"
                f"Actual: {class_names[error['actual']]}\n"
                f"Predicted: {class_names[error['predicted']]}\n"
                f"Confidence: {error['confidence']:.1%}",
                fontsize=10,
                pad=10,
            )

            ax.axis("off")

        window_number = start // IMAGES_PER_WINDOW + 1
        total_windows = math.ceil(
            len(group) / IMAGES_PER_WINDOW
        )

        fig.suptitle(
            f"Black Rust → {predicted_name}\n"
            f"Window {window_number} of {total_windows}",
            fontsize=14,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()