from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from prepare_data import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    class_names,
    train_ds,
    val_ds,
)
from model import WheatDiseaseCNN


# 1. Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# Keep this experiment separate from all previous checkpoints.
MODEL_PATH = MODEL_DIR / "best_model_cleaned_384_jitter_epochs30.pth"

EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 8

WEIGHT_DECAY = 0.0001
MAX_GRAD_NORM = 1.0

MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Number of classes: {len(class_names)}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Training images: {len(train_ds.dataset)}")
print(f"Validation images: {len(val_ds.dataset)}")
print(f"Training batches per epoch: {len(train_ds)}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")



# 2. Create the model and training tools

model = WheatDiseaseCNN(
    num_classes=len(class_names)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# OneCycleLR updates the learning rate after every batch.
scheduler = OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=EPOCHS,
    steps_per_epoch=len(train_ds),
    pct_start=0.15,
)


# 3. Train for one epoch

def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_ds:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear gradients from the previous batch.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: generate class scores.
        outputs = model(images)

        # Calculate the classification loss.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients.
        loss.backward()

        # Clip gradients before updating model parameters.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
        )

        # Update model parameters.
        optimizer.step()

        # Update learning rate for the next batch.
        scheduler.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (
            (predictions == labels).sum().item()
        )
        total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


# 4. Evaluate on validation data

def evaluate():
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    # Do not calculate gradients during validation.
    with torch.no_grad():
        for images, labels in val_ds:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct_predictions += (
                (predictions == labels).sum().item()
            )
            total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy



# 5. Training loop with early stopping

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_one_epoch()
    val_loss, val_accuracy = evaluate()

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.2%} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.2%} "
        f"| LR: {current_lr:.6g}"
    )

    # Save the checkpoint whenever validation loss improves.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,

                # Optimizer and scheduler configuration
                "batch_size": train_ds.batch_size,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "AdamW",
                "scheduler": "OneCycleLR",
                "scheduler_max_lr": LEARNING_RATE,
                "scheduler_pct_start": 0.15,
                "gradient_clip_max_norm": MAX_GRAD_NORM,

                # Architecture and preprocessing configuration
                "architecture": "WheatDiseaseCNN_deep",
                "convolutional_layers": 8,
                "preprocessing": (
                    "letterbox_resize_384_with_gray_padding;"
                    "training_only_random_augmentation;"
                    "ImageNet_normalization"
                ),
                "normalize_mean": NORMALIZE_MEAN,
                "normalize_std": NORMALIZE_STD,
            },
            MODEL_PATH,
        )

        print(f"  Best model saved to: {MODEL_PATH}")

    else:
        epochs_without_improvement += 1

    # Stop if validation loss has not improved for PATIENCE epochs.
    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping: validation loss did not improve "
            f"for {PATIENCE} consecutive epochs."
        )
        break



# 6. Training summary

print("\nTraining finished.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Saved model: {MODEL_PATH}")
