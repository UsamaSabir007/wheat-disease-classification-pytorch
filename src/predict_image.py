from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image

from prepare_data import (
    IMAGE_SIZE,
    class_names,
    test_dataset,
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


# 2. Get the image path

if len(sys.argv) != 2:
    print(
        'Usage: python src\\predict_image.py "path\\to\\image.jpg"'
    )
    raise SystemExit(1)

IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# 3. Load the trained model

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

if checkpoint["class_names"] != class_names:
    raise ValueError("Checkpoint class order does not match the dataset.")

if tuple(checkpoint["image_size"]) != tuple(IMAGE_SIZE):
    raise ValueError("Checkpoint image size does not match IMAGE_SIZE.")

model = WheatDiseaseCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# 4. Load and preprocess the image

with Image.open(IMAGE_PATH) as original:
    image = original.convert("RGB")

# Reuse the test dataset's deterministic transform so
# inference uses the same preprocessing as evaluation.
transform = test_dataset.transform

if transform is None:
    raise ValueError("The test dataset has no preprocessing transform.")

image_tensor = transform(image).unsqueeze(0).to(device)


# 5. Predict the disease class

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

predicted_index = int(probabilities.argmax().item())
predicted_class = class_names[predicted_index]
confidence = float(probabilities[predicted_index].item())

print("\nPrediction Results")
print("=" * 35)
print(f"Image: {IMAGE_PATH.name}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

print("\nProbabilities by class:")

for index in probabilities.argsort(descending=True).tolist():
    print(f"{class_names[index]:<16} {probabilities[index].item():.2%}")


# 6. Display the image and prediction

plt.figure(figsize=(8, 7))
plt.imshow(image)
plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence:.2%}"
)
plt.axis("off")
plt.tight_layout()
plt.show()