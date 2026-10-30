from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "Wheat_Disease"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLES_PER_CLASS = 3

random.seed(42)

class_dirs = sorted(
    path for path in DATASET_DIR.iterdir()
    if path.is_dir()
)

if not class_dirs:
    raise FileNotFoundError(
        f"No class folders found inside {DATASET_DIR}"
    )

fig, axes = plt.subplots(
    len(class_dirs),
    SAMPLES_PER_CLASS,
    figsize=(12, 4 * len(class_dirs))
)

if len(class_dirs) == 1:
    axes = axes.reshape(1, -1)

for row, class_dir in enumerate(class_dirs):
    image_paths = [
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\n{class_dir.name}: {len(image_paths)} images")

    if not image_paths:
        continue

    # Inspect image dimensions
    dimensions = Counter()

    for path in image_paths:
        try:
            with Image.open(path) as img:
                dimensions[img.size] += 1
        except Exception:
            print(f"Could not read: {path}")

    print("Most common image dimensions:")
    for size, count in dimensions.most_common(5):
        print(f"  {size[0]} x {size[1]}: {count} images")

    samples = random.sample(
        image_paths,
        min(SAMPLES_PER_CLASS, len(image_paths))
    )

    for col in range(SAMPLES_PER_CLASS):
        ax = axes[row, col]

        if col >= len(samples):
            ax.axis("off")
            continue

        with Image.open(samples[col]) as img:
            ax.imshow(img.convert("RGB"))

        ax.set_title(class_dir.name)
        ax.axis("off")

plt.tight_layout()
plt.show()
