from pathlib import Path
from collections import Counter

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "Wheat_Disease" / "train"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "Black Rust",
    "Brown Rust",
    "Healthy Wheat",
    "Yellow Rust",
]

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

total_images = 0

print(f"Dataset directory: {DATASET_DIR}")
print("\nImage counts by class:")

for class_name in CLASS_NAMES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}"
        )

    image_paths = [
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    count = len(image_paths)
    total_images += count

    print(f"{class_name}: {count}")

print(f"\nTotal active images: {total_images}")
