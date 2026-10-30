from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


# 1. Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "Wheat_Disease" / "train"

# Keep the previous experiment's splits untouched.
SPLIT_DIR = PROJECT_ROOT / "data" / "splits_cleaned"

IMAGE_SIZE = (384, 384)  # (height, width)
BATCH_SIZE = 32
SEED = 42

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

PIN_MEMORY = torch.cuda.is_available()
PADDING_COLOR = (128, 128, 128)

# ImageNet normalization.
# These values must also be used during evaluation and inference.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

EXCLUSION_FILE = PROJECT_ROOT / "data" / "excluded_images.csv"

# True only when creating the new splits for the first time.
# Change to False after successfully creating them.
REBUILD_SPLITS = False

# Do not create the directory yet. First check whether existing
# split files would be overwritten.


# 2. Validate dataset and discover classes

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR}"
    )

# Explicitly use the four intended classes.
# This prevents archive or other folders from becoming classes.
class_names = [
    "Black Rust",
    "Brown Rust",
    "Healthy Wheat",
    "Yellow Rust",
]

for class_name in class_names:
    class_dir = DATASET_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Expected class folder not found: {class_dir}"
        )

# Ensure there are no unexpected directories inside the active
# dataset root that could cause confusion.
unexpected_dirs = {
    path.name
    for path in DATASET_DIR.iterdir()
    if path.is_dir()
} - set(class_names)

if unexpected_dirs:
    raise RuntimeError(
        "Unexpected directories found inside the active dataset: "
        f"{sorted(unexpected_dirs)}. "
        "Move archives and backups outside the train directory."
    )

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

print("Classes:", class_names)
print("Class-to-index mapping:", class_to_index)
print("Dataset directory:", DATASET_DIR)
print("Split directory:", SPLIT_DIR)


# 3. Collect images and apply optional exclusions

image_paths = []
labels = []

excluded_filenames = set()

if EXCLUSION_FILE.exists():
    exclusion_df = pd.read_csv(EXCLUSION_FILE)

    if "filename" not in exclusion_df.columns:
        raise ValueError(
            f"{EXCLUSION_FILE} must contain a 'filename' column."
        )

    excluded_filenames = set(
        exclusion_df["filename"].dropna().astype(str)
    )

for class_name in class_names:
    class_dir = DATASET_DIR / class_name

    for path in sorted(class_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # Preserve the existing exclusion-list behavior.
        if path.name in excluded_filenames:
            continue

        image_paths.append(str(path.resolve()))
        labels.append(class_to_index[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int64)

print("Excluded filenames in exclusion list:", len(excluded_filenames))
print("Total included images:", len(image_paths))

if len(image_paths) == 0:
    raise ValueError("No supported images found.")

if len(set(image_paths)) != len(image_paths):
    raise ValueError("Duplicate image paths found.")

# Verify that every intended class has images.
for class_name, class_index in class_to_index.items():
    count = int(np.sum(labels == class_index))

    if count == 0:
        raise ValueError(
            f"No images found for class: {class_name}"
        )

    print(f"{class_name}: {count} images")


# 4. Create or reuse fixed splits

SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}

all_split_files_exist = all(
    path.exists() for path in SPLIT_FILES.values()
)

any_split_file_exists = any(
    path.exists() for path in SPLIT_FILES.values()
)

if REBUILD_SPLITS:
    # Never overwrite existing splits automatically.
    if any_split_file_exists:
        raise RuntimeError(
            f"Split files already exist in {SPLIT_DIR}.\n"
            "To reuse them, set REBUILD_SPLITS = False.\n"
            "To intentionally create another split version, "
            "choose a new SPLIT_DIR."
        )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # First split: 70% training, 30% temporary.
    train_paths, temp_paths, train_labels, temp_labels = (
        train_test_split(
            image_paths,
            labels,
            test_size=0.30,
            random_state=SEED,
            stratify=labels,
        )
    )

    # Divide the remaining 30% equally between validation and test.
    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.50,
            random_state=SEED,
            stratify=temp_labels,
        )
    )

    for name, paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        pd.DataFrame({
            "path": paths,
            "label": split_labels,
        }).to_csv(SPLIT_FILES[name], index=False)

    print(f"\nCreated new splits in: {SPLIT_DIR}")

else:
    if not all_split_files_exist:
        raise RuntimeError(
            f"Missing split files in {SPLIT_DIR}.\n"
            "Set REBUILD_SPLITS = True only when intentionally "
            "creating a new set of splits in an empty directory."
        )

    split_data = {}
    current_paths = set(image_paths.tolist())

    for name, split_file in SPLIT_FILES.items():
        df = pd.read_csv(split_file)

        if not {"path", "label"}.issubset(df.columns):
            raise ValueError(
                f"{split_file} must contain path and label columns."
            )

        paths = df["path"].astype(str).to_numpy()
        split_labels = df["label"].to_numpy(dtype=np.int64)

        if len(paths) != len(split_labels):
            raise RuntimeError(
                f"Path/label count mismatch in {split_file}."
            )

        if len(set(paths)) != len(paths):
            raise RuntimeError(
                f"Duplicate paths found within {name} split."
            )

        missing_paths = set(paths) - current_paths

        if missing_paths:
            raise RuntimeError(
                f"{name} split contains missing or excluded images. "
                "Review the dataset and split files before proceeding."
            )

        expected_labels = np.array([
            class_to_index[Path(path).parent.name]
            for path in paths
        ])

        if not np.array_equal(split_labels, expected_labels):
            raise RuntimeError(
                f"Labels in {split_file} do not match class folders."
            )

        split_data[name] = (paths, split_labels)

    train_paths, train_labels = split_data["train"]
    val_paths, val_labels = split_data["val"]
    test_paths, test_labels = split_data["test"]

    all_assigned_paths = (
        list(train_paths)
        + list(val_paths)
        + list(test_paths)
    )

    if len(all_assigned_paths) != len(set(all_assigned_paths)):
        raise RuntimeError(
            "An image appears in more than one split."
        )

    if set(all_assigned_paths) != current_paths:
        raise RuntimeError(
            "Saved splits do not cover the current dataset exactly. "
            "Review additions/removals before proceeding."
        )

    print(f"\nLoaded existing splits from: {SPLIT_DIR}")


# 5. Display class distributions

def show_distribution(name, split_labels):
    counts = np.bincount(
        split_labels,
        minlength=len(class_names),
    )

    print(f"\n{name}: {len(split_labels)} images")

    for class_name, count in zip(class_names, counts):
        print(f"  {class_name}: {count}")


show_distribution("Training", train_labels)
show_distribution("Validation", val_labels)
show_distribution("Test", test_labels)

print(
    "\nTotal images assigned to splits:",
    len(train_paths) + len(val_paths) + len(test_paths),
)


# 6. Preserve aspect ratio using letterbox resizing

class LetterboxResize:
    """
    Resize an image to fit IMAGE_SIZE without stretching it.

    Empty space is filled with the configured gray padding color.
    Input and output are PIL RGB images.
    """

    def __init__(self, image_size, padding_color=(128, 128, 128)):
        self.image_size = image_size
        self.padding_color = padding_color

    def __call__(self, image):
        target_height, target_width = self.image_size
        original_width, original_height = image.size

        scale = min(
            target_width / original_width,
            target_height / original_height,
        )

        new_width = max(1, round(original_width * scale))
        new_height = max(1, round(original_height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "RGB",
            (target_width, target_height),
            self.padding_color,
        )

        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2

        canvas.paste(image, (left, top))

        return canvas


# 7. Define training and evaluation transforms

letterbox = LetterboxResize(
    IMAGE_SIZE,
    padding_color=PADDING_COLOR,
)

# Random transformations are applied ONLY to training images.
train_transform = transforms.Compose([
    letterbox,

    # Moderate zoom/crop while keeping the crop close to square.
    transforms.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.80, 1.00),
        ratio=(0.90, 1.10),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=15,
        fill=PADDING_COLOR,
    ),

    # Keep color changes moderate because rust color is informative.
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.10,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    ),
])

# Validation and test images remain deterministic.
eval_transform = transforms.Compose([
    letterbox,
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    ),
])


# 8. Custom PyTorch Dataset

class WheatDiseaseDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        label = int(self.labels[index])

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# 9. Create datasets and data loaders

train_dataset = WheatDiseaseDataset(
    train_paths,
    train_labels,
    train_transform,
)

val_dataset = WheatDiseaseDataset(
    val_paths,
    val_labels,
    eval_transform,
)

test_dataset = WheatDiseaseDataset(
    test_paths,
    test_labels,
    eval_transform,
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_ds = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY,
    generator=generator,
)

val_ds = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY,
)

test_ds = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY,
)


# 10. Verify a batch

images, batch_labels = next(iter(train_ds))

print("\nTraining batch image shape:", tuple(images.shape))
print("Training batch label shape:", tuple(batch_labels.shape))
print(
    "Normalized training tensor range:",
    images.min().item(),
    "to",
    images.max().item(),
)
print("Image tensor dtype:", images.dtype)
print("Label tensor dtype:", batch_labels.dtype)

# Also inspect deterministic evaluation preprocessing.
val_images, val_batch_labels = next(iter(val_ds))

print("\nValidation batch image shape:", tuple(val_images.shape))
print(
    "Normalized validation tensor range:",
    val_images.min().item(),
    "to",
    val_images.max().item(),
)

print("GPU available:", torch.cuda.is_available())
print("Pinned memory:", PIN_MEMORY)
