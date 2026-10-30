# Wheat Disease Classification Using PyTorch

A deep learning project that uses a Convolutional Neural Network (CNN) built with PyTorch to classify wheat leaf images into four categories:

- Black Rust
- Brown Rust
- Healthy Wheat
- Yellow Rust

This project was developed as a hands-on machine learning learning experiment, covering dataset preparation, image preprocessing, CNN architecture design, model training, evaluation, and inference on individual images.

## Project Status

**Completed as a learning experiment.**

The model achieved 90.16% accuracy on the held-out test split. However, predictions on some independently downloaded internet images were less reliable, particularly for Brown Rust and Yellow Rust.

Dataset quality, label reliability, and potential duplicate images across splits are important limitations. The model should not be considered a reliable agricultural disease diagnostic tool.

## Dataset

The dataset is available on Kaggle:

[Wheat Disease Dataset – Kaggle](https://www.kaggle.com/datasets/pesaridivya/wheat-disease)

The dataset contains four classes:

| Class | Original Images |
|---|---:|
| Black Rust | 1,206 |
| Brown Rust | 1,353 |
| Healthy Wheat | 1,411 |
| Yellow Rust | 1,312 |
| **Total** | **5,282** |

The project excluded 24 images using an exclusion list during dataset preparation.

The dataset itself is not included in this repository.

### Dataset Structure

After downloading and extracting the dataset, arrange it as follows:

```text
data/
└── Wheat_Disease/
    └── train/
        ├── Black Rust/
        ├── Brown Rust/
        ├── Healthy Wheat/
        └── Yellow Rust/
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow (PIL)

## Model Architecture

A custom Convolutional Neural Network was implemented using PyTorch.

The architecture consists of:

- Four convolutional blocks.
- Batch Normalization.
- ReLU activation.
- Max Pooling.
- Adaptive Average Pooling.
- Dropout (0.3).
- Fully Connected output layer with four classes.

The convolutional blocks progressively increase the number of feature channels:

| Block | Input Channels | Output Channels |
|---|---:|---:|
| 1 | 3 | 32 |
| 2 | 32 | 64 |
| 3 | 64 | 128 |
| 4 | 128 | 256 |

## Image Preprocessing and Augmentation

Images are resized to 384 × 384 pixels while preserving their original aspect ratio using letterbox padding.

Gray padding is used instead of stretching images, helping avoid distortion.

Training augmentation includes:

- Random Resized Crop
- Horizontal Flip
- Vertical Flip
- Random Rotation
- Color Jitter

ImageNet normalization is applied before feeding images into the model.

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 384 × 384 |
| Batch Size | 32 |
| Epochs | 30 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Weight Decay | 0.0001 |
| Scheduler | OneCycleLR |
| Early Stopping Patience | 8 |
| Dropout | 0.3 |
| Train / Validation / Test | 70% / 15% / 15% |

The dataset was split using stratified sampling with a random seed of 42. The generated splits were saved and reused for consistent experiments.

## Results

The final checkpoint achieved the following results:

| Metric | Score |
|---|---:|
| Test Accuracy | **90.16%** |
| Macro F1-Score | 89.97% |
| Weighted F1-Score | 90.14% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Black Rust | 81.77% | 86.74% | 84.18% |
| Brown Rust | 90.22% | 81.77% | 85.79% |
| Healthy Wheat | 91.56% | 97.17% | 94.28% |
| Yellow Rust | 96.88% | 94.42% | 95.63% |

### Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual / Predicted | Black Rust | Brown Rust | Healthy | Yellow Rust |
|---|---:|---:|---:|---:|
| Black Rust | 157 | 15 | 8 | 1 |
| Brown Rust | 28 | 166 | 6 | 3 |
| Healthy Wheat | 2 | 2 | 206 | 2 |
| Yellow Rust | 5 | 1 | 5 | 186 |

The most significant confusion occurred between Black Rust and Brown Rust.

### Important Evaluation Limitations

During dataset investigation, potential duplicate or near-duplicate images were identified across the training, validation, and test splits. These were not fully resolved.

As a result, the reported test accuracy may be optimistic and should not be interpreted as a reliable estimate of performance on completely unseen photographs.

The model also struggled with some independently downloaded images, particularly for Brown Rust and Yellow Rust. This highlighted the importance of dataset quality, reliable labels, and representative training data.

## Project Structure

```text
WheatDiseaseCNN/
│
├── models/
│   └── best_model_cleaned_384_jitter_epochs30.pth
│
├── results/
│
├── src/
│   ├── model.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_image.py
│   ├── explore_data.py
│   ├── visualize_data.py
│   └── inspect_errors.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd WheatDiseaseCNN
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv-torch
```

Activate it on Windows PowerShell:

```powershell
.\.venv-torch\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the dataset from Kaggle and arrange it using the directory structure described above.

Make sure the dataset preparation configuration points to the correct dataset location.

## Running the Project

Run commands from the project root directory.

### Explore the Dataset

```bash
python src/explore_data.py
```

### Visualize the Dataset

```bash
python src/visualize_data.py
```

### Train the Model

```bash
python src/train_model.py
```

### Evaluate the Model

```bash
python src/evaluate_model.py
```

### Predict an Individual Image

```bash
python src/predict_image.py "path/to/image.jpg"
```

The prediction script displays the predicted class, confidence, and probability scores for all four classes.

### Inspect Misclassified Images

```bash
python src/inspect_errors.py
```


## Key Learnings

Through this project, I gained practical experience with:

- Building a CNN from scratch using PyTorch.
- Understanding convolutional layers, pooling, and batch normalization.
- Preparing image datasets and implementing custom preprocessing.
- Applying data augmentation.
- Training models using GPU acceleration.
- Implementing learning-rate scheduling and early stopping.
- Evaluating classification models using precision, recall, F1-score, and confusion matrices.
- Investigating misclassified images and dataset quality issues.
- Saving trained checkpoints and implementing single-image inference.
- Understanding the difference between test-set performance and real-world generalization.

## Conclusion

This project was an important step in my practical machine learning journey. Although the model achieved encouraging results on the available test split, its limitations on independently sourced images demonstrated that model accuracy alone does not guarantee real-world reliability.

The experiment provided valuable experience in the complete machine learning workflow, particularly dataset investigation, model evaluation, and understanding generalization limitations.