import torch
import torch.nn as nn

class WheatDiseaseCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Extract visual features from images.
        # Each block now has TWO convolutional layers.

        self.features = nn.Sequential(
            # Block 1: 3 input channels -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            # Block 4: 128 -> 256 feature maps
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )

        # Reduce each feature map to one value.
        # Output shape becomes (batch_size, 256, 1, 1).
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Convert extracted features into four class scores.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = WheatDiseaseCNN()

    # Simulate a batch of 32 RGB images, each 384 x 384.
    sample_images = torch.randn(32, 3, 384, 384)

    predictions = model(sample_images)

    print(model)
    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)
    print(
        "Total parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
