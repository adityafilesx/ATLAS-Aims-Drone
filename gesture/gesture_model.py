"""
gesture_model.py — CNN for Hand Gesture Classification
========================================================
A lightweight Convolutional Neural Network built with PyTorch.
Input:  128×128 grayscale image
Output: (gesture_label, confidence) for 13 gesture classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from gesture.gesture_labels import NUM_CLASSES, get_label


class GestureCNN(nn.Module):
    """
    Simple CNN architecture for gesture classification.

    Architecture:
        Conv2d(1,32) → ReLU → MaxPool
        Conv2d(32,64) → ReLU → MaxPool
        Conv2d(64,128) → ReLU → MaxPool
        Flatten → FC(128*16*16, 256) → ReLU → Dropout → FC(256, 13)
    """

    def __init__(self):
        super(GestureCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)

        # Pooling layer (reused)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 3× pooling: 128 → 64 → 32 → 16, so feature map is 128×16×16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch, 1, 128, 128)
        Returns:
            logits: Tensor of shape (batch, NUM_CLASSES)
        """
        x = self.pool(F.relu(self.conv1(x)))   # → (batch, 32, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))   # → (batch, 64, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))   # → (batch, 128, 16, 16)

        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))               # → (batch, 256)
        x = self.dropout(x)
        x = self.fc2(x)                       # → (batch, NUM_CLASSES)
        return x


class GesturePredictor:
    """
    Wrapper that loads a trained GestureCNN and makes predictions.
    Returns (gesture_label, confidence) tuples.
    """

    DEFAULT_MODEL_PATH = os.path.join("models", "gesture_cnn.pth")

    def __init__(self, model_path=None):
        """
        Load the trained model weights.
        Args:
            model_path: path to the .pth file (default: models/gesture_cnn.pth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model = GestureCNN().to(self.device)
        self.model_path = model_path or self.DEFAULT_MODEL_PATH

        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            self.loaded = True
        else:
            print(f"[GesturePredictor] Model not found at {self.model_path}. "
                  "Run train_model.py first.")
            self.loaded = False

    def predict(self, image_tensor):
        """
        Predict gesture from a preprocessed image tensor.
        Args:
            image_tensor: Tensor of shape (1, 1, 128, 128), normalized [0,1]
        Returns:
            (label: str, confidence: float) — e.g. ("thumb_up", 0.93)
        """
        if not self.loaded:
            return ("unknown", 0.0)

        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            label = get_label(predicted.item())
            return (label, confidence.item())
