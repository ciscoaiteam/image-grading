# predictor.py

import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from typing import Tuple

# Image transformation pipeline (must match training pipeline)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiTaskModel(nn.Module):
    """
    Multi-task model for image quality prediction.
    Outputs a binary classification (good/bad) and a continuous regression score.
    """
    def __init__(self, pretrained: bool = True):
        super(MultiTaskModel, self).__init__()
        # Backbone: EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        # Feature dimension from backbone
        feature_dim = self.backbone.classifier[1].in_features
        # Remove final classifier
        self.backbone.classifier = nn.Identity()

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification head (binary)
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Regression head (continuous score)
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, 224, 224).
        Returns:
            Tuple of (classification_prob, regression_score)
        """
        features = self.backbone(x)
        shared = self.shared(features)
        cls_out = self.classification_head(shared)
        reg_out = self.regression_head(shared)
        return cls_out, reg_out


def load_model(model_path: str = "model.pth", device: str = "cpu") -> MultiTaskModel:
    """
    Load model weights into the MultiTaskModel architecture.
    """
    model = MultiTaskModel(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_quality(model: MultiTaskModel, image_bytes: bytes, device: str = "cpu") -> Tuple[float, float]:
    """
    Predict image quality.

    Args:
        model (MultiTaskModel): Loaded model.
        image_bytes (bytes): Raw image bytes.
        device (str): Device to run inference on.

    Returns:
        classification_prob (float): Probability [0,1] for good quality.
        regression_score (float): Continuous quality score.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = image_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_out, reg_out = model(tensor)
        return float(cls_out.item()), float(reg_out.item())
    except Exception as e:
        raise RuntimeError(f"Failed to predict image quality: {e}")
