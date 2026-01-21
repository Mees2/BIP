import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    """
    This function returns your model architecture for glaucoma detection using EfficientNet V2.
    
    Args:
        num_classes (int): Number of output classes (2 for binary classification: glaucoma vs normal)
    Returns:
        model: Your PyTorch model (EfficientNet V2 with custom classifier head)
    """
    # Load pre-trained EfficientNet V2 Small
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # Optionally freeze some layers (here, freeze the feature extractor)
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    in_features = model.classifier[1].in_features

    # Replace the classifier head with custom layers
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)  # Output layer
    )

    return model
