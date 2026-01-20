import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    """
    This function returns your model architecture for glaucoma detection.
    
    Args:
        num_classes (int): Number of output classes (2 for binary classification:
                          glaucoma vs normal)
    
    Returns:
        model: Your PyTorch model (ResNet50 with custom classifier head)
    """
    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Unfreeze layer2 and deeper for better fine-tuning
    for param in model.layer1.parameters():
        param.requires_grad = False
    # Allow layer2, layer3, layer4 to train for better adaptation
    
    # Get the number of input features for the classifier
    in_features = model.fc.in_features
    
    # Replace the classifier head with custom layers
    model.fc = nn.Sequential(
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
