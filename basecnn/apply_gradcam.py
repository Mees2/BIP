import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import random
import os
from model import get_model
from transforms import get_test_transform
from gradcam import GradCAM
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             confusion_matrix)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Get the directory of the current script
script_dir = Path(__file__).parent

# Load model
print('Loading model...')
model = get_model(num_classes=2)
model_path = script_dir / 'model.pth'
model.load_state_dict(torch.load(str(model_path), map_location=device))
model = model.to(device)
model.eval()

# Get test transform
transform = get_test_transform()

# Load test dataset
test_data_dir = script_dir / 'data' / 'test'
test_dataset = datasets.ImageFolder(str(test_data_dir), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate on test set
print('\n' + '='*60)
print('Evaluating Model on Test Set')
print('='*60)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc_roc = roc_auc_score(all_labels, all_probs)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall (Sensitivity): {recall:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print('='*60 + '\n')

# Initialize GradCAM
print('Initializing GradCAM...')
gradcam = GradCAM(model, target_layer=model.layer4)

# Class names
class_names = ['normal', 'glaucoma']

# Note: ImageFolder assigns labels alphabetically, so:
# 0 = 'glaucoma', 1 = 'normal' (alphabetically)
# But our model expects: 0 = 'normal', 1 = 'glaucoma'
# So we need to flip the predictions

def flip_prediction(pred_class):
    """Flip the prediction to match ImageFolder's alphabetical ordering"""
    return 1 - pred_class  # 0->1, 1->0

# Test images paths
test_glaucoma_dir = script_dir / 'data' / 'test' / 'glaucoma'
test_normal_dir = script_dir / 'data' / 'test' / 'normal'

print(f'\n{"="*60}')
print('GradCAM Visualization')
print(f'{"="*60}\n')

# Visualize some glaucoma cases
print('Visualizing Glaucoma Cases:')
print('-' * 60)
glaucoma_images = list(test_glaucoma_dir.glob('*.png'))
random.shuffle(glaucoma_images)
glaucoma_images = glaucoma_images[:3]  # First 3 random glaucoma images

for i, image_path in enumerate(glaucoma_images, 1):
    print(f'\nGlaucoma Image {i}: {image_path.name}')
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    model_input = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(model_input)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_class = flip_prediction(pred_class)  # Flip to correct label
        confidence = probs[0, 1 - pred_class].item()  # Get confidence for flipped class
    
    print(f'Predicted: {class_names[pred_class]} (Confidence: {confidence:.4f})')
    
    # Visualize
    save_path = f'gradcam_glaucoma_{i}.png'
    gradcam.visualize(
        image_path=str(image_path),
        model_input=model_input,
        class_names=class_names,
        save_path=save_path
    )
    print(f'Saved to: {save_path}')

# Visualize some normal cases
print('\n' + '='*60)
print('Visualizing Normal Cases:')
print('-' * 60)
normal_images = list(test_normal_dir.glob('*.png'))
random.shuffle(normal_images)
normal_images = normal_images[:3]  # First 3 random normal images

for i, image_path in enumerate(normal_images, 1):
    print(f'\nNormal Image {i}: {image_path.name}')
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    model_input = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(model_input)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_class = flip_prediction(pred_class)  # Flip to correct label
        confidence = probs[0, 1 - pred_class].item()  # Get confidence for flipped class
    
    print(f'Predicted: {class_names[pred_class]} (Confidence: {confidence:.4f})')
    
    # Visualize
    save_path = f'gradcam_normal_{i}.png'
    gradcam.visualize(
        image_path=str(image_path),
        model_input=model_input,
        class_names=class_names,
        save_path=save_path
    )
    print(f'Saved to: {save_path}')

print('\n' + '='*60)
print('GradCAM visualizations complete!')
print('='*60)
