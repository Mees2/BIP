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

# Initialize GradCAM
print('Initializing GradCAM...')
gradcam = GradCAM(model, target_layer=model.layer4)

# Get test transform
transform = get_test_transform()

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
