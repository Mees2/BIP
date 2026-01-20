# I want to make a convolutional neural network to recognize glaucoma.
# I already have split data. I have training data and test data. 
# We use PyTorch for training.
# I indeed like to use a pre-trained model.
# Eventually we will hook it up with a surrogate model made to predict the cnn so that we can can explain why it thinks there is a case of glaucoma which is crucial for medical use.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import get_model
from transforms import get_train_transform, get_test_transform
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             confusion_matrix)
import numpy as np
from collections import Counter

# Get the directory of the current script
script_dir = Path(__file__).parent

# Define paths to training and testing data
train_data_dir = script_dir / 'data' / 'train'
test_data_dir = script_dir / 'data' / 'test'

# Image dimensions
img_height, img_width = 224, 224
batch_size = 32

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data augmentation and normalization for training
train_transforms = get_train_transform()
test_transforms = get_test_transform()

# Load training data
train_dataset = datasets.ImageFolder(str(train_data_dir), transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load testing data
test_dataset = datasets.ImageFolder(str(test_data_dir), transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get model
model = get_model(num_classes=2)
model = model.to(device)

# Calculate class weights for imbalanced data
class_counts = Counter(train_dataset.targets)
total_samples = len(train_dataset)
class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in [class_counts[i] for i in range(len(class_counts))]], dtype=torch.float32)
class_weights = class_weights.to(device)

print(f'Class distribution: {dict(class_counts)}')
print(f'Class weights: {class_weights.cpu().numpy()}')

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Training loop
num_epochs = 30
print(f'Starting training for {num_epochs} epochs...')

best_f1 = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    # Calculate F1 on training set for scheduler
    model.eval()
    epoch_preds = []
    epoch_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
    
    epoch_f1 = f1_score(epoch_labels, epoch_preds, zero_division=0)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, F1: {epoch_f1:.4f}')
    
    # Update learning rate based on F1 score
    scheduler.step(epoch_f1)

# Evaluate the model on training data
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

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
print('\n' + '='*50)
print('Training Set Metrics')
print('='*50)
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall (Sensitivity): {recall:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print('='*50)

# Save the trained model
model_path = script_dir / 'model.pth'
torch.save(model.state_dict(), str(model_path))
print('Model saved as model.pth')

# generate new training data with the model
# weighted loss function

# visual transformer
# 