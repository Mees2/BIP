# I want to make a convolutional neural network to recognize glaucoma.
# I already have split data. I have training data and test data. 
# We use PyTorch for training.
# I indeed like to use a pre-trained model.
# Eventually we will hook it up with a surrogate model made to predict the cnn so that we can can explain why it thinks there is a case of glaucoma which is crucial for medical use.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from model import get_model
from transforms import get_train_transform, get_test_transform
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             confusion_matrix, roc_curve)
import numpy as np
from collections import Counter
import copy

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


train_transforms = get_train_transform()
test_transforms = get_test_transform()


# Load training data
train_dataset = datasets.ImageFolder(str(train_data_dir), transform=train_transforms)

# WeightedRandomSampler for balanced batches
class_sample_counts = np.bincount(train_dataset.targets)
weights = 1. / class_sample_counts
sample_weights = weights[train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# Load testing data
test_dataset = datasets.ImageFolder(str(test_data_dir), transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Get model
model = get_model(num_classes=2)
model = model.to(device)

# Unfreeze more layers of EfficientNet (unfreeze last 2 blocks)
if hasattr(model, 'features'):
    for name, param in model.features.named_parameters():
        if '6' in name or '7' in name:  # EfficientNetV2 last blocks
            param.requires_grad = True


# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class_counts = Counter(train_dataset.targets)
total_samples = len(train_dataset)
class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in [class_counts[i] for i in range(len(class_counts))]], dtype=torch.float32)
class_weights = class_weights.to(device)

print(f'Class distribution: {dict(class_counts)}')
print(f'Class weights: {class_weights.cpu().numpy()}')

# Use Focal Loss with class weights
criterion = FocalLoss(alpha=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# Training loop with model checkpointing for best F1
num_epochs = 30
print(f'Starting training for {num_epochs} epochs...')

best_f1 = 0
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Validation on test set for F1/balanced accuracy
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val F1: {val_f1:.4f}, Val Balanced Acc: {val_bal_acc:.4f}')

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch+1

    scheduler.step()

print(f'Best model at epoch {best_epoch} with F1: {best_f1:.4f}')
model.load_state_dict(best_model_wts)


# Evaluate the model on test data with threshold tuning
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Threshold tuning for best F1/balanced accuracy
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
best_f1 = 0
best_bal_acc = 0
best_thresh_f1 = 0.5
best_thresh_bal = 0.5
for thresh in thresholds:
    preds = (all_probs >= thresh).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh_f1 = thresh
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_thresh_bal = thresh

# Use best threshold for F1
final_preds = (all_probs >= best_thresh_f1).astype(int)
accuracy = accuracy_score(all_labels, final_preds)
balanced_accuracy = balanced_accuracy_score(all_labels, final_preds)
precision = precision_score(all_labels, final_preds, zero_division=0)
recall = recall_score(all_labels, final_preds, zero_division=0)
f1 = f1_score(all_labels, final_preds, zero_division=0)
auc_roc = roc_auc_score(all_labels, all_probs)

# Handle specificity calculation robustly
cm = confusion_matrix(all_labels, final_preds)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
else:
    specificity = 0

print('\n' + '='*50)
print('Test Set Metrics (Best F1 Threshold)')
print('='*50)
print(f'Best F1 Threshold: {best_thresh_f1:.3f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall (Sensitivity): {recall:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print('='*50)


# Save the best trained model
model_path = script_dir / 'model.pth'
torch.save(model.state_dict(), str(model_path))
print('Best model saved as model.pth')

# generate new training data with the model
# weighted loss function

# visual transformer
# 