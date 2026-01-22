"""
Surrogate Model for CNN Explainability
========================================
This module creates an interpretable surrogate model that mimics the CNN's predictions.
It uses feature extraction from the CNN (before the classifier) and trains a decision tree
or linear model to approximate the CNN's behavior, providing human-readable explanations.

This is crucial for medical applications where doctors need to understand the model's reasoning.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from model import get_model
from transforms import get_test_transform, get_train_transform
from PIL import Image


class FeatureExtractor:
    """Extract CNN features (embeddings) without classification head"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.features = None
        
        # Register hook to extract features before classifier
        if hasattr(model, 'classifier'):
            # EfficientNet has a classifier head
            model.classifier[0].register_forward_hook(self._hook)
        elif hasattr(model, 'fc'):
            # ResNet has an fc layer
            model.fc[0].register_forward_hook(self._hook)
    
    def _hook(self, module, input, output):
        """Store the output from the layer before classification"""
        self.features = input[0].detach()
    
    def extract(self, images):
        """Extract features from images"""
        with torch.no_grad():
            _ = self.model(images)
        return self.features.cpu().numpy()


class SurrogateModel:
    """Surrogate model for CNN explainability"""
    
    def __init__(self, cnn_model, device, surrogate_type='tree'):
        """
        Args:
            cnn_model: Trained CNN model
            device: torch device
            surrogate_type: 'tree' (DecisionTree) or 'linear' (LogisticRegression)
        """
        self.cnn_model = cnn_model
        self.device = device
        self.surrogate_type = surrogate_type
        self.feature_extractor = FeatureExtractor(cnn_model, device)
        self.surrogate = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def train(self, train_loader, max_depth=10):
        """
        Train surrogate model on CNN features
        
        Args:
            train_loader: DataLoader for training
            max_depth: Max depth for decision tree
        """
        print(f"Extracting features from CNN for surrogate training...")
        
        all_features = []
        all_labels = []
        all_cnn_preds = []
        
        self.cnn_model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                
                # Extract CNN predictions
                outputs = self.cnn_model(images)
                cnn_preds = outputs.argmax(dim=1).cpu().numpy()
                
                # Extract features
                features = self.feature_extractor.extract(images)
                
                all_features.append(features)
                all_labels.append(labels.numpy())
                all_cnn_preds.append(cnn_preds)
        
        X = np.vstack(all_features)
        y_cnn = np.hstack(all_cnn_preds)  # Use CNN predictions as targets
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create feature names
        self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Train surrogate
        if self.surrogate_type == 'tree':
            self.surrogate = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:  # linear
            self.surrogate = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        
        self.surrogate.fit(X_scaled, y_cnn)
        
        # Evaluate surrogate
        surrogate_preds = self.surrogate.predict(X_scaled)
        accuracy = accuracy_score(y_cnn, surrogate_preds)
        f1 = f1_score(y_cnn, surrogate_preds, zero_division=0)
        
        print(f"Surrogate Model Trained ({self.surrogate_type})")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Fidelity (how well it mimics CNN): {accuracy:.2%}")
        
        return accuracy
    
    def predict(self, image_path, class_names=['normal', 'glaucoma']):
        """
        Make prediction and provide explanation
        
        Args:
            image_path: Path to image
            class_names: List of class names
        
        Returns:
            dict with prediction, explanation, and confidence
        """
        # Load and preprocess image
        transform = get_test_transform()
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get CNN prediction
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_output = self.cnn_model(image_tensor)
            cnn_pred = cnn_output.argmax(dim=1).item()
            cnn_probs = torch.softmax(cnn_output, dim=1)[0]
        
        # Get CNN features
        features = self.feature_extractor.extract(image_tensor)
        X_scaled = self.scaler.transform(features)
        
        # Get surrogate prediction
        surrogate_pred = self.surrogate.predict(X_scaled)[0]
        surrogate_probs = self.surrogate.predict_proba(X_scaled)[0]
        
        # Generate explanation
        explanation = self._generate_explanation(
            X_scaled[0], surrogate_pred, class_names
        )
        
        return {
            'cnn_prediction': class_names[cnn_pred],
            'cnn_confidence': cnn_probs[cnn_pred].item(),
            'surrogate_prediction': class_names[surrogate_pred],
            'surrogate_confidence': surrogate_probs[surrogate_pred],
            'explanation': explanation,
            'agreement': cnn_pred == surrogate_pred
        }
    
    def _generate_explanation(self, features, prediction, class_names):
        """
        Generate human-readable explanation for prediction
        
        Args:
            features: Scaled feature vector
            prediction: Predicted class
            class_names: List of class names
        
        Returns:
            str: Explanation text
        """
        if self.surrogate_type == 'tree':
            return self._explain_tree(prediction, class_names)
        else:
            return self._explain_linear(features, prediction, class_names)
    
    def _explain_tree(self, prediction, class_names):
        """Explain decision tree prediction"""
        tree_rules = export_text(
            self.surrogate,
            feature_names=self.feature_names
        )
        
        explanation = f"""
DECISION TREE EXPLANATION
==========================
Predicted Class: {class_names[prediction]}

Decision Path:
{tree_rules}

Interpretation:
The decision tree follows a logical path through feature thresholds
to arrive at this prediction. Each node represents a decision based
on specific feature values extracted from the retinal image.
"""
        return explanation
    
    def _explain_linear(self, features, prediction, class_names):
        """Explain logistic regression prediction"""
        coef = self.surrogate.coef_[0]
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(coef))[-5:][::-1]
        
        explanation = f"""
LOGISTIC REGRESSION EXPLANATION
================================
Predicted Class: {class_names[prediction]}

Top 5 Most Influential Features:
"""
        for rank, idx in enumerate(top_indices, 1):
            direction = "increases" if coef[idx] > 0 else "decreases"
            magnitude = abs(coef[idx])
            explanation += f"\n  {rank}. Feature {idx}: {direction} likelihood (strength: {magnitude:.4f})"
        
        explanation += f"""

Interpretation:
The model uses logistic regression to combine image features.
The features above have the strongest influence on whether
the model classifies the retina as having glaucoma or being normal.
Positive weights favor glaucoma classification, negative weights favor normal.
"""
        return explanation
    
    def get_feature_importance(self):
        """
        Get feature importance ranking
        
        Returns:
            dict: Feature importance scores
        """
        if self.surrogate_type == 'tree':
            importances = self.surrogate.feature_importances_
        else:
            importances = np.abs(self.surrogate.coef_[0])
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        return {
            'feature_names': [self.feature_names[i] for i in sorted_idx],
            'importances': importances[sorted_idx],
            'top_features': sorted_idx[:10]
        }


def main():
    """Example usage of surrogate model"""
    
    script_dir = Path(__file__).parent
    train_data_dir = script_dir / 'data' / 'train'
    test_data_dir = script_dir / 'data' / 'test'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Load trained CNN
    print('Loading CNN model...')
    cnn_model = get_model(num_classes=2)
    model_path = script_dir / 'model.pth'
    cnn_model.load_state_dict(torch.load(str(model_path), map_location=device))
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # Create surrogate model
    print('Creating surrogate model (Decision Tree)...\n')
    surrogate = SurrogateModel(cnn_model, device, surrogate_type='tree')
    
    # Load training data for surrogate training
    train_dataset = datasets.ImageFolder(str(train_data_dir), transform=get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # Train surrogate
    surrogate.train(train_loader, max_depth=12)
    
    # Get feature importance
    print('\n' + '='*60)
    print('Feature Importance')
    print('='*60)
    importance = surrogate.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    for i, (name, imp) in enumerate(zip(importance['feature_names'][:10], 
                                        importance['importances'][:10]), 1):
        print(f"  {i}. {name}: {imp:.6f}")
    
    # Example prediction with explanation
    print('\n' + '='*60)
    print('Example Predictions with Explanations')
    print('='*60)
    
    test_dataset = datasets.ImageFolder(str(test_data_dir), transform=get_test_transform())
    test_images_normal = [p for p in (test_data_dir / 'normal').glob('*.png')][:1]
    test_images_glaucoma = [p for p in (test_data_dir / 'glaucoma').glob('*.png')][:1]
    
    for image_path in test_images_normal + test_images_glaucoma:
        print(f"\nImage: {image_path.name}")
        print('-' * 60)
        result = surrogate.predict(str(image_path))
        print(f"CNN Prediction: {result['cnn_prediction']} (confidence: {result['cnn_confidence']:.4f})")
        print(f"Surrogate Prediction: {result['surrogate_prediction']} (confidence: {result['surrogate_confidence']:.4f})")
        print(f"Agreement: {'✓ Yes' if result['agreement'] else '✗ No'}")
        print(result['explanation'])


if __name__ == '__main__':
    main()
