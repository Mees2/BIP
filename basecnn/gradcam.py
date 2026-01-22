import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
def flip_prediction(pred):
    """
    Flip the prediction index for correct class naming.
    Args:
        pred: Original prediction index (0 or 1)
    Returns:
        flipped_pred: Flipped prediction index
    """
    return 1 - pred


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visualizing 
    which regions of an image are important for model predictions.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: The layer to compute gradients for (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        """Hook to save activations"""
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, 224, 224)
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (224, 224)
            pred_class: Predicted class index
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()
        
        # Compute Grad-CAM
        # Shape of gradients: (1, 2048, 7, 7) for ResNet50
        # Shape of activations: (1, 2048, 7, 7) for ResNet50
        gradients = self.gradients[0]  # (2048, 7, 7)
        activations = self.activations[0]  # (2048, 7, 7)
        
        # Compute weights (average gradient)
        weights = gradients.mean(dim=(1, 2))  # (2048,)
        
        # Compute weighted activation
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        
        # Apply ReLU to get only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), class_idx
    
    def visualize(self, image_path, model_input, class_names=['normal', 'glaucoma'], save_path=None):
        """
        Visualize Grad-CAM overlay on original image
        
        Args:
            image_path: Path to original image
            model_input: Model input tensor (1, 3, 224, 224)
            class_names: List of class names
            save_path: Path to save visualization (optional)
        """
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        original_img_np = np.array(original_img)
        
        # Generate CAM
        cam, pred_class = self.generate_cam(model_input)
        
        # Resize CAM to match image size
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
        cam_resized = np.array(cam_resized) / 255.0
        
        # Create heatmap using matplotlib colormap
        colormap = cm.get_cmap('jet')
        heatmap = colormap(cam_resized)[:, :, :3]  # RGB only
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay on original image
        overlay = (0.6 * original_img_np + 0.4 * heatmap).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (Predicted: {class_names[flip_prediction(pred_class)]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return pred_class
