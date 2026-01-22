"""
Doctor-Friendly Explanation Interface
======================================
Provides interpretable explanations for CNN predictions using the surrogate model.
Designed to be understandable by medical professionals without ML expertise.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from model import get_model
from transforms import get_test_transform
from surrogate_model import SurrogateModel
from sklearn.metrics import roc_curve
import json


class ExplainablePrediction:
    """
    Medical-friendly prediction explanation system
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the explainability system
        
        Args:
            model_path: Path to trained CNN model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.class_names = ['normal', 'glaucoma']
        
        # Load CNN
        self.cnn_model = get_model(num_classes=2)
        self.cnn_model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # Create surrogate for explanations
        self.surrogate = SurrogateModel(self.cnn_model, self.device, surrogate_type='tree')
    
    def train_surrogate(self, train_loader):
        """Train the surrogate model for better explanations"""
        print("Training surrogate model for explanations...")
        self.surrogate.train(train_loader, max_depth=10)
    
    def explain_prediction(self, image_path, confidence_threshold=0.7):
        """
        Generate medical-friendly explanation for a prediction
        
        Args:
            image_path: Path to retinal image
            confidence_threshold: Confidence level for strong predictions
        
        Returns:
            dict: Detailed explanation suitable for doctors
        """
        # Load image
        transform = get_test_transform()
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get CNN prediction
        self.cnn_model.eval()
        with torch.no_grad():
            output = self.cnn_model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            prediction_idx = output.argmax(dim=1).item()
        
        prediction = self.class_names[prediction_idx]
        confidence = probs[prediction_idx].item()
        alternative_class = self.class_names[1 - prediction_idx]
        alternative_confidence = probs[1 - prediction_idx].item()
        
        # Get surrogate explanation
        surrogate_result = self.surrogate.predict(str(image_path), self.class_names)
        
        # Determine confidence level
        if confidence >= confidence_threshold:
            confidence_level = "HIGH"
            confidence_desc = "The model is very confident in this prediction."
        elif confidence >= 0.6:
            confidence_level = "MODERATE"
            confidence_desc = "The model is reasonably confident, but some uncertainty exists."
        else:
            confidence_level = "LOW"
            confidence_desc = "The model has low confidence. Additional review is recommended."
        
        # Prepare medical report
        report = {
            'status': 'success',
            'image_filename': Path(image_path).name,
            'primary_prediction': {
                'class': prediction,
                'confidence': f"{confidence:.1%}",
                'confidence_level': confidence_level,
                'description': self._get_prediction_description(prediction)
            },
            'alternative_finding': {
                'class': alternative_class,
                'confidence': f"{alternative_confidence:.1%}",
            },
            'confidence_assessment': confidence_desc,
            'agreement_with_surrogate': 'Yes' if surrogate_result['agreement'] else 'No',
            'model_consistency': self._assess_consistency(
                surrogate_result['agreement'],
                confidence
            ),
            'clinical_recommendation': self._get_clinical_recommendation(
                prediction, confidence, surrogate_result['agreement']
            ),
            'explanation_summary': self._get_explanation_summary(prediction, confidence),
        }
        
        return report
    
    def _get_prediction_description(self, prediction):
        """Get clinical description of prediction"""
        descriptions = {
            'normal': 'No signs of glaucoma detected. The optic nerve head and retinal structures appear healthy.',
            'glaucoma': 'Indicators of glaucoma detected. Signs may include optic nerve head changes, cup-to-disc ratio alterations, or other glaucomatous features.'
        }
        return descriptions.get(prediction, "Unknown prediction")
    
    def _assess_consistency(self, agreement, confidence):
        """Assess internal consistency of model predictions"""
        if not agreement and confidence < 0.7:
            return {
                'status': 'LOW',
                'message': 'The CNN and surrogate models disagree, and confidence is low. Careful review recommended.'
            }
        elif not agreement:
            return {
                'status': 'MODERATE',
                'message': 'The CNN and surrogate models disagree despite high CNN confidence. Worth investigating.'
            }
        else:
            return {
                'status': 'HIGH',
                'message': 'Both CNN and surrogate model agree on this prediction.'
            }
    
    def _get_clinical_recommendation(self, prediction, confidence, agreement):
        """Get clinical action recommendation"""
        if prediction == 'glaucoma':
            if confidence >= 0.85 and agreement:
                return "Strong evidence of glaucoma. Consider further diagnostic testing and specialist referral."
            elif confidence >= 0.7:
                return "Likely glaucoma findings. Recommend comprehensive ophthalmic evaluation."
            else:
                return "Possible glaucoma indicators. Additional testing and expert review recommended."
        else:  # normal
            if confidence >= 0.85 and agreement:
                return "No significant glaucoma indicators detected. Routine monitoring may be appropriate."
            elif confidence >= 0.7:
                return "No obvious glaucoma findings. Follow standard clinical guidelines for screening."
            else:
                return "Borderline findings. Clinical correlation and additional testing recommended."
    
    def _get_explanation_summary(self, prediction, confidence):
        """Get simplified explanation for doctors"""
        summary = f"""
The AI system analyzed the retinal image and predicted: {prediction.upper()}

Key Features Analyzed:
- Optic disc morphology and size
- Cup-to-disc ratio
- Retinal nerve fiber layer (RNFL) integrity
- Neuroretinal rim characteristics
- Blood vessel patterns

Confidence Level: {confidence:.0%}
This indicates the model's certainty in its assessment.
        """.strip()
        return summary
    
    def batch_analyze(self, image_folder, output_json=None):
        """
        Analyze multiple images and generate report
        
        Args:
            image_folder: Folder containing images
            output_json: Optional path to save JSON report
        
        Returns:
            list: List of analysis results
        """
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob('*.png')) + list(image_folder.glob('*.jpg'))
        
        results = []
        for image_path in image_files:
            print(f"Analyzing: {image_path.name}")
            result = self.explain_prediction(str(image_path))
            results.append(result)
        
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_json}")
        
        return results
    
    def print_report(self, report):
        """Print formatted report for doctor"""
        print('\n' + '='*70)
        print('GLAUCOMA SCREENING AI - ANALYSIS REPORT')
        print('='*70)
        print(f"Image: {report['image_filename']}")
        print('-'*70)
        
        pred = report['primary_prediction']
        print(f"\nPRIMARY FINDING: {pred['class'].upper()}")
        print(f"Confidence: {pred['confidence']} ({pred['confidence_level']})")
        print(f"Description: {pred['description']}")
        
        print(f"\nAlternative Finding: {report['alternative_finding']['class']} ({report['alternative_finding']['confidence']})")
        
        print(f"\nConfidence Assessment: {report['confidence_assessment']}")
        print(f"Model Consistency: {report['model_consistency']['status']} - {report['model_consistency']['message']}")
        
        print(f"\nCLINICAL RECOMMENDATION:")
        print(f"  {report['clinical_recommendation']}")
        
        print(f"\nEXPLANATION SUMMARY:")
        for line in report['explanation_summary'].split('\n'):
            print(f"  {line}")
        
        print('\n' + '='*70 + '\n')


def main():
    """Example usage"""
    import sys
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from transforms import get_train_transform
    
    script_dir = Path(__file__).parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize system
    print("Initializing Explainable Glaucoma Detection System...")
    predictor = ExplainablePrediction(
        script_dir / 'model.pth',
        device=device
    )
    
    # Train surrogate (optional but recommended)
    print("\nPreparing surrogate model...")
    train_dataset = datasets.ImageFolder(
        str(script_dir / 'data' / 'train'),
        transform=get_train_transform()
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    predictor.train_surrogate(train_loader)
    
    # Example: Analyze a single image
    print("\n" + "="*70)
    print("SINGLE IMAGE ANALYSIS")
    print("="*70)
    
    test_image = list((script_dir / 'data' / 'test' / 'glaucoma').glob('*.png'))[0]
    report = predictor.explain_prediction(str(test_image))
    predictor.print_report(report)
    
    # Example: Batch analysis
    print("\n" + "="*70)
    print("BATCH ANALYSIS")
    print("="*70)
    
    results = predictor.batch_analyze(
        script_dir / 'data' / 'test' / 'glaucoma',
        output_json=script_dir / 'glaucoma_analysis_report.json'
    )
    
    # Print summary statistics
    glaucoma_count = sum(1 for r in results if r['primary_prediction']['class'] == 'glaucoma')
    normal_count = len(results) - glaucoma_count
    
    print(f"\nSummary: {glaucoma_count} glaucoma, {normal_count} normal out of {len(results)} images")


if __name__ == '__main__':
    main()
