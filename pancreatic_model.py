# ml-server/pancreatic_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class PancreaticCancerResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(PancreaticCancerResNet50, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')  # Using pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 5-class classification
    
    def forward(self, x):
        return self.model(x)

class PancreaticCancerInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PancreaticCancerResNet50().to(self.device)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Define the same transforms used in training validation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class mapping as defined in training
        self.class_mapping = {
            'Non-Cancer': 0,
            'Cancer_Stage1': 1,
            'Cancer_Stage2': 2,
            'Cancer_Stage3': 3,
            'Cancer_Stage4': 4
        }
        
        # Reverse mapping for prediction labels
        self.class_labels = {v: k for k, v in self.class_mapping.items()}
    
    def preprocess_image(self, pil_image):
        """Preprocess PIL image for ResNet50"""
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def predict(self, pil_image):
        """Main prediction function"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(pil_image)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Get predicted class and confidence
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                prediction_label = self.class_labels[predicted_class]
                
                # Calculate cancer probability (all cancer stages combined)
                cancer_probability = float(np.sum(probabilities[1:]))  # Sum of all cancer stages
            
            return prediction_label, confidence, cancer_probability, probabilities
            
        except Exception as e:
            raise Exception(f"Error in pancreatic cancer prediction: {str(e)}")
    
    def get_stage_info(self, prediction_label):
        """Get detailed information about cancer stage"""
        stage_info = {
            'Non-Cancer': {
                'description': 'No malignant findings detected',
                'prognosis': 'Excellent',
                'treatment': 'Routine monitoring'
            },
            'Cancer_Stage1': {
                'description': 'Early stage pancreatic cancer, localized to pancreas',
                'prognosis': 'Good with treatment',
                'treatment': 'Surgical resection (Whipple procedure), adjuvant chemotherapy'
            },
            'Cancer_Stage2': {
                'description': 'Cancer has spread to nearby tissues or lymph nodes',
                'prognosis': 'Moderate with aggressive treatment',
                'treatment': 'Surgery if resectable, neoadjuvant chemotherapy, radiation'
            },
            'Cancer_Stage3': {
                'description': 'Cancer has spread to major blood vessels or multiple lymph nodes',
                'prognosis': 'Poor, but treatment can extend life',
                'treatment': 'Chemotherapy, radiation, palliative surgery'
            },
            'Cancer_Stage4': {
                'description': 'Cancer has metastasized to distant organs',
                'prognosis': 'Very poor',
                'treatment': 'Palliative chemotherapy, supportive care, clinical trials'
            }
        }
        return stage_info.get(prediction_label, {})