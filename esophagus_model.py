# ml-server/esophagus_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class EsophagusCancerResNet50(nn.Module):
    def __init__(self):
        super(EsophagusCancerResNet50, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')  # Using pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Binary classification
    
    def forward(self, x):
        return self.model(x)

class EsophagusCancerInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EsophagusCancerResNet50().to(self.device)
        
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
        
        self.class_labels = ["Non-Cancer", "Cancer"]
    
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
                output = self.model(image_tensor)
                probability = torch.sigmoid(output).cpu().numpy()[0][0]
                
                # Binary classification logic
                predicted_class = 1 if probability > 0.5 else 0
                confidence = probability if predicted_class == 1 else (1 - probability)
                prediction_label = self.class_labels[predicted_class]
            
            return prediction_label, float(confidence), float(probability)
            
        except Exception as e:
            raise Exception(f"Error in esophagus cancer prediction: {str(e)}")