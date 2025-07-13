# ml-server/liver_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class LiverCancerResNet50(nn.Module):
    def __init__(self):
        super(LiverCancerResNet50, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')  # Using pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification (2 classes)
    
    def forward(self, x):
        return self.model(x)

class LiverCancerInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LiverCancerResNet50().to(self.device)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict = False)
        self.model.eval()
        
        # Define the same transforms used in training validation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_labels = ["noCancer", "Cancer"]
        self.class_mapping = {'noCancer': 0, 'Cancer': 1}
    
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
                cancer_probability = float(probabilities[1])  # Cancer class probability
            
            return prediction_label, confidence, cancer_probability, probabilities
            
        except Exception as e:
            raise Exception(f"Error in liver cancer prediction: {str(e)}")