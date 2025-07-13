import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import os
from ultralytics import YOLO

# Load Pretrained EfficientNet & Modify for 4 Classes
class BrainTumorEfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorEfficientNet, self).__init__()
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.gradients = None  # Store gradients for Grad-CAM

    def save_gradient(self, grad):
        self.gradients = grad  # Hook function to store gradients

    def forward(self, x):
        x = self.model.features(x)  # Extract features
        if not x.requires_grad:
            x.requires_grad_()
        x.register_hook(self.save_gradient)  # Register hook for Grad-CAM
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

class BrainTumorInference:
    def __init__(self, model_path, yolo_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BrainTumorEfficientNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.class_labels = ["glioma", "meningioma", "pituitary", "notumour"]
        
        # Load YOLO if path provided
        self.yolo_model = None
        if yolo_path and os.path.exists(yolo_path):
            self.yolo_model = YOLO(yolo_path)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def generate_gradcam(self, image_tensor, predicted_class):
        self.model.zero_grad()
        output = self.model(image_tensor)
        class_score = output[0, predicted_class]
        class_score.backward()
        gradients = self.model.gradients.cpu().data.numpy()[0]
        feature_maps = self.model.model.features(image_tensor).cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (256, 256))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def overlay_heatmap(self, original, heatmap):
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        return overlay

    def predict_with_gradcam(self, pil_image):
        """Main prediction function with Grad-CAM visualization"""
        # Preprocess image
        image_tensor = self.preprocess_image(pil_image)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = float(probabilities[predicted_class])
        
        # Generate Grad-CAM
        heatmap = self.generate_gradcam(image_tensor, predicted_class)
        
        # Convert PIL to CV2 format
        image_cv = np.array(pil_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Create overlay
        overlayed_image = self.overlay_heatmap(image_cv, heatmap)
        
        # Convert back to PIL for upload
        overlayed_pil = Image.fromarray(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
        
        prediction_label = self.class_labels[predicted_class]
        
        return prediction_label, confidence, overlayed_pil, probabilities

    def run_yolo_detection(self, pil_image):
        """Run YOLO detection if model is available"""
        if not self.yolo_model:
            return [], None
        
        image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results = self.yolo_model(image_cv2)
        confidence_scores = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                confidence_scores.append(confidence)
                cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert back to PIL
        yolo_result_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        
        return confidence_scores, yolo_result_pil