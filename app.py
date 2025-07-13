from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from flask_cors import CORS
import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configuration Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "roi_ouput")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
GRADCAM_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gradcam_output.png")
YOLO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.png")

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

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorEfficientNet().to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

class_labels = ["glioma", "meningioma", "pituitary", "notumour"]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def generate_gradcam(image_tensor, predicted_class):
    model.zero_grad()
    output = model(image_tensor)
    class_score = output[0, predicted_class]
    class_score.backward()
    gradients = model.gradients.cpu().data.numpy()[0]  # Get stored gradients
    feature_maps = model.model.features(image_tensor).cpu().data.numpy()[0]  # Get feature maps
    weights = np.mean(gradients, axis=(1, 2))  # Global Average Pooling
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def overlay_heatmap(original, heatmap):
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay

def predict_with_visualization_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    heatmap = generate_gradcam(image_tensor, predicted_class)
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    overlayed_image = overlay_heatmap(image_cv, heatmap)
    cv2.imwrite(GRADCAM_OUTPUT_PATH, overlayed_image)
    return class_labels[predicted_class], GRADCAM_OUTPUT_PATH

from ultralytics import YOLO

def run_yolo_inference_from_bytes(image_bytes):
    model = YOLO(YOLO_MODEL_PATH)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image_cv2)
    confidence_scores = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            confidence_scores.append(confidence)
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(YOLO_OUTPUT_PATH, image_cv2)
    return confidence_scores

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    file_bytes = file.read()
    predicted_label, saved_image_path = predict_with_visualization_from_bytes(file_bytes)
    confidence_values = run_yolo_inference_from_bytes(file_bytes)
    confidence_str = str(confidence_values[0]) if confidence_values else "0.0"
    buffered = io.BytesIO()
    image = Image.open(io.BytesIO(file_bytes))
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({'diagnosis': predicted_label, 'confidence': confidence_str, 'image': img_base64})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
