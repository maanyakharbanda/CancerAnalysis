import os
import numpy as np

def preprocess_breast(pil_image):
    """Breast cancer specific preprocessing - return PIL image for breast model"""
    return pil_image  # Breast model handles its own preprocessing

def preprocess_lung(pil_image):
    """Lung cancer specific preprocessing - return PIL image for lung model"""
    return pil_image  # Lung model handles its own preprocessing

def preprocess_brain(pil_image):
    """Brain tumor specific preprocessing - just return PIL image for brain model"""
    return pil_image  # Brain model handles its own preprocessing

def preprocess_prostate(img):
    img = img.resize((224, 224)).convert("L")
    import numpy as np
    return np.expand_dims(np.array(img), axis=-1) / 255.0

def preprocess_pancreatic(pil_image):
    """Pancreatic cancer specific preprocessing - return PIL image for pancreatic model"""
    return pil_image  # Pancreatic model handles its own preprocessing

def preprocess_liver(pil_image):
    """Liver cancer specific preprocessing - return PIL image for liver model"""
    return pil_image  # Liver model handles its own preprocessing

def preprocess_esophagus(pil_image):
    """Esophagus cancer specific preprocessing - return PIL image for esophagus model"""
    return pil_image  # Esophagus model handles its own preprocessing


# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Map cancer types to model paths and preprocessors
MODEL_PATHS = {
    "breast-cancer": os.path.join(BASE_DIR, "models", "breast cancer", "best_resnet50_augmented_model.pth"),
    "lung-cancer": os.path.join(BASE_DIR, "models", "lung cancer", "best_resnet50_augmented_model.pth"),
    "brain-tumor": os.path.join(BASE_DIR, "models", "Brain", "best_model.pth"),
    "pancreatic-cancer": os.path.join(BASE_DIR, "models", "pancreatic cancer", "best_resnet50_multiclass.pth"),
    "liver-cancer": os.path.join(BASE_DIR, "models", "liver cancer", "resnet50_liver_binary.pth"),
    "esophagus-cancer": os.path.join(BASE_DIR, "models", "esophagus cancer", "best_esophagus_model.pth"),
    "prostate-cancer": os.path.join(BASE_DIR, "models", "prostate_model.pth")
}


PREPROCESS_FUNCS = {
    "breast-cancer": preprocess_breast,
    "lung-cancer": preprocess_lung,
    "brain-tumor": preprocess_brain,
    "prostate-cancer": preprocess_prostate,
    "pancreatic-cancer": preprocess_pancreatic,
    "liver-cancer": preprocess_liver,
    "esophagus-cancer": preprocess_esophagus
}

YOLO_PATHS = {
    "brain-tumor": os.path.join(BASE_DIR, "models", "Brain", "best.pt"),
}