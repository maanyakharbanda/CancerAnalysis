from cloudinary_utils import upload_image_to_cloudinary
from PIL import Image, ImageDraw
import torch
import torchvision.models as models
from brain_model import BrainTumorInference
from breast_model import BreastCancerInference
from esophagus_model import EsophagusCancerInference 
from liver_model import LiverCancerInference
from lung_model import LungCancerInference
from pancreatic_model import PancreaticCancerInference 
from cancer_config import MODEL_PATHS, YOLO_PATHS
import numpy as np


def run_model(model_path, processed_img):
    # Check cancer type from model path
    if "brain" in model_path.lower() or "Brain" in model_path:
        return run_brain_model(model_path, processed_img)
    elif "breast" in model_path.lower():
        return run_breast_model(model_path, processed_img)
    elif "esophagus" in model_path.lower():
        return run_esophagus_model(model_path, processed_img)
    elif "liver" in model_path.lower():
        return run_liver_model(model_path, processed_img)
    elif "lung" in model_path.lower():
        return run_lung_model(model_path, processed_img)
    elif "pancreatic" in model_path.lower():
        return run_pancreatic_model(model_path, processed_img)
    
    # Default dummy logic for other models
    return run_default_model(model_path, processed_img)


def run_lung_model(model_path, pil_image):
    """Run lung cancer inference with ResNet50"""
    try:
        # Initialize lung cancer inference
        lung_inference = LungCancerInference(model_path)
        
        # Get prediction
        prediction_label, confidence, probability = lung_inference.predict(pil_image)
        
        # Generate medical findings
        findings = generate_lung_findings(prediction_label, confidence, probability)
        
        # Create result visualization
        result_image = create_lung_result_image(pil_image, prediction_label, confidence, probability)
        
        # Upload to Cloudinary
        result_image_url = upload_image_to_cloudinary(result_image, public_id=f"lung_{prediction_label.lower().replace(' ', '_')}")
        
        result = {
            "prediction": prediction_label,
            "confidence": confidence,
            "additionalFindings": findings,
            "resultImageUrl": result_image_url,
            "noduleProbability": probability,
            "modelVersion": "ResNet50-Lung-Cancer-v1.0",
            "analysisType": "Chest X-ray/CT Analysis"
        }
        return result
        
    except Exception as e:
        return {
            "prediction": "analysis_error",
            "confidence": 0.0,
            "additionalFindings": [f"Error in lung cancer analysis: {str(e)}"],
            "resultImageUrl": None,
            "modelVersion": "ResNet50-Lung-Cancer-v1.0"
        }


def create_lung_result_image(original_image, prediction, confidence, probability):
    """Create visualization for lung cancer results"""
    # Create a result image
    result_img = Image.new("RGB", (650, 450), color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    # Resize original image to fit
    original_resized = original_image.resize((280, 280))
    result_img.paste(original_resized, (10, 60))
    
    # Add text information
    draw.text((310, 60), "LUNG CANCER ANALYSIS", fill=(0, 0, 0), font=None)
    draw.text((310, 100), f"Prediction: {prediction}", fill=(0, 0, 0))
    draw.text((310, 130), f"Confidence: {confidence:.3f}", fill=(0, 0, 0))
    draw.text((310, 160), f"Nodule Probability: {probability:.3f}", fill=(0, 0, 0))
    
    # Add risk assessment
    if prediction == "Nodule Detected":
        risk_color = (255, 140, 0)  # Orange for nodule
        risk_text = "NODULE DETECTED - Further evaluation needed"
    else:
        risk_color = (0, 150, 0)  # Green
        risk_text = "NO NODULE DETECTED - Continue routine screening"
    
    draw.text((310, 200), risk_text, fill=risk_color)
    
    # Add confidence bar visualization
    bar_width = int(confidence * 200)
    draw.rectangle([310, 240, 310 + bar_width, 260], fill=(0, 100, 200))
    draw.rectangle([310, 240, 510, 260], outline=(0, 0, 0))
    draw.text((310, 270), f"Confidence: {confidence:.1%}", fill=(0, 0, 0))
    
    # Add nodule probability bar
    nodule_bar_width = int(probability * 200)
    draw.rectangle([310, 300, 310 + nodule_bar_width, 320], fill=(255, 140, 0))
    draw.rectangle([310, 300, 510, 320], outline=(0, 0, 0))
    draw.text((310, 330), f"Nodule Probability: {probability:.1%}", fill=(0, 0, 0))
    
    # Add imaging type info
    draw.text((310, 370), "Analysis Type: Chest X-ray/CT Scan", fill=(50, 50, 50))
    
    # Add footer
    draw.text((10, 410), "Note: This is AI-assisted analysis. Always consult a pulmonologist or radiologist.", fill=(100, 100, 100))
    
    return result_img



def run_esophagus_model(model_path, pil_image):
    """Run esophagus cancer inference with ResNet50"""
    try:
        # Initialize esophagus cancer inference
        esophagus_inference = EsophagusCancerInference(model_path)
        
        # Get prediction
        prediction_label, confidence, probability = esophagus_inference.predict(pil_image)
        
        # Generate medical findings
        findings = generate_esophagus_findings(prediction_label, confidence, probability)
        
        # Create result visualization
        result_image = create_esophagus_result_image(pil_image, prediction_label, confidence, probability)
        
        # Upload to Cloudinary
        result_image_url = upload_image_to_cloudinary(result_image, public_id=f"esophagus_{prediction_label.lower()}")
        
        result = {
            "prediction": prediction_label,
            "confidence": confidence,
            "additionalFindings": findings,
            "resultImageUrl": result_image_url,
            "probability": probability,
            "modelVersion": "ResNet50-Esophagus-Cancer-v1.0",
            "analysisType": "Endoscopic Analysis"
        }
        return result
        
    except Exception as e:
        return {
            "prediction": "analysis_error",
            "confidence": 0.0,
            "additionalFindings": [f"Error in esophagus cancer analysis: {str(e)}"],
            "resultImageUrl": None,
            "modelVersion": "ResNet50-Esophagus-Cancer-v1.0"
        }


def create_esophagus_result_image(original_image, prediction, confidence, probability):
    """Create visualization for esophagus cancer results"""
    # Create a result image
    result_img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    # Resize original image to fit
    original_resized = original_image.resize((280, 280))
    result_img.paste(original_resized, (10, 60))
    
    # Add text information
    draw.text((310, 60), "Esophagus CANCER ANALYSIS", fill=(0, 0, 0), font=None)
    draw.text((310, 100), f"Prediction: {prediction}", fill=(0, 0, 0))
    draw.text((310, 130), f"Confidence: {confidence:.3f}", fill=(0, 0, 0))
    draw.text((310, 160), f"Cancer Probability: {probability:.3f}", fill=(0, 0, 0))
    
    # Add risk assessment
    if prediction == "Cancer":
        risk_color = (255, 0, 0)  # Red
        risk_text = "HIGH RISK - Immediate consultation required"
    else:
        risk_color = (0, 150, 0)  # Green
        risk_text = "LOW RISK - Continue routine surveillance"
    draw.text((310, 200), risk_text, fill=risk_color)
    
    # Add confidence bar visualization
    bar_width = int(confidence * 200)
    draw.rectangle([310, 240, 310 + bar_width, 260], fill=(0, 100, 200))
    draw.rectangle([310, 240, 510, 260], outline=(0, 0, 0))
    draw.text((310, 270), f"Confidence: {confidence:.1%}", fill=(0, 0, 0))
    
    # Add endoscopy-specific info
    draw.text((310, 300), "Analysis Type: Endoscopic Imaging", fill=(50, 50, 50))
    
    # Add footer
    draw.text((10, 360), "Note: This is AI-assisted analysis. Always consult a gastroenterologist.", fill=(100, 100, 100))
    
    return result_img


def run_breast_model(model_path, pil_image):
    """Run breast cancer inference with ResNet50"""
    try:
        # Initialize breast cancer inference
        breast_inference = BreastCancerInference(model_path)
        
        # Get prediction
        prediction_label, confidence, probability = breast_inference.predict(pil_image)
        
        # Generate medical findings
        findings = generate_breast_findings(prediction_label, confidence, probability)
        
        # Create result visualization
        result_image = create_breast_result_image(pil_image, prediction_label, confidence, probability)
        
        # Upload to Cloudinary
        result_image_url = upload_image_to_cloudinary(result_image, public_id=f"breast_{prediction_label.lower()}")
        result = {
            "prediction": prediction_label,
            "confidence": confidence,
            "additionalFindings": findings,
            "resultImageUrl": result_image_url,
            "probability": probability,
            "modelVersion": "ResNet50-Breast-Cancer-v1.0",
            "analysisType": "Mammography Analysis"
        }
        return result
        
    except Exception as e:
        return {
            "prediction": "analysis_error",
            "confidence": 0.0,
            "additionalFindings": [f"Error in breast cancer analysis: {str(e)}"],
            "resultImageUrl": None,
            "modelVersion": "ResNet50-Breast-Cancer-v1.0"
        }


def create_breast_result_image(original_image, prediction, confidence, probability):
    """Create visualization for breast cancer results"""
    # Create a result image
    result_img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    # Resize original image to fit
    original_resized = original_image.resize((280, 280))
    result_img.paste(original_resized, (10, 60))
    
    # Add text information
    draw.text((310, 60), "BREAST CANCER ANALYSIS", fill=(0, 0, 0), font=None)
    draw.text((310, 100), f"Prediction: {prediction}", fill=(0, 0, 0))
    draw.text((310, 130), f"Confidence: {confidence:.3f}", fill=(0, 0, 0))
    draw.text((310, 160), f"Cancer Probability: {probability:.3f}", fill=(0, 0, 0))
    
    # Add risk assessment
    if prediction == "Cancer":
        risk_color = (255, 0, 0)  # Red
        risk_text = "HIGH RISK - Immediate consultation required"
    else:
        risk_color = (0, 150, 0)  # Green
        risk_text = "LOW RISK - Continue routine screening"
    
    draw.text((310, 200), risk_text, fill=risk_color)
    
    # Add confidence bar visualization
    bar_width = int(confidence * 200)
    draw.rectangle([310, 240, 310 + bar_width, 260], fill=(0, 100, 200))
    draw.rectangle([310, 240, 510, 260], outline=(0, 0, 0))
    draw.text((310, 270), f"Confidence: {confidence:.1%}", fill=(0, 0, 0))
    
    # Add footer
    draw.text((10, 360), "Note: This is AI-assisted analysis. Always consult a medical professional.", fill=(100, 100, 100))
    
    return result_img


def run_default_model(model_path, processed_img):
    """Default dummy implementation for other cancer types"""
    dummy_prediction = "benign"
    dummy_confidence = 0.87
    
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 90), f"Prediction: {dummy_prediction}", fill=(0, 0, 0))
    d.text((10, 120), f"Confidence: {dummy_confidence:.2f}", fill=(0, 0, 0))
    
    result_image_url = upload_image_to_cloudinary(img)
    
    result = {
        "prediction": dummy_prediction,
        "confidence": dummy_confidence,
        "additionalFindings": ["Analysis completed", "Consult healthcare provider"],
        "resultImageUrl": result_image_url,
        "modelVersion": "v1.0.0"
    }
    return result


def run_brain_model(model_path, pil_image):
    """Run brain tumor inference with EfficientNet + YOLO"""
    try:
        # Get YOLO path if available
        yolo_path = YOLO_PATHS.get("brain-tumor")
        
        # Initialize brain inference
        brain_inference = BrainTumorInference(model_path, yolo_path)
        
        # Run prediction with Grad-CAM
        prediction, confidence, gradcam_image, probabilities = brain_inference.predict_with_gradcam(pil_image)
        
        # Upload Grad-CAM result to Cloudinary
        gradcam_url = upload_image_to_cloudinary(gradcam_image, public_id=f"brain_gradcam_{prediction}")
        
        # Run YOLO detection
        yolo_confidences, yolo_image = brain_inference.run_yolo_detection(pil_image)
        yolo_url = None
        if yolo_image:
            yolo_url = upload_image_to_cloudinary(yolo_image, public_id=f"brain_yolo_{prediction}")
        
        # Generate medical findings
        findings = generate_brain_findings(prediction, confidence, yolo_confidences)

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "additionalFindings": findings,
            "resultImageUrl": gradcam_url,
            "yoloResultUrl": yolo_url,
            "yoloConfidences": yolo_confidences,
            "probabilities": {
                "glioma": float(probabilities[0]),
                "meningioma": float(probabilities[1]),
                "pituitary": float(probabilities[2]),
                "notumour": float(probabilities[3])
            },
            "modelVersion": "EfficientNet-B3 + YOLO"
        }
        return result
        
    except Exception as e:
        return {
            "prediction": "analysis_error",
            "confidence": 0.0,
            "additionalFindings": [f"Error in brain analysis: {str(e)}"],
            "resultImageUrl": None,
            "modelVersion": "EfficientNet-B3 + YOLO"
        }


def run_liver_model(model_path, pil_image):
    """Run liver cancer inference with ResNet50"""
    try:
        # Initialize liver cancer inference
        liver_inference = LiverCancerInference(model_path)
        
        # Get prediction
        prediction_label, confidence, cancer_probability, probabilities = liver_inference.predict(pil_image)
        
        # Generate medical findings
        findings = generate_liver_findings(prediction_label, confidence, cancer_probability, probabilities)
        
        # Create result visualization
        result_image = create_liver_result_image(pil_image, prediction_label, confidence, cancer_probability, probabilities)
        
        # Upload to Cloudinary
        result_image_url = upload_image_to_cloudinary(result_image, public_id=f"liver_{prediction_label.lower()}")
        result = {
            "prediction": prediction_label,
            "confidence": confidence,
            "additionalFindings": findings,
            "resultImageUrl": result_image_url,
            "cancerProbability": cancer_probability,
            "probabilities": {
                "noCancer": float(probabilities[0]),
                "Cancer": float(probabilities[1])
            },
            "modelVersion": "ResNet50-Liver-Cancer-v1.0",
            "analysisType": "Hepatic Imaging Analysis"
        }
        return result
    
    except Exception as e:
        return {
            "prediction": "analysis_error",
            "confidence": 0.0,
            "additionalFindings": [f"Error in liver cancer analysis: {str(e)}"],
            "resultImageUrl": None,
            "modelVersion": "ResNet50-Liver-Cancer-v1.0"
        }


def create_liver_result_image(original_image, prediction, confidence, cancer_probability, probabilities):
    """Create visualization for liver cancer results"""
    # Create a result image
    result_img = Image.new("RGB", (650, 450), color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    # Resize original image to fit
    original_resized = original_image.resize((280, 280))
    result_img.paste(original_resized, (10, 60))
    
    # Add text information
    draw.text((310, 60), "LIVER CANCER ANALYSIS", fill=(0, 0, 0), font=None)
    draw.text((310, 100), f"Prediction: {prediction}", fill=(0, 0, 0))
    draw.text((310, 130), f"Confidence: {confidence:.3f}", fill=(0, 0, 0))
    draw.text((310, 160), f"Cancer Probability: {cancer_probability:.3f}", fill=(0, 0, 0))
    
    # Add risk assessment
    if prediction == "Cancer":
        risk_color = (255, 0, 0)  # Red
        risk_text = "HIGH RISK - Immediate hepatologist consultation"
    else:
        risk_color = (0, 150, 0)  # Green
        risk_text = "LOW RISK - Continue routine monitoring"

    draw.text((310, 200), risk_text, fill=risk_color)
    
    # Add confidence bar visualization
    bar_width = int(confidence * 200)
    draw.rectangle([310, 240, 310 + bar_width, 260], fill=(0, 100, 200))
    draw.rectangle([310, 240, 510, 260], outline=(0, 0, 0))
    draw.text((310, 270), f"Confidence: {confidence:.1%}", fill=(0, 0, 0))
    
    # Add probability breakdown
    draw.text((310, 300), "Probability Breakdown:", fill=(0, 0, 0))
    draw.text((310, 320), f"Non-Cancer: {probabilities[0]:.3f}", fill=(0, 150, 0))
    draw.text((310, 340), f"Cancer: {probabilities[1]:.3f}", fill=(255, 0, 0) if probabilities[1] > 0.5 else (0, 0, 0))
    
    # Add imaging type info
    draw.text((310, 370), "Analysis Type: CT/MRI Hepatic Imaging", fill=(50, 50, 50))
    
    # Add footer
    draw.text((10, 410), "Note: This is AI-assisted analysis. Always consult a hepatologist or oncologist.", fill=(100, 100, 100))
    
    return result_img



def generate_brain_findings(prediction, confidence, yolo_confidences):
    """Generate medical findings for brain tumor"""
    findings = []
    
    if prediction == "glioma":
        findings = [
            "Glioma detected - Aggressive brain tumor requiring immediate attention",
            "Recommend urgent neurosurgical consultation",
            "MRI with contrast strongly recommended",
            "Molecular profiling may be needed for treatment planning"
        ]
    elif prediction == "meningioma":
        findings = [
            "Meningioma detected - Usually benign tumor of brain membranes",
            "Growth rate monitoring recommended",
            "Neurosurgical evaluation advised",
            "Consider hormonal factors in female patients"
        ]
    elif prediction == "pituitary":
        findings = [
            "Pituitary adenoma detected",
            "Endocrine function assessment required",
            "Visual field examination recommended",
            "Consider prolactin and growth hormone levels"
        ]
    else:  # notumour
        findings = [
            "No tumor detected in brain scan",
            "Normal brain anatomy visualized",
            "Continue routine clinical monitoring"
        ]
    
    # Add YOLO detection results
    if yolo_confidences:
        avg_yolo_conf = sum(yolo_confidences) / len(yolo_confidences)
        findings.append(f"Object detection confidence: {avg_yolo_conf:.3f}")
        findings.append(f"Total detections: {len(yolo_confidences)}")
    
    # Add confidence-based recommendations
    if confidence < 0.7:
        findings.append("⚠️ Low confidence - Consider additional imaging or expert review")
    elif confidence > 0.9:
        findings.append("✅ High confidence prediction")
    
    return findings


def generate_breast_findings(prediction, confidence, probability):
    """Generate medical findings for breast cancer"""
    findings = []
    
    if prediction == "Cancer":
        findings = [
            "🔴 Suspicious findings detected in mammography",
            "Immediate radiologist review recommended",
            "Consider additional imaging (ultrasound, MRI)",
            "Biopsy may be required for definitive diagnosis",
            "Discuss family history and genetic testing options",
            "Follow breast cancer screening guidelines"
        ]
        
        if confidence > 0.9:
            findings.append("⚠️ High confidence detection - urgent follow-up advised")
        elif confidence < 0.7:
            findings.append("⚠️ Moderate confidence - additional imaging recommended")
    else:  # Healthy
        findings = [
            "✅ No suspicious findings detected",
            "Continue routine mammography screening",
            "Maintain breast self-examination practices",
            "Follow age-appropriate screening guidelines",
            "Discuss any concerns with healthcare provider"
        ]
        
        if confidence > 0.9:
            findings.append("✅ High confidence normal result")
        elif confidence < 0.7:
            findings.append("⚠️ Consider repeat imaging in 6 months")
    
    # Add probability-based insights
    if 0.4 <= probability <= 0.6:
        findings.append("⚠️ Borderline result - clinical correlation advised")
    
    findings.append(f"Model confidence: {confidence:.1%}")
    findings.append(f"Cancer probability: {probability:.1%}")
    
    return findings
      

def generate_esophagus_findings(prediction, confidence, probability):
    """Generate medical findings for esophagus cancer"""
    findings = []
    
    if prediction == "Cancer":
        findings = [
            "🔴 Suspicious findings detected in esophagus examination",
            "Immediate gastroenterologist consultation recommended",
            "Consider staging studies (CT, EUS, PET scan)",
            "Tissue biopsy required for definitive diagnosis",
            "Evaluate for Barrett's esophagus or dysplasia",
            "Assess swallowing function and nutritional status"
        ]
        
        if confidence > 0.9:
            findings.append("⚠️ High confidence detection - urgent endoscopic evaluation")
        elif confidence < 0.7:
            findings.append("⚠️ Moderate confidence - repeat endoscopy with enhanced imaging")
            
    else:  # Non-Cancer
        findings = [
            "✅ No suspicious findings detected",
            "Continue routine surveillance endoscopy",
            "Monitor for symptoms (dysphagia, odynophagia)",
            "Consider risk factors (GERD, smoking, alcohol)",
            "Follow screening guidelines for high-risk patients"
        ]
        
        if confidence > 0.9:
            findings.append("✅ High confidence normal result")
        elif confidence < 0.7:
            findings.append("⚠️ Consider repeat examination with narrow-band imaging")
    
    # Add probability-based insights
    if 0.4 <= probability <= 0.6:
        findings.append("⚠️ Borderline result - multidisciplinary team review recommended")
    
    # Add technical details
    findings.append(f"Model confidence: {confidence:.1%}")
    findings.append(f"Cancer probability: {probability:.1%}")
    # Add specific esophagus cancer guidance
    if prediction == "Cancer":
        findings.extend([
            "Consider tumor location (upper, middle, lower third)",
            "Evaluate for metastatic disease",
            "Assess candidacy for surgical resection",
            "Discuss palliative care options if advanced"
        ])
    else:
        findings.extend([
            "Continue acid suppression therapy if indicated",
            "Address lifestyle modifications (diet, smoking cessation)",
            "Screen for H. pylori if gastric involvement"
        ])
    
    return findings


def generate_liver_findings(prediction, confidence, cancer_probability, probabilities):
    """Generate medical findings for liver cancer"""
    findings = []
    
    if prediction == "Cancer":
        findings = [
            "🔴 Suspicious hepatic lesion detected",
            "Immediate hepatologist/oncologist consultation required",
            "Consider triphasic CT or MRI with contrast for characterization",
            "Evaluate for hepatocellular carcinoma (HCC) vs metastasis",
            "Check AFP (alpha-fetoprotein) levels",
            "Assess liver function tests and viral hepatitis status"
        ]
        
        # Add confidence-based recommendations
        if confidence > 0.9:
            findings.append("⚠️ High confidence malignant lesion - urgent referral advised")
        elif confidence < 0.7:
            findings.append("⚠️ Moderate confidence - multiphase imaging recommended")
        
        # Add staging recommendations
        findings.extend([
            "Consider staging studies (chest CT, bone scan if indicated)",
            "Evaluate for portal vein thrombosis",
            "Assess candidacy for surgical resection or transplantation",
            "Consider Barcelona Clinic Liver Cancer (BCLC) staging"
        ])
            
    else:  # Non-Cancer
        findings = [
            "✅ No suspicious hepatic lesions detected",
            "Continue routine liver surveillance if at risk",
            "Monitor liver function and tumor markers",
            "Consider risk factors (cirrhosis, viral hepatitis, alcohol)",
            "Follow screening guidelines for high-risk patients"
        ]
        
        if confidence > 0.9:
            findings.append("✅ High confidence benign/normal result")
        elif confidence < 0.7:
            findings.append("⚠️ Consider repeat imaging in 3-6 months")
        
        # Add prevention recommendations
        findings.extend([
            "Maintain hepatitis B/C treatment if applicable",
            "Continue alcohol cessation if history of abuse",
            "Monitor for development of cirrhosis"
        ])
    
    # Add probability-based insights
    if 0.4 <= cancer_probability <= 0.6:
        findings.append("⚠️ Borderline result - consider tissue sampling or short-term follow-up")
    
    # Add technical details
    findings.append(f"Model confidence: {confidence:.1%}")
    findings.append(f"Cancer probability: {cancer_probability:.1%}")
    findings.append(f"Non-cancer probability: {probabilities[0]:.1%}")
    
    # Add liver-specific guidance
    if prediction == "Cancer":
        findings.extend([
            "Evaluate Child-Pugh score for liver function",
            "Consider local ablative therapies if early stage",
            "Discuss systemic therapy options if advanced",
            "Evaluate for liver transplantation candidacy"
        ])
    else:
        findings.extend([
            "Continue surveillance imaging per guidelines",
            "Address underlying liver disease if present",
            "Lifestyle modifications (diet, exercise, alcohol cessation)"
        ])
    
    return findings


def generate_lung_findings(prediction, confidence, probability):
    """Generate medical findings for lung cancer"""
    findings = []
    
    if prediction == "Nodule Detected":
        findings = [
            "🟡 Pulmonary nodule detected in chest imaging",
            "Pulmonologist or thoracic radiologist consultation recommended",
            "Consider high-resolution CT (HRCT) for detailed characterization",
            "Follow Lung-RADS guidelines for nodule management",
            "Evaluate nodule size, morphology, and growth pattern",
            "Consider patient risk factors (smoking history, age, family history)"
        ]
        
        # Add confidence-based recommendations
        if confidence > 0.9:
            findings.append("⚠️ High confidence nodule detection - priority follow-up advised")
        elif confidence < 0.7:
            findings.append("⚠️ Moderate confidence - consider repeat imaging in 3-6 months")
        
        # Add probability-based recommendations
        if probability > 0.8:
            findings.extend([
                "High probability nodule - urgent evaluation recommended",
                "Consider PET-CT for metabolic assessment",
                "Evaluate for biopsy candidacy if appropriate size",
                "Discuss multidisciplinary team evaluation"
            ])
        elif probability > 0.6:
            findings.extend([
                "Moderate probability nodule - close surveillance required",
                "Consider 3-month follow-up imaging",
                "Document nodule characteristics (size, density, location)"
            ])
        else:
            findings.extend([
                "Low probability nodule - routine surveillance appropriate",
                "Consider 6-12 month follow-up based on risk factors"
            ])
    else:  # Healthy
        findings = [
            "✅ No pulmonary nodules detected",
            "Continue routine lung cancer screening if high-risk",
            "Monitor for respiratory symptoms",
            "Consider smoking cessation counseling if applicable",
            "Follow age-appropriate screening guidelines"
        ]
        
        if confidence > 0.9:
            findings.append("✅ High confidence normal result")
        elif confidence < 0.7:
            findings.append("⚠️ Consider repeat imaging if clinically indicated")
        
        # Add prevention recommendations
        findings.extend([
            "Maintain healthy lifestyle (no smoking, regular exercise)",
            "Address occupational/environmental exposures",
            "Continue annual screening if in high-risk category"
        ])

    # Add probability-based insights
    if 0.4 <= probability <= 0.6:
        findings.append("⚠️ Borderline result - clinical correlation and follow-up recommended")
    
    # Add technical details
    findings.append(f"Model confidence: {confidence:.1%}")
    findings.append(f"Nodule probability: {probability:.1%}")
    
    # Add lung-specific guidance
    if prediction == "Nodule Detected":
        findings.extend([
            "Document smoking history (pack-years)",
            "Evaluate family history of lung cancer",
            "Consider radon exposure assessment",
            "Assess fitness for potential interventions"
        ])
    else:
        findings.extend([
            "Continue lung health maintenance",
            "Monitor for persistent cough, shortness of breath",
            "Annual imaging if high-risk (heavy smoker, family history)"
        ])
    
    return findings


def run_pancreatic_model(model_path, pil_image):
    """Run pancreatic cancer inference with ResNet50"""
    try:
        # Initialize pancreatic cancer inference
        pancreatic_inference = PancreaticCancerInference(model_path)
        
        # Get prediction
        prediction_label, confidence, cancer_probability, probabilities = pancreatic_inference.predict(pil_image)
        
        # Get stage information
        stage_info = pancreatic_inference.get_stage_info(prediction_label)
        
        # Generate medical findings
        findings = generate_pancreatic_findings(prediction_label, confidence, cancer_probability, probabilities, stage_info)
        
        # Create result visualization
        result_image = create_pancreatic_result_image(pil_image, prediction_label, confidence, cancer_probability, probabilities, stage_info)
        
        # Upload to Cloudinary
        result_image_url = upload_image_to_cloudinary(result_image, public_id=f"pancreatic_{prediction_label.lower().replace('_', '-')}")
        
        result = {
            "prediction": prediction_label,
            "confidence": confidence,
            "additionalFindings": findings,
            "resultImageUrl": result_image_url,
            "cancerProbability": cancer_probability,
            "stageInfo": stage_info,
            "probabilities": {
                "Non-Cancer": float(probabilities[0]),
                "Cancer_Stage1": float(probabilities[1]),
                "Cancer_Stage2": float(probabilities[2]),
                "Cancer_Stage3": float(probabilities[3]),
                "Cancer_Stage4": float(probabilities[4])
            },
            "modelVersion": "ResNet50-Pancreatic-Cancer-v1.0",
            "analysisType": "CT/MRI Pancreatic Analysis"
        }
        return result
        
    except Exception as e:
        return {
            "prediction": "analysis_error",
            "confidence": 0.0,
            "additionalFindings": [f"Error in pancreatic cancer analysis: {str(e)}"],
            "resultImageUrl": None,
            "modelVersion": "ResNet50-Pancreatic-Cancer-v1.0"
        }

def create_pancreatic_result_image(original_image, prediction, confidence, cancer_probability, probabilities, stage_info):
    """Create visualization for pancreatic cancer results"""
    # Create a result image
    result_img = Image.new("RGB", (700, 500), color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    # Resize original image to fit
    original_resized = original_image.resize((250, 250))
    result_img.paste(original_resized, (10, 60))
    
    # Add text information
    draw.text((280, 60), "PANCREATIC CANCER ANALYSIS", fill=(0, 0, 0), font=None)
    draw.text((280, 100), f"Prediction: {prediction}", fill=(0, 0, 0))
    draw.text((280, 130), f"Confidence: {confidence:.3f}", fill=(0, 0, 0))
    draw.text((280, 160), f"Cancer Probability: {cancer_probability:.3f}", fill=(0, 0, 0))
    
    # Add stage-specific information
    if prediction != "Non-Cancer":
        stage_color = get_stage_color(prediction)
        draw.text((280, 190), f"Stage: {prediction.replace('Cancer_', '')}", fill=stage_color)
        draw.text((280, 220), f"Prognosis: {stage_info.get('prognosis', 'N/A')}", fill=stage_color)
    else:
        draw.text((280, 190), "Status: No Cancer Detected", fill=(0, 150, 0))
    
    # Add probability breakdown
    draw.text((10, 330), "Stage Probabilities:", fill=(0, 0, 0))
    y_pos = 350
    stage_names = ["Non-Cancer", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    for i, (stage, prob) in enumerate(zip(stage_names, probabilities)):
        color = get_stage_color_by_index(i)
        bar_width = int(prob * 150)
        
        # Draw probability bar
        draw.rectangle([10, y_pos, 10 + bar_width, y_pos + 15], fill=color)
        draw.rectangle([10, y_pos, 160, y_pos + 15], outline=(0, 0, 0))
        
        # Draw text
        draw.text((170, y_pos), f"{stage}: {prob:.3f}", fill=(0, 0, 0))
        y_pos += 20
    
    # Add footer
    draw.text((10, 460), "Note: This is AI-assisted analysis. Always consult an oncologist or gastroenterologist.", fill=(100, 100, 100))
    
    return result_img

def get_stage_color(prediction):
    """Get color based on cancer stage"""
    color_map = {
        'Non-Cancer': (0, 150, 0),      # Green
        'Cancer_Stage1': (255, 165, 0), # Orange
        'Cancer_Stage2': (255, 140, 0), # Dark Orange
        'Cancer_Stage3': (255, 69, 0),  # Red Orange
        'Cancer_Stage4': (255, 0, 0)    # Red
    }
    return color_map.get(prediction, (0, 0, 0))

def get_stage_color_by_index(index):
    """Get color based on stage index"""
    colors = [
        (0, 150, 0),      # Non-Cancer - Green
        (255, 165, 0),    # Stage 1 - Orange
        (255, 140, 0),    # Stage 2 - Dark Orange
        (255, 69, 0),     # Stage 3 - Red Orange
        (255, 0, 0)       # Stage 4 - Red
    ]
    return colors[index] if index < len(colors) else (0, 0, 0)

def generate_pancreatic_findings(prediction, confidence, cancer_probability, probabilities, stage_info):
    """Generate medical findings for pancreatic cancer"""
    findings = []
    
    if prediction == "Non-Cancer":
        findings = [
            "✅ No malignant findings detected in pancreatic imaging",
            "Continue routine surveillance if high-risk",
            "Monitor pancreatic function and diabetes status",
            "Address modifiable risk factors (smoking, diet)",
            "Follow screening guidelines for familial pancreatic cancer"
        ]
        
        if confidence > 0.9:
            findings.append("✅ High confidence normal result")
        elif confidence < 0.7:
            findings.append("⚠️ Consider repeat imaging in 6 months")
            
    else:  # Cancer detected
        stage_num = prediction.replace('Cancer_Stage', '')
        findings = [
            f"🔴 Pancreatic cancer detected - Stage {stage_num}",
            "Immediate oncology and gastroenterology consultation required",
            "Multidisciplinary team evaluation essential",
            stage_info.get('description', ''),
            f"Prognosis: {stage_info.get('prognosis', 'Variable')}",
            f"Treatment approach: {stage_info.get('treatment', 'To be determined')}"
        ]
        
        # Add confidence-based recommendations
        if confidence > 0.9:
            findings.append("⚠️ High confidence malignant finding - urgent referral required")
        elif confidence < 0.7:
            findings.append("⚠️ Moderate confidence - consider additional imaging or biopsy")
        
        # Stage-specific recommendations
        if prediction in ["Cancer_Stage1", "Cancer_Stage2"]:
            findings.extend([
                "Assess resectability with surgical oncology",
                "Consider neoadjuvant chemotherapy",
                "Evaluate performance status for surgery"
            ])
        elif prediction in ["Cancer_Stage3", "Cancer_Stage4"]:
            findings.extend([
                "Palliative care consultation recommended",
                "Consider clinical trial eligibility",
                "Manage symptoms and quality of life",
                "Genetic counseling for family members"
            ])
    
    # Add probability-based insights
    if 0.3 <= cancer_probability <= 0.7:
        findings.append("⚠️ Intermediate cancer probability - close monitoring required")
    
    # Add technical details
    findings.append(f"Model confidence: {confidence:.1%}")
    findings.append(f"Overall cancer probability: {cancer_probability:.1%}")
    
    # Add pancreatic-specific guidance
    if prediction != "Non-Cancer":
        findings.extend([
            "Evaluate CA 19-9 tumor marker levels",
            "Assess for jaundice and biliary obstruction",
            "Consider ERCP or PTC for biliary drainage",
            "Nutritional assessment and pancreatic enzyme supplementation"
        ])
    else:
        findings.extend([
            "Monitor for new-onset diabetes in elderly patients",
            "Watch for weight loss or abdominal pain",
            "Consider family history and genetic testing if indicated"
        ])
    
    return findings
