# chatbot.py
import torch
import json
import pickle
import logging
import numpy as np
import re
import warnings
from typing import Optional, Dict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gemini_config import generate_medical_response
from model import Classifier as CancerClassifier, Config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatBot:
    def __init__(self, 
                 kb_path: str = "cancer_knowledge_base.json",
                 model_path: str = "best_medical_model.pth",
                 tag_mappings_path: str = "idx2tag.pickle"):
        self.kb_path = kb_path
        self.model_path = model_path
        self.tag_mappings_path = tag_mappings_path
        self.min_confidence = 0.6
        self.fallback_mode = False

        self.load_knowledge_base()
        self.load_intent_model()
        self.load_semantic_model()

    def load_knowledge_base(self):
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                kb_json = json.load(f)
            self.kb_entries = kb_json['intents']
            self.pattern_to_entry = {}
            self.all_patterns = []

            for entry in self.kb_entries:
                for pattern in entry.get("patterns", []):
                    self.all_patterns.append(pattern)
                    self.pattern_to_entry[pattern] = entry

            logger.info(f"Knowledge base loaded with {len(self.kb_entries)} intents")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self.kb_entries = []

    def load_intent_model(self):
        try:
            with open(self.tag_mappings_path, 'rb') as f:
                self.idx_to_tag = pickle.load(f)

            config = Config()
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.intent_model = CancerClassifier(len(self.idx_to_tag)).to(self.device)
            self.intent_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.intent_model.eval()

            logger.info("Intent model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            self.fallback_mode = True
            self.intent_model = None
            self.tokenizer = None

    def load_semantic_model(self):
        try:
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.pattern_embeddings = self.semantic_model.encode(self.all_patterns, show_progress_bar=True)
            logger.info("Semantic model and embeddings ready")
        except Exception as e:
            logger.error(f"Semantic model error: {e}")
            self.semantic_model = None

    def is_medical_question(self, text: str) -> bool:
       # ...existing code...
        medical_keywords = [
    # Cancer & Oncology
    'cancer', 'tumor', 'malignant', 'benign', 'metastasis', 'oncology', 'oncologist', 'carcinoma', 'sarcoma',
    'lymphoma', 'leukemia', 'melanoma', 'biopsy', 'chemotherapy', 'chemo', 'radiation', 'radiotherapy',
    'immunotherapy', 'targeted therapy', 'remission', 'recurrence', 'screening', 'staging', 'prognosis',
    'palliative', 'neoplasm', 'cytology', 'histology', 'in situ', 'adenocarcinoma', 'squamous cell', 'basal cell',
    'metastatic', 'primary tumor', 'secondary tumor', 'oncogene', 'tumor marker', 'PET scan', 'CT scan', 'MRI',
    'ultrasound', 'mammogram', 'colonoscopy', 'pap smear', 'PSA', 'BRCA', 'HER2', 'biomarker', 'sentinel node',
    'lymph node', 'surgical oncology', 'radiologist', 'pathologist', 'clinical trial', 'protocol', 'invasive',
    'noninvasive', 'adjuvant', 'neoadjuvant', 'maintenance therapy', 'precision medicine', 'genetic testing',

    # Symptoms & Signs
    'symptom', 'pain', 'fatigue', 'fever', 'nausea', 'vomiting', 'cough', 'bleeding', 'swelling', 'weight loss',
    'appetite loss', 'night sweats', 'itching', 'rash', 'headache', 'dizziness', 'shortness of breath', 'dyspnea',
    'chills', 'sore throat', 'diarrhea', 'constipation', 'jaundice', 'anemia', 'bruising', 'lump', 'mass',
    'palpitation', 'edema', 'inflammation', 'infection', 'ulcer', 'lesion', 'discharge', 'cramping', 'stiffness',
    'weakness', 'numbness', 'tingling', 'confusion', 'memory loss', 'seizure', 'blurred vision', 'double vision',

    # Diagnosis & Tests
    'diagnosis', 'diagnose', 'test', 'screening', 'blood test', 'CBC', 'urinalysis', 'x-ray', 'MRI', 'CT scan',
    'PET scan', 'biopsy', 'endoscopy', 'colonoscopy', 'mammogram', 'ultrasound', 'genetic test', 'histopathology',
    'cytology', 'tumor marker', 'PSA', 'BRCA', 'EKG', 'ECG', 'EEG', 'spirometry', 'bone scan', 'liver function',
    'kidney function', 'cholesterol', 'glucose', 'HbA1c', 'stool test', 'pap smear', 'culture', 'antibody test',

    # Treatments & Procedures
    'treatment', 'therapy', 'medication', 'drug', 'prescription', 'surgery', 'operation', 'procedure', 'transplant',
    'chemotherapy', 'radiation', 'immunotherapy', 'targeted therapy', 'hormone therapy', 'stem cell', 'bone marrow',
    'dialysis', 'infusion', 'injection', 'vaccine', 'vaccination', 'rehabilitation', 'physical therapy',
    'occupational therapy', 'speech therapy', 'palliative care', 'hospice', 'pain management', 'analgesic',
    'antibiotic', 'antiviral', 'antifungal', 'antidepressant', 'anticoagulant', 'antihypertensive', 'insulin',
    'statin', 'chemoprevention', 'prophylaxis', 'supplement', 'vitamin', 'nutritional support', 'IV', 'catheter',
    'stent', 'pacemaker', 'defibrillator', 'intubation', 'ventilator', 'oxygen therapy', 'laser therapy',

    # Medical Specialties & Professionals
    'doctor', 'physician', 'nurse', 'specialist', 'surgeon', 'oncologist', 'hematologist', 'radiologist',
    'pathologist', 'cardiologist', 'neurologist', 'endocrinologist', 'gastroenterologist', 'dermatologist',
    'urologist', 'gynecologist', 'pediatrician', 'psychiatrist', 'psychologist', 'therapist', 'pharmacist',
    'nutritionist', 'dietitian', 'anesthesiologist', 'immunologist', 'rheumatologist', 'pulmonologist',
    'nephrologist', 'orthopedist', 'ophthalmologist', 'otolaryngologist', 'ENT', 'primary care', 'family doctor',

    # Diseases & Conditions
    'disease', 'disorder', 'illness', 'infection', 'virus', 'bacteria', 'fungus', 'parasite', 'autoimmune',
    'diabetes', 'hypertension', 'high blood pressure', 'stroke', 'heart attack', 'myocardial infarction',
    'arrhythmia', 'asthma', 'COPD', 'bronchitis', 'pneumonia', 'tuberculosis', 'hepatitis', 'cirrhosis',
    'kidney failure', 'renal failure', 'liver failure', 'anemia', 'leukemia', 'lymphoma', 'HIV', 'AIDS',
    'arthritis', 'osteoporosis', 'epilepsy', 'seizure', 'migraine', 'depression', 'anxiety', 'bipolar',
    'schizophrenia', 'autism', 'dementia', 'Alzheimer', 'Parkinson', 'multiple sclerosis', 'lupus', 'psoriasis',
    'eczema', 'dermatitis', 'obesity', 'overweight', 'malnutrition', 'dehydration', 'allergy', 'anaphylaxis',

    # Anatomy & Physiology
    'organ', 'tissue', 'cell', 'gene', 'chromosome', 'DNA', 'RNA', 'protein', 'enzyme', 'hormone', 'immune',
    'antibody', 'antigen', 'blood', 'plasma', 'serum', 'bone', 'muscle', 'nerve', 'artery', 'vein', 'capillary',
    'heart', 'lung', 'liver', 'kidney', 'pancreas', 'spleen', 'stomach', 'intestine', 'colon', 'rectum', 'bladder',
    'prostate', 'uterus', 'ovary', 'testis', 'brain', 'spinal cord', 'skin', 'hair', 'nail', 'eye', 'ear', 'nose',
    'throat', 'mouth', 'tongue', 'tooth', 'gland', 'lymph', 'lymph node', 'bone marrow',

    # General Health & Wellness
    'health', 'wellness', 'prevention', 'screening', 'checkup', 'vaccination', 'immunization', 'exercise',
    'fitness', 'diet', 'nutrition', 'weight', 'BMI', 'hydration', 'sleep', 'stress', 'mental health', 'anxiety',
    'depression', 'smoking', 'alcohol', 'substance abuse', 'addiction', 'recovery', 'support group', 'counseling',
    'therapy', 'self-care', 'hygiene', 'sanitation', 'first aid', 'emergency', 'triage', 'ambulance', 'paramedic',

    # Side Effects & Complications
    'side effect', 'adverse effect', 'complication', 'allergy', 'reaction', 'toxicity', 'overdose', 'withdrawal',
    'dependency', 'tolerance', 'resistance', 'mutation', 'relapse', 'secondary infection', 'superinfection',
    'immunosuppression', 'neutropenia', 'thrombocytopenia', 'anemia', 'alopecia', 'mucositis', 'neuropathy',
    'cardiotoxicity', 'hepatotoxicity', 'nephrotoxicity', 'ototoxicity', 'myelosuppression', 'fatigue',

    # Medical Equipment & Devices
    'monitor', 'infusion pump', 'IV', 'catheter', 'stent', 'pacemaker', 'defibrillator', 'ventilator', 'oxygen',
    'wheelchair', 'walker', 'crutch', 'prosthesis', 'implant', 'hearing aid', 'glasses', 'contact lens', 'brace',
    'splint', 'cast', 'sling', 'bandage', 'dressing', 'suture', 'scalpel', 'forceps', 'syringe', 'needle',

    # Miscellaneous
    'clinical trial', 'protocol', 'placebo', 'randomized', 'double-blind', 'informed consent', 'ethics',
    'insurance', 'coverage', 'copay', 'deductible', 'referral', 'consultation', 'follow-up', 'discharge',
    'admission', 'outpatient', 'inpatient', 'ward', 'ICU', 'ER', 'emergency room', 'hospital', 'clinic', 'pharmacy',
    'prescription', 'over-the-counter', 'generic', 'brand name', 'formulary', 'prior authorization', 'medicare',
    'medicaid', 'HMO', 'PPO', 'primary care', 'specialist', 'telemedicine', 'virtual visit', 'appointment'
]
        return any(word in text.lower() for word in medical_keywords)

    def get_kb_response(self, query: str) -> Optional[Dict]:
        for pattern, entry in self.pattern_to_entry.items():
            if pattern.lower() in query.lower():
                return {"answer": entry['answer'], "source": "knowledge_base", "confidence": 1.0}

        if hasattr(self, 'semantic_model'):
            query_emb = self.semantic_model.encode([query])
            sims = cosine_similarity(query_emb, self.pattern_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            print(f"Semantic similarity score: {best_score:.2f} for '{query}'")
            if best_score > 0.70:
                pattern = self.all_patterns[best_idx]
                entry = self.pattern_to_entry[pattern]
                return {"answer": entry['answer'], "source": "semantic_search", "confidence": best_score}

        return None

    def get_gemini_response(self, query: str) -> str:
        return generate_medical_response(query)

    def get_response(self, query: str, show_debug: bool = False) -> str:
        if not self.is_medical_question(query):
            return ("I'm specialized in cancer and medical topics. Please ask something related to cancer, treatment, symptoms, or diagnosis.")

        kb_result = self.get_kb_response(query)
        if kb_result:
            response = kb_result['answer']
            if show_debug:
                response += f"\n\n[Source: {kb_result['source']} | Confidence: {kb_result['confidence']:.2f}]"
            return response

        gemini_response = self.get_gemini_response(query)
        if show_debug:
            gemini_response += "\n\n[Source: Gemini AI]"
        return gemini_response
