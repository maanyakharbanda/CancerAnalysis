# ml-server/file_handler.py
import numpy as np
from PIL import Image
import pydicom
import nibabel as nib
import io

def detect_file_type(file_content):
    """Detect file type from file content bytes"""
    # Check DICOM (starts with specific header)
    if file_content[:4] == b'DICM' or b'DICM' in file_content[:200]:
        return 'dicom'
    
    # Check NIFTI/NIfTI (has specific magic numbers)
    if file_content[:4] in [b'\x5c\x01\x00\x00', b'\x00\x00\x01\x5c']:
        return 'nifti'
    
    # Check for common image headers
    if file_content[:2] == b'\xff\xd8':  # JPEG
        return 'jpeg'
    elif file_content[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
        return 'png'
    elif file_content[:2] in [b'BM']:  # BMP
        return 'bmp'
    elif file_content[:4] in [b'RIFF'] and file_content[8:12] == b'WEBP':  # WebP
        return 'webp'
    
    return 'image'

def preprocess_by_file_type(file_content, file_type):
    """Convert any file type to PIL Image (no model-specific preprocessing)"""
    if file_type == 'dicom':
        return convert_dicom_to_pil(file_content)
    elif file_type == 'nifti':
        return convert_nifti_to_pil(file_content)
    else:  # Standard images
        return convert_image_to_pil(file_content)

def convert_dicom_to_pil(file_content):
    """Convert DICOM to PIL Image"""
    try:
        dicom_file = pydicom.dcmread(io.BytesIO(file_content))
        pixel_array = dicom_file.pixel_array
        
        # Basic normalization only
        if pixel_array.dtype != np.uint8:
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            pixel_array = (pixel_array * 255).astype(np.uint8)
        
        return Image.fromarray(pixel_array, mode='L' if len(pixel_array.shape) == 2 else 'RGB')
    except Exception as e:
        raise ValueError(f"Error processing DICOM: {str(e)}")

def convert_nifti_to_pil(file_content):
    """Convert NIfTI to PIL Image"""
    try:
        nifti_file = nib.load(io.BytesIO(file_content))
        data = nifti_file.get_fdata()
        
        # Take middle slice if 3D/4D
        if len(data.shape) >= 3:
            middle_slice = data.shape[2] // 2
            data = data[:, :, middle_slice]
            if len(data.shape) == 3:  # 4D case
                data = data[:, :, 0]
        
        # Basic normalization
        data = (data - data.min()) / (data.max() - data.min())
        data = (data * 255).astype(np.uint8)
        
        return Image.fromarray(data, mode='L')
    except Exception as e:
        raise ValueError(f"Error processing NIfTI: {str(e)}")

def convert_image_to_pil(file_content):
    """Convert standard image to PIL Image"""
    try:
        pil_image = Image.open(io.BytesIO(file_content))
        if pil_image.mode not in ['RGB', 'L']:
            pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")