"""
Input validation and preprocessing module for Medical AI System
Handles user uploads and prepares data for the Supervisor node
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class ProcessedInput:
    """Structured input after preprocessing."""
    input_type: str  # 'image', 'text', 'multimodal'
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    original_filename: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Type alias for metadata dict
MetadataDict = Dict[str, Any]


class InputPreprocessor:
    """
    Preprocesses user input (images and text) for medical classification.
    Extracts metadata and prepares structured data for the Supervisor.
    """
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.dcm'}
    
    # Maximum file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"InputPreprocessor initialized (upload_dir: {upload_dir})")
    
    def process_text_input(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessedInput:
        """
        Process text input from user.
        
        Args:
            text: Raw text input
            metadata: Additional metadata (age, sex, etc.)
        
        Returns:
            ProcessedInput with extracted information
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Extract structured data from text
        extracted_data = self._extract_structured_data(cleaned_text)
        
        return ProcessedInput(
            input_type='text',
            text_content=cleaned_text,
            metadata={
                **(metadata or {}),
                **extracted_data,
                'word_count': len(cleaned_text.split()),
                'has_lab_values': self._has_lab_values(cleaned_text),
                'has_imaging_terms': self._has_imaging_terms(cleaned_text),
                'has_pathology_terms': self._has_pathology_terms(cleaned_text)
            }
        )
    
    def process_image_upload(self, file_data: bytes, filename: str, 
                            accompanying_text: Optional[str] = None) -> ProcessedInput:
        """
        Process uploaded image file.
        
        Args:
            file_data: Binary image data
            filename: Original filename
            accompanying_text: Optional text description
        
        Returns:
            ProcessedInput with saved image path
        """
        # Validate file
        self._validate_image_file(file_data, filename)
        
        # Save file securely
        safe_filename = self._sanitize_filename(filename)
        import time
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{safe_filename}"
        file_path = self.upload_dir / unique_filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(file_data)
            logger.info(f"Image saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise ValueError(f"Failed to save uploaded image: {e}")
        
        # Determine image type from filename
        image_type = self._infer_image_type(filename)
        
        return ProcessedInput(
            input_type='multimodal' if accompanying_text else 'image',
            image_path=str(file_path),
            text_content=self._clean_text(accompanying_text) if accompanying_text else None,
            original_filename=filename,
            metadata={
                'file_size': len(file_data),
                'inferred_type': image_type,
                'has_text': accompanying_text is not None
            }
        )
    
    def build_supervisor_input(self, processed: ProcessedInput) -> Dict[str, Any]:
        """
        Convert processed input into format expected by Supervisor node.
        
        Args:
            processed: ProcessedInput from text or image processing
        
        Returns:
            Dictionary ready for MedicalGraph.run()
        """
        input_data = {
            'input_type': processed.input_type,
            'original_filename': processed.original_filename,
            **(processed.metadata or {})
        }
        
        # Add text content if present
        if processed.text_content:
            input_data['text_content'] = processed.text_content
            
            # Try to extract specific medical fields
            extracted = self._extract_medical_fields(processed.text_content)
            input_data.update(extracted)
        
        # Add image path if present
        if processed.image_path:
            input_data['image_path'] = processed.image_path
        
        return input_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text input."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Normalize medical abbreviations
        replacements = {
            ' c/o ': ' complains of ',
            ' w/ ': ' with ',
            ' w/o ': ' without ',
            ' s/p ': ' status post ',
            ' r/o ': ' rule out ',
        }
        
        for abbrev, full in replacements.items():
            cleaned = cleaned.replace(abbrev, full)
        
        return cleaned.strip()
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured medical data from text."""
        data = {}
        text_lower = text.lower()
        
        # Extract lab values using patterns
        import re
        
        # LDL patterns
        ldl_match = re.search(r'ldl[\s:]*(\d+)', text_lower)
        if ldl_match:
            data['ldl'] = int(ldl_match.group(1))
        
        # HDL patterns
        hdl_match = re.search(r'hdl[\s:]*(\d+)', text_lower)
        if hdl_match:
            data['hdl'] = int(hdl_match.group(1))
        
        # Total cholesterol
        tc_match = re.search(r'total cholesterol[\s:]*(\d+)', text_lower)
        if tc_match:
            data['total_cholesterol'] = int(tc_match.group(1))
        
        # Triglycerides
        tg_match = re.search(r'triglycerides[\s:]*(\d+)', text_lower)
        if tg_match:
            data['triglycerides'] = int(tg_match.group(1))
        
        # Stenosis percentage
        stenosis_match = re.search(r'(\d+)%?\s*stenosis', text_lower)
        if stenosis_match:
            data['stenosis_percent'] = int(stenosis_match.group(1))
        
        # Vessel names
        vessels = ['lad', 'lcx', 'rca', 'lm', 'left main']
        for vessel in vessels:
            if vessel in text_lower:
                data['vessel'] = vessel.upper() if vessel != 'left main' else 'LM'
                break
        
        # BI-RADS
        birads_match = re.search(r'bi-rads\s*(\d+[a-d]?)', text_lower)
        if birads_match:
            data['birads_category'] = f"BI-RADS {birads_match.group(1).upper()}"
        
        # Age
        age_match = re.search(r'(\d+)[\s-]*year[\s-]*old', text_lower)
        if age_match:
            data['age'] = int(age_match.group(1))
        
        # Sex
        if 'female' in text_lower or ' woman' in text_lower:
            data['sex'] = 'Female'
        elif 'male' in text_lower or ' man' in text_lower:
            data['sex'] = 'Male'
        
        return data
    
    def _extract_medical_fields(self, text: str) -> Dict[str, Any]:
        """Extract all possible medical fields from text."""
        fields = {}
        text_lower = text.lower()
        
        # Try to determine the main content type
        if any(term in text_lower for term in ['pathology', 'biopsy', 'histology', 'specimen']):
            fields['report_text'] = text
        elif any(term in text_lower for term in ['mammogram', 'ultrasound', 'birads', 'breast']):
            fields['finding'] = text
        elif any(term in text_lower for term in ['stenosis', 'coronary', 'lad', 'lcx', 'rca']):
            fields['finding'] = text
        else:
            # Default: store as general finding or report
            fields['finding'] = text
        
        # Merge with structured data extraction
        fields.update(self._extract_structured_data(text))
        
        return fields
    
    def _has_lab_values(self, text: str) -> bool:
        """Check if text contains lab values."""
        text_lower = text.lower()
        lab_terms = ['ldl', 'hdl', 'cholesterol', 'triglycerides', 'mg/dl', 'mmol/l']
        return any(term in text_lower for term in lab_terms)
    
    def _has_imaging_terms(self, text: str) -> bool:
        """Check if text contains imaging terminology."""
        text_lower = text.lower()
        imaging_terms = ['mammogram', 'ultrasound', 'ct scan', 'angiography', 'birads', 
                        'stenosis', 'mass', 'lesion', 'calcification']
        return any(term in text_lower for term in imaging_terms)
    
    def _has_pathology_terms(self, text: str) -> bool:
        """Check if text contains pathology terminology."""
        text_lower = text.lower()
        path_terms = ['biopsy', 'pathology', 'histology', 'carcinoma', 'adenocarcinoma',
                     'grade', 'stage', 'specimen', 'tissue']
        return any(term in text_lower for term in path_terms)
    
    def _validate_image_file(self, file_data: bytes, filename: str):
        """Validate uploaded image file."""
        # Check file size
        if len(file_data) > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {self.MAX_FILE_SIZE / (1024*1024):.1f}MB")
        
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.SUPPORTED_IMAGE_FORMATS}")
        
        # Basic magic number check for common formats
        if len(file_data) < 4:
            raise ValueError("File too small to be valid image")
        
        # JPEG
        if ext in ['.jpg', '.jpeg'] and not file_data[:2] == b'\xff\xd8':
            raise ValueError("Invalid JPEG file")
        
        # PNG
        if ext == '.png' and not file_data[:4] == b'\x89PNG':
            raise ValueError("Invalid PNG file")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security."""
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove non-alphanumeric characters except safe ones
        safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
        sanitized = ''.join(c for c in filename if c in safe_chars)
        
        # Ensure not empty
        if not sanitized or sanitized == '.':
            sanitized = 'upload'
        
        return sanitized
    
    def _infer_image_type(self, filename: str) -> str:
        """Infer medical image type from filename."""
        fname_lower = filename.lower()
        
        if any(term in fname_lower for term in ['ct', 'coronary', 'cardiac', 'ccta']):
            return 'ct_coronary'
        elif any(term in fname_lower for term in ['mammogram', 'mammography', 'breast']):
            return 'breast_imaging'
        elif any(term in fname_lower for term in ['ultrasound', 'us']):
            return 'ultrasound'
        elif any(term in fname_lower for term in ['xray', 'x-ray', 'radiograph']):
            return 'xray'
        else:
            return 'unknown'
    
    def cleanup_uploads(self, max_age_hours: int = 24):
        """Clean up old uploaded files."""
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned = 0
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old upload files")


# Convenience function for quick preprocessing
def preprocess_user_input(text: Optional[str] = None, 
                         image_data: Optional[bytes] = None,
                         image_filename: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick preprocessing function for user input.
    
    Args:
        text: Text input from user
        image_data: Binary image data
        image_filename: Original image filename
        metadata: Additional metadata
    
    Returns:
        Dictionary ready for MedicalGraph.run()
    """
    preprocessor = InputPreprocessor()
    
    if image_data and image_filename:
        processed = preprocessor.process_image_upload(
            file_data=image_data,
            filename=image_filename,
            accompanying_text=text
        )
    elif text:
        processed = preprocessor.process_text_input(text, metadata)
    else:
        raise ValueError("Either text or image data must be provided")
    
    return preprocessor.build_supervisor_input(processed)