"""
Medical AI API Server - User-Driven Workflow
Single request-response API for medical diagnosis
Memory-efficient with lazy loading and cleanup
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from functools import wraps

from app.graph import MedicalGraph
from app.input_preprocessor import InputPreprocessor, ProcessedInput
from app.state import WorkflowStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
# Keep model warm for low-latency inference after startup.
medical_graph = MedicalGraph(preload_model=True)
input_preprocessor = InputPreprocessor(upload_dir="uploads")

# Track request metrics
request_count = 0
error_count = 0


def handle_errors(f):
    """Decorator for consistent error handling."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        global error_count
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            error_count += 1
            logger.warning(f"Validation error: {e}")
            return jsonify({
                'success': False,
                'error': 'Validation error',
                'message': str(e)
            }), 400
        except Exception as e:
            error_count += 1
            logger.exception("Unexpected error")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'message': str(e)
            }), 500
    return wrapper


def format_response(state, elapsed_time: float) -> Dict[str, Any]:
    """Format the workflow state into API response."""
    structured_assessment = getattr(state, "structured_assessment", None) or {}
    response = {
        'success': state.status == WorkflowStatus.COMPLETED,
        'timestamp': datetime.now().isoformat(),
        'processing_time_seconds': round(elapsed_time, 2),
        'classification': {
            'task_type': state.task_type,
            'confidence': getattr(state, 'classification_confidence', 'medium')
        },
        'result': None,
        'error': state.error if state.error else None
    }
    
    if state.status == WorkflowStatus.COMPLETED:
        response['result'] = {
            'structured_assessment': structured_assessment,
            'clinical_summary': structured_assessment.get('clinical_summary', ''),
            'primary_diagnosis': structured_assessment.get('primary_diagnosis', ''),
            'treatment_plan': structured_assessment.get('treatment_plan', ''),
            'follow_up': structured_assessment.get('follow_up', '')
        }
    
    return response


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'model_loaded': medical_graph._model_loaded
    })


@app.route('/api/diagnose/text', methods=['POST'])
@handle_errors
def diagnose_text():
    """
    Submit text for medical diagnosis.
    
    Request Body:
        {
            "text": "Patient presents with chest pain...",
            "metadata": {  // Optional
                "age": 58,
                "sex": "Male"
            }
        }
    
    Returns:
        Diagnosis and prescription for the classified case
    """
    global request_count
    request_count += 1
    
    start_time = time.time()
    
    # Get request data
    data = request.get_json()
    if not data:
        raise ValueError("No JSON data provided")
    
    text = data.get('text', '').strip()
    if not text:
        raise ValueError("Text field is required")
    
    metadata = data.get('metadata', {})
    
    logger.info(f"[{request_count}] Text diagnosis request received")
    logger.info(f"  Text length: {len(text)} chars")
    
    # Preprocess input
    processed = input_preprocessor.process_text_input(text, metadata)
    input_data = input_preprocessor.build_supervisor_input(processed)
    
    logger.info(f"  Preprocessed input type: {processed.input_type}")
    
    # Run workflow
    state = medical_graph.run(input_data=input_data, cleanup_after=False)
    
    # Format response
    elapsed = time.time() - start_time
    response = format_response(state, elapsed)
    
    logger.info(f"  Completed in {elapsed:.2f}s | Task: {state.task_type} | Status: {state.status}")
    
    status_code = 200 if response['success'] else 422
    return jsonify(response), status_code


@app.route('/api/diagnose/image', methods=['POST'])
@handle_errors
def diagnose_image():
    """
    Submit image for medical diagnosis.
    
    Form Data:
        - image: Image file (required)
        - text: Accompanying text description (optional)
        - metadata: JSON string with additional metadata (optional)
    
    Returns:
        Diagnosis and prescription for the classified case
    """
    global request_count
    request_count += 1
    
    start_time = time.time()
    
    # Check if image is present
    if 'image' not in request.files:
        raise ValueError("No image file provided (field name: 'image')")
    
    image_file = request.files['image']
    if image_file.filename == '':
        raise ValueError("No image file selected")
    
    # Read image data
    image_data = image_file.read()
    if len(image_data) == 0:
        raise ValueError("Empty image file")
    
    # Get optional text
    accompanying_text = request.form.get('text', '').strip() or None
    
    # Get optional metadata
    import json
    metadata_str = request.form.get('metadata', '{}')
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata = {}
    
    logger.info(f"[{request_count}] Image diagnosis request received")
    logger.info(f"  Filename: {image_file.filename}")
    logger.info(f"  Size: {len(image_data) / 1024:.1f} KB")
    logger.info(f"  Has text: {accompanying_text is not None}")
    
    # Preprocess input
    processed = input_preprocessor.process_image_upload(
        file_data=image_data,
        filename=image_file.filename,
        accompanying_text=accompanying_text,
        metadata=metadata
    )
    input_data = input_preprocessor.build_supervisor_input(processed)
    
    logger.info(f"  Inferred type: {processed.metadata.get('inferred_type', 'unknown')}")
    
    # Run workflow
    state = medical_graph.run(input_data=input_data, cleanup_after=False)
    
    # Clean up uploaded file after processing
    if processed.image_path and os.path.exists(processed.image_path):
        try:
            os.remove(processed.image_path)
            logger.info(f"  Cleaned up: {processed.image_path}")
        except Exception as e:
            logger.warning(f"  Failed to clean up {processed.image_path}: {e}")
    
    # Format response
    elapsed = time.time() - start_time
    response = format_response(state, elapsed)
    
    logger.info(f"  Completed in {elapsed:.2f}s | Task: {state.task_type} | Status: {state.status}")
    
    status_code = 200 if response['success'] else 422
    return jsonify(response), status_code


@app.route('/api/diagnose', methods=['POST'])
@handle_errors
def diagnose():
    """
    Universal diagnosis endpoint - accepts both text and image.
    
    Content-Type: multipart/form-data or application/json
    
    For JSON (text only):
        {
            "text": "Patient presents with..."
        }
    
    For multipart (image + optional text):
        - image: Image file
        - text: Accompanying description
    
    Returns:
        Diagnosis and prescription
    """
    global request_count
    request_count += 1
    
    start_time = time.time()
    
    # Determine content type
    content_type = request.content_type or ''
    
    if 'multipart/form-data' in content_type:
        # Handle image upload
        if 'image' not in request.files:
            raise ValueError("No image file provided")
        
        image_file = request.files['image']
        if image_file.filename == '':
            raise ValueError("No image file selected")
        image_data = image_file.read()
        if len(image_data) == 0:
            raise ValueError("Empty image file")
        accompanying_text = request.form.get('text', '').strip() or None
        
        import json
        metadata_str = request.form.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {}
        
        logger.info(f"[{request_count}] Universal endpoint - Image request")
        
        processed = input_preprocessor.process_image_upload(
            file_data=image_data,
            filename=image_file.filename,
            accompanying_text=accompanying_text,
            metadata=metadata
        )
        
        # Clean up after processing
        cleanup_path = processed.image_path
        
    else:
        # Handle text input
        data = request.get_json()
        if not data:
            raise ValueError("No data provided")
        
        text = data.get('text', '').strip()
        if not text:
            raise ValueError("Text field is required")
        
        metadata = data.get('metadata', {})
        
        logger.info(f"[{request_count}] Universal endpoint - Text request")
        
        processed = input_preprocessor.process_text_input(text, metadata)
        cleanup_path = None
    
    # Build input and run
    input_data = input_preprocessor.build_supervisor_input(processed)
    state = medical_graph.run(input_data=input_data, cleanup_after=False)
    
    # Clean up file if image was uploaded
    if cleanup_path and os.path.exists(cleanup_path):
        try:
            os.remove(cleanup_path)
        except Exception:
            pass
    
    # Format response
    elapsed = time.time() - start_time
    response = format_response(state, elapsed)
    
    logger.info(f"  Completed in {elapsed:.2f}s | Task: {state.task_type}")
    
    status_code = 200 if response['success'] else 422
    return jsonify(response), status_code


@app.route('/api/classify', methods=['POST'])
@handle_errors
def classify_only():
    """
    Classify input without running full diagnosis.
    Useful for testing classification logic.
    
    Request Body:
        {
            "text": "Patient has LDL of 150...",
            "metadata": {}
        }
    
    Returns:
        Classification result only
    """
    data = request.get_json()
    if not data:
        raise ValueError("No JSON data provided")
    
    text = data.get('text', '').strip()
    metadata = data.get('metadata', {})
    
    # Preprocess
    processed = input_preprocessor.process_text_input(text, metadata)
    input_data = input_preprocessor.build_supervisor_input(processed)
    
    # Just classify, don't run full workflow
    task_type = medical_graph._classify_with_medgemma_only(input_data)
    
    return jsonify({
        'success': True,
        'classification': {
            'task_type': task_type,
            'input_type': processed.input_type,
            'metadata': processed.metadata
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics."""
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return jsonify({
        'requests': {
            'total': request_count,
            'errors': error_count,
            'success_rate': f"{((request_count - error_count) / max(request_count, 1) * 100):.1f}%"
        },
        'system': {
            'model_loaded': medical_graph._model_loaded,
            'memory_usage_mb': round(memory_info.rss / (1024 * 1024), 2),
            'memory_percent': round(process.memory_percent(), 2)
        }
    })


@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint."""
    return jsonify({
        'name': 'Medical AI API',
        'version': '1.0.0',
        'description': 'User-driven medical diagnosis with MedGemma',
        'endpoints': {
            'POST /api/diagnose/text': 'Submit text for diagnosis',
            'POST /api/diagnose/image': 'Submit image for diagnosis',
            'POST /api/diagnose': 'Universal endpoint (text or image)',
            'POST /api/classify': 'Classify without diagnosis',
            'GET /health': 'Health check',
            'GET /api/stats': 'Usage statistics'
        },
        'features': [
            'Automatic task classification',
            'Memory-efficient with lazy loading',
            'Supports text and image inputs',
            'Single request-response workflow'
        ],
        'supported_tasks': [
            'CT Coronary Angiography',
            'Lipid Profile Analysis',
            'Breast Imaging / Mammogram',
            'Biopsy Report Analysis'
        ]
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Maximum file size is 16MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Medical AI API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080, avoids AirPlay conflict)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🏥 Medical AI API Server Starting")
    logger.info("="*60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug: {args.debug}")
    logger.info("")
    logger.info("Ready to accept requests!")
    logger.info("Test with: curl http://localhost:{}/health".format(args.port))
    logger.info("="*60)
    
    # Run server
    app.run(host=args.host, port=args.port, debug=args.debug)
