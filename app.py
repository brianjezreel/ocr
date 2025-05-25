from flask import Flask, request, jsonify, render_template
from PIL import Image
import pytesseract
import os
from werkzeug.utils import secure_filename
from collections import defaultdict
import cv2
import numpy as np
import io
import base64
import subprocess
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def verify_tesseract_setup():
    """Verify Tesseract installation and configuration"""
    logger.info("Verifying Tesseract setup...")
    
    # Check Tesseract executable
    tesseract_path = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
    if not os.path.exists(tesseract_path):
        raise RuntimeError(f"Tesseract executable not found at: {tesseract_path}")
    
    # Check TESSDATA_PREFIX and possible alternative locations
    tessdata_paths = [
        os.getenv('TESSDATA_PREFIX', '/usr/share/tesseract-ocr/tessdata'),
        '/usr/share/tesseract-ocr/tessdata',
        '/usr/local/share/tessdata',
        '/usr/share/tessdata',
        '/usr/share/tesseract-ocr/4.00/tessdata'
    ]
    
    eng_traineddata = None
    for tessdata_dir in tessdata_paths:
        if not tessdata_dir:
            continue
        potential_path = os.path.join(tessdata_dir, 'eng.traineddata')
        if os.path.exists(potential_path):
            eng_traineddata = potential_path
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            logger.info(f"Found eng.traineddata at: {eng_traineddata}")
            logger.info(f"Set TESSDATA_PREFIX to: {tessdata_dir}")
            break
    
    if not eng_traineddata:
        # Try to find eng.traineddata anywhere in the system
        try:
            find_output = subprocess.check_output(['find', '/', '-name', 'eng.traineddata'], stderr=subprocess.STDOUT)
            found_paths = find_output.decode().strip().split('\n')
            if found_paths:
                eng_traineddata = found_paths[0]
                tessdata_dir = os.path.dirname(eng_traineddata)
                os.environ['TESSDATA_PREFIX'] = tessdata_dir
                logger.info(f"Found eng.traineddata at: {eng_traineddata}")
                logger.info(f"Set TESSDATA_PREFIX to: {tessdata_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error searching for eng.traineddata: {e.output.decode()}")
    
    if not eng_traineddata:
        raise RuntimeError("English language training data not found in any standard location")
    
    # Test Tesseract
    try:
        version = subprocess.check_output([tesseract_path, '--version'], stderr=subprocess.STDOUT)
        logger.info(f"Tesseract version: {version.decode()}")
        
        # List available languages
        langs = subprocess.check_output([tesseract_path, '--list-langs'], stderr=subprocess.STDOUT)
        logger.info(f"Available languages: {langs.decode()}")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to verify Tesseract: {e.output.decode()}")

# Try to set up Tesseract
try:
    verify_tesseract_setup()
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
except Exception as e:
    logger.error(f"Failed to set up Tesseract: {str(e)}")
    raise

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Log Tesseract configuration
def check_tesseract():
    logger.info("Checking Tesseract configuration...")
    
    # Log the explicitly set path
    logger.info(f"Tesseract command path set to: {pytesseract.pytesseract.tesseract_cmd}")
    
    # Check if the file exists
    if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        logger.info("✓ Tesseract executable found at the specified path")
    else:
        logger.error("✗ Tesseract executable NOT found at the specified path")
        
        # Try to find tesseract in common locations
        common_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/local/bin/tesseract'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"Found Tesseract at alternative path: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    # Try to run tesseract version command
    try:
        version_output = subprocess.check_output([pytesseract.pytesseract.tesseract_cmd, '--version'], stderr=subprocess.STDOUT)
        logger.info(f"Tesseract version output:\n{version_output.decode()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get Tesseract version: {e.output.decode() if e.output else str(e)}")
    except Exception as e:
        logger.error(f"Error checking Tesseract version: {str(e)}")

    # Check tessdata directory
    tessdata_paths = [
        '/usr/share/tesseract-ocr/4.00/tessdata',
        '/usr/share/tesseract-ocr/tessdata',
        '/usr/local/share/tessdata',
        os.getenv('TESSDATA_PREFIX', '')
    ]

    tessdata_found = False
    for tessdata_dir in tessdata_paths:
        if os.path.exists(tessdata_dir):
            logger.info(f"✓ Found tessdata directory at: {tessdata_dir}")
            try:
                files = os.listdir(tessdata_dir)
                logger.info(f"Tessdata contents: {files}")
                tessdata_found = True
                break
            except Exception as e:
                logger.error(f"Error listing tessdata directory: {str(e)}")

    if not tessdata_found:
        logger.error("✗ Tessdata directory not found in any of the expected locations")
        logger.info("Searching for tessdata in system...")
        try:
            # Try to find tessdata using find command
            find_output = subprocess.check_output(['find', '/', '-name', 'tessdata', '-type', 'd'], stderr=subprocess.STDOUT)
            found_paths = find_output.decode().strip().split('\n')
            logger.info(f"Found tessdata directories: {found_paths}")
        except Exception as e:
            logger.error(f"Error searching for tessdata: {str(e)}")

# Run the check on startup
check_tesseract()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_paragraphs(data, blocks):
    """
    Analyze text blocks to detect paragraph boundaries using multiple heuristics
    """
    # Calculate average metrics
    valid_indices = [i for i in range(len(data['text'])) if data['text'][i].strip()]
    if not valid_indices:
        return []

    heights = [data['height'][i] for i in valid_indices]
    avg_height = sum(heights) / len(heights)
    
    # Analyze line patterns
    line_patterns = []
    for block_num, block_lines in blocks.items():
        sorted_lines = sorted(block_lines.items(), key=lambda x: x[1][0]['top'])
        for i, (line_num, words) in enumerate(sorted_lines):
            sorted_words = sorted(words, key=lambda x: x['left'])
            
            # Calculate line metrics
            line_left = sorted_words[0]['left']
            line_length = sorted_words[-1]['left'] + sorted_words[-1]['width'] - line_left
            line_text = ' '.join(w['text'].strip() for w in sorted_words)
            
            line_info = {
                'text': line_text,
                'left': line_left,
                'length': line_length,
                'top': sorted_words[0]['top'],
                'bottom': max(w['top'] + w['height'] for w in sorted_words),
                'block_num': block_num,
                'line_num': line_num,
                'words': len(sorted_words),
                'is_sentence_end': line_text.strip().endswith(('.', '!', '?')),
                'indent': line_left
            }
            line_patterns.append(line_info)
    
    # Sort all lines by vertical position
    line_patterns.sort(key=lambda x: x['top'])
    
    # Detect paragraph boundaries
    paragraphs = []
    current_paragraph = []
    
    for i, line in enumerate(line_patterns):
        is_paragraph_break = False
        
        if i > 0:
            prev_line = line_patterns[i-1]
            
            # Check various conditions for paragraph breaks
            vertical_gap = line['top'] - prev_line['bottom']
            indent_change = line['indent'] - prev_line['indent']
            
            # Conditions for paragraph break:
            is_paragraph_break = any([
                # Significant vertical gap
                vertical_gap > avg_height * 1.5,
                
                # New block
                line['block_num'] != prev_line['block_num'],
                
                # Indentation change with sentence end
                prev_line['is_sentence_end'] and abs(indent_change) > avg_height,
                
                # Short previous line with sentence end (possible paragraph end)
                prev_line['is_sentence_end'] and prev_line['length'] < prev_line['indent'] + avg_height * 10,
                
                # Significant indentation change
                abs(indent_change) > avg_height * 2
            ])
        
        if is_paragraph_break and current_paragraph:
            paragraphs.append(current_paragraph)
            current_paragraph = []
        
        current_paragraph.append(line)
    
    # Add the last paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def organize_text_blocks(data):
    """
    This function is now simplified since we're using Tesseract's native line detection
    """
    # The text already contains proper line breaks from Tesseract
    return data['text']

def preprocess_image(image):
    """
    Apply various preprocessing techniques to improve OCR accuracy
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Make a copy of the original image
    original = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

    # Dilation and erosion to improve text connectivity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Store all preprocessed versions
    preprocessed_versions = {
        'original': cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        'grayscale': cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        'threshold': cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB),
        'denoised': cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB),
        'enhanced': cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB),
        'final': cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
    }

    # Convert preprocessed images to base64
    preprocessed_images = {}
    for name, img in preprocessed_versions.items():
        pil_img = Image.fromarray(img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        preprocessed_images[name] = f'data:image/png;base64,{img_str}'

    # Configure Tesseract parameters for better line detection
    custom_config = r'--psm 6 --oem 3'  # PSM 6: Assume a uniform block of text

    # Attempt OCR on both processed versions and original
    results = []
    images = [
        ('original', original),
        ('enhanced', enhanced),
        ('processed', eroded)
    ]

    best_result = None
    max_confidence = 0

    for name, img in images:
        # Convert back to PIL Image for Tesseract
        if name == 'original' and isinstance(image, Image.Image):
            pil_img = image
        else:
            if len(img.shape) == 2:  # If grayscale
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Get OCR result with confidence scores
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Get raw text with line breaks preserved
        raw_text = pytesseract.image_to_string(pil_img, config=custom_config)
        
        confidences = [int(conf) for conf, text in zip(data['conf'], data['text']) if str(text).strip()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        result = {
            'text': raw_text,  # Use raw text with preserved line breaks
            'confidence': avg_confidence,
            'boxes': []
        }

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                result['boxes'].append({
                    'x': float(data['left'][i]) / pil_img.width,
                    'y': float(data['top'][i]) / pil_img.height,
                    'width': float(data['width'][i]) / pil_img.width,
                    'height': float(data['height'][i]) / pil_img.height,
                    'text': data['text'][i],
                    'conf': float(data['conf'][i]),
                    'block_num': data['block_num'][i],
                    'line_num': data['line_num'][i]
                })

        results.append((result, avg_confidence))
        
        if avg_confidence > max_confidence:
            max_confidence = avg_confidence
            best_result = result

    # Add preprocessed images to the result
    best_result['preprocessed_images'] = preprocessed_images
    
    # Print raw output with visible markers
    print('\n' + '='*50)
    print('RAW OCR OUTPUT (↵ = line break, · = space):')
    print('='*50)
    debug_text = best_result['text'].replace(' ', '·').replace('\n', '↵\n')
    print(debug_text)
    print('='*50 + '\n')
    
    return best_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file temporarily
            file.save(filepath)
            
            # Open and preprocess the image
            image = Image.open(filepath)
            result = preprocess_image(image)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 