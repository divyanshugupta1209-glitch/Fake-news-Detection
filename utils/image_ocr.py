# utils/image_ocr.py

import pytesseract
from PIL import Image
import re
import os

# Set Tesseract path for Windows
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print("✅ OCR module loaded successfully")
else:
    print("⚠️ Tesseract not found!")
    print(f"   Expected at: {TESSERACT_PATH}")
    print("   Please install from: https://github.com/UB-Mannheim/tesseract/wiki")


def extract_text_from_image(image):
    """
    Extract text from image using Tesseract OCR
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Extracted text
    """
    try:
        # Convert to grayscale for better OCR accuracy
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        # Extract text with English language
        text = pytesseract.image_to_string(gray_image, lang='eng')
        
        # Clean up text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
        
        return text
    
    except Exception as e:
        print(f"⚠️ OCR extraction error: {e}")
        return ""


def analyze_image_text(image, claim_text):
    """
    Extract text from image and check if it matches the claim
    
    Args:
        image: PIL Image object
        claim_text: The news claim text
        
    Returns:
        dict with OCR analysis results
    """
    print("   📄 Running OCR on image...")
    
    # Extract text
    extracted_text = extract_text_from_image(image)
    
    # Check if any text was found
    if not extracted_text or len(extracted_text.strip()) < 10:
        print("   ℹ️ No significant text detected")
        return {
            'has_text': False,
            'extracted_text': '',
            'score': 0.7,  # Neutral score when no text
            'match_ratio': 0,
            'info': 'No text detected in image (this is normal for photos)',
            'warning': None
        }
    
    print(f"   ✅ Extracted {len(extracted_text)} characters")
    
    # Compare extracted text with claim
    claim_lower = claim_text.lower()
    extracted_lower = extracted_text.lower()
    
    # Extract significant words (4+ letters to avoid noise)
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_lower))
    extracted_words = set(re.findall(r'\b\w{4,}\b', extracted_lower))
    
    # Remove common stopwords
    stopwords = {
        'this', 'that', 'with', 'from', 'have', 'been', 'will',
        'their', 'there', 'about', 'which', 'when', 'them', 'some',
        'would', 'make', 'like', 'into', 'time', 'than', 'look',
        'only', 'come', 'over', 'think', 'also', 'back', 'after',
        'work', 'first', 'well', 'even', 'want', 'because', 'these',
        'give', 'most', 'should', 'where', 'much', 'before', 'through',
        'just', 'very', 'then', 'other', 'being', 'such', 'could'
    }
    claim_words -= stopwords
    extracted_words -= stopwords
    
    # Handle empty claim
    if not claim_words or len(claim_words) < 2:
        return {
            'has_text': True,
            'extracted_text': extracted_text[:500],
            'score': 0.7,
            'match_ratio': 0,
            'info': 'Image contains text but claim is too short to compare',
            'warning': None
        }
    
    # Calculate match ratio
    overlap = len(claim_words & extracted_words)
    match_ratio = overlap / len(claim_words)
    
    print(f"   🔍 Text similarity: {match_ratio:.1%} ({overlap}/{len(claim_words)} keywords match)")
    
    # Score based on match ratio
    if match_ratio > 0.5:
        # High match - text in image supports claim
        return {
            'has_text': True,
            'extracted_text': extracted_text[:500],
            'score': 0.9,
            'match_ratio': match_ratio,
            'info': f'✅ Image text strongly matches claim ({match_ratio:.0%} keyword match)',
            'warning': None
        }
    
    elif match_ratio > 0.2:
        # Moderate match - partial relevance
        return {
            'has_text': True,
            'extracted_text': extracted_text[:500],
            'score': 0.6,
            'match_ratio': match_ratio,
            'info': f'⚠️ Image text partially matches claim ({match_ratio:.0%} keyword match)',
            'warning': None
        }
    
    else:
        # Low match - potential mismatch
        return {
            'has_text': True,
            'extracted_text': extracted_text[:500],
            'score': 0.3,
            'match_ratio': match_ratio,
            'info': f'🚨 Image text does not match claim',
            'warning': f'Image contains different text than claim (only {match_ratio:.0%} keyword match)'
        }