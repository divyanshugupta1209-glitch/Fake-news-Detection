# utils/reverse_image_search.py

import os
from google.cloud import vision
import io
from pathlib import Path
from dotenv import load_dotenv

# Force reload environment variables
load_dotenv(override=True)

# Initialize Google Vision client (Windows-compatible)
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
client = None
initialization_error = None

print(f"🔍 Looking for credentials: {credentials_path}")

if not credentials_path:
    initialization_error = "GOOGLE_APPLICATION_CREDENTIALS not set in .env"
    print(f"⚠️ {initialization_error}")
else:
    # Convert to Path object (works on Windows)
    cred_file = Path(credentials_path)
    
    # If relative path, make it absolute
    if not cred_file.is_absolute():
        cred_file = Path.cwd() / cred_file
    
    print(f"🔍 Full path: {cred_file}")
    print(f"📁 Current directory: {Path.cwd()}")
    
    if not cred_file.exists():
        initialization_error = f"Credentials file not found: {cred_file}"
        print(f"⚠️ {initialization_error}")
        # List JSON files in current directory
        json_files = list(Path.cwd().glob("*.json"))
        print(f"📁 JSON files in current dir: {[f.name for f in json_files]}")
    else:
        try:
            # Set environment variable with absolute path
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
            client = vision.ImageAnnotatorClient()
            print(f"✅ Google Vision API initialized: {cred_file.name}")
        except Exception as e:
            initialization_error = f"Failed to initialize: {str(e)}"
            print(f"⚠️ {initialization_error}")


def reverse_image_search(image):
    """
    Find where this image has appeared before on the web.
    """
    if not client:
        return {
            'error': initialization_error or 'Google Vision API not initialized',
            'found_elsewhere': False,
            'score': 0.5
        }
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        vision_image = vision.Image(content=img_byte_arr)
        
        # Call API
        print("   📡 Calling Google Vision API...")
        response = client.web_detection(image=vision_image)
        web_detection = response.web_detection
        print("   ✅ API responded successfully")
        
        # Extract results
        pages_with_image = []
        if web_detection.pages_with_matching_images:
            for page in web_detection.pages_with_matching_images[:10]:
                pages_with_image.append({
                    'url': page.url,
                    'title': getattr(page, 'page_title', 'Unknown')
                })
        
        best_guess = ""
        if web_detection.best_guess_labels:
            best_guess = web_detection.best_guess_labels[0].label
        
        entities = []
        if web_detection.web_entities:
            entities = [entity.description for entity in web_detection.web_entities[:5] 
                       if hasattr(entity, 'description')]
        
        print(f"   📊 Found in {len(pages_with_image)} sources")
        if best_guess:
            print(f"   🏷️ Google says: {best_guess}")
        
        return {
            'found_elsewhere': len(pages_with_image) > 0,
            'num_sources': len(pages_with_image),
            'sources': pages_with_image[:5],
            'best_guess': best_guess,
            'entities': entities,
            'error': None
        }
    
    except Exception as e:
        error_msg = f"API error: {str(e)}"
        print(f"   ⚠️ {error_msg}")
        return {
            'error': error_msg,
            'found_elsewhere': False,
            'score': 0.5
        }


def check_image_context(image, claim_text):
    """
    Check if image context matches the claim.
    """
    search_results = reverse_image_search(image)
    
    # Handle errors
    if search_results.get('error'):
        return {
            'score': 0.5,
            'warning': None,
            'info': f'⚠️ Reverse search unavailable: {search_results["error"]}',
            'sources': []
        }
    
    # Image not found elsewhere = original
    if not search_results['found_elsewhere']:
        return {
            'score': 0.85,
            'warning': None,
            'info': '✅ Image appears to be original (not found elsewhere online)',
            'sources': []
        }
    
    # Image found - check context match
    num_sources = search_results['num_sources']
    entities = search_results.get('entities', [])
    best_guess = search_results.get('best_guess', '')
    
    # Extract keywords from claim
    claim_lower = claim_text.lower()
    claim_words = set(claim_lower.split())
    
    # Extract keywords from image context
    image_context = ' '.join(entities + [best_guess]).lower()
    image_words = set(image_context.split())
    
    # Calculate match
    overlap = len(claim_words & image_words)
    match_ratio = overlap / max(len(claim_words), 1)
    
    print(f"   🔍 Context match: {match_ratio:.2%}")
    
    # Decision
    if match_ratio > 0.4:
        # Good match
        return {
            'score': 0.75,
            'warning': None,
            'info': f'✅ Image found in {num_sources} sources with similar context',
            'sources': search_results['sources'][:3],
            'context': f"Image context: {', '.join(entities[:3]) or best_guess}"
        }
    else:
        # MISMATCH - out of context!
        return {
            'score': 0.15,
            'warning': '🚨 OUT-OF-CONTEXT IMAGE DETECTED!',
            'info': f'This image appears in {num_sources} other sources with DIFFERENT context',
            'sources': search_results['sources'][:3],
            'context': f"Image actually shows: {best_guess or ', '.join(entities[:3])}"
        }