# utils/cache.py

import hashlib
import json
import time
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_EXPIRY = 3600 * 24 * 7  # 7 days in seconds


def get_cache_key(claim_text, has_image=False):
    """Generate unique cache key from claim"""
    text = claim_text.strip().lower()
    suffix = "_img" if has_image else "_txt"
    return hashlib.md5(f"{text}{suffix}".encode()).hexdigest()


def get_cached_result(claim_text, has_image=False):
    """
    Retrieve cached result if exists and not expired.
    
    Returns:
        dict or None: Cached result or None if not found/expired
    """
    key = get_cache_key(claim_text, has_image)
    cache_file = CACHE_DIR / f"{key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check expiry
        if time.time() - data.get('timestamp', 0) > CACHE_EXPIRY:
            cache_file.unlink()  # Delete expired cache
            return None
        
        return data.get('result')
    
    except Exception as e:
        print(f"Cache read error: {e}")
        return None


def save_to_cache(claim_text, result, has_image=False):
    """
    Save result to cache.
    
    Args:
        claim_text: The claim text
        result: Dict with {label, score, details, explanation}
        has_image: Whether image was used in analysis
    """
    key = get_cache_key(claim_text, has_image)
    cache_file = CACHE_DIR / f"{key}.json"
    
    try:
        data = {
            'timestamp': time.time(),
            'claim': claim_text,
            'has_image': has_image,
            'result': result
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    except Exception as e:
        print(f"Cache write error: {e}")


def clear_expired_cache():
    """Clean up expired cache files"""
    try:
        count = 0
        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                if time.time() - data.get('timestamp', 0) > CACHE_EXPIRY:
                    cache_file.unlink()
                    count += 1
            except:
                cache_file.unlink()  # Delete corrupted cache
                count += 1
        
        if count > 0:
            print(f"🧹 Cleared {count} expired cache files")
    
    except Exception as e:
        print(f"Cache cleanup error: {e}")


def get_cache_stats():
    """Get cache statistics"""
    try:
        files = list(CACHE_DIR.glob("*.json"))
        valid = 0
        expired = 0
        
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                if time.time() - data.get('timestamp', 0) <= CACHE_EXPIRY:
                    valid += 1
                else:
                    expired += 1
            except:
                expired += 1
        
        return {
            'total': len(files),
            'valid': valid,
            'expired': expired
        }
    except:
        return {'total': 0, 'valid': 0, 'expired': 0}


# Clean expired cache on module load
clear_expired_cache()

print("✅ Cache module loaded")