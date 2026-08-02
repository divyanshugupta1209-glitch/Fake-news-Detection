# utils/image_utils.py

from PIL import Image
import io

def compress_image(image, max_size=(800, 800), quality=85):
    """
    Compress image to reduce processing time.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimensions (width, height)
        quality: JPEG quality (1-100)
    
    Returns:
        Compressed PIL Image
    """
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if larger than max_size
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Compress to bytes and reload (reduces memory)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    
    compressed = Image.open(buffer)
    
    return compressed


def get_image_info(image):
    """Get image metadata"""
    return {
        'size': image.size,
        'mode': image.mode,
        'format': getattr(image, 'format', 'Unknown'),
        'width': image.width,
        'height': image.height,
    }

print("✅ Image utilities module loaded")