# test_ocr.py

from PIL import Image, ImageDraw, ImageFont
from utils import image_ocr

print("=" * 60)
print("OCR MODULE TEST")
print("=" * 60)

# Test 1: Create image with text
print("\nTest 1: Image with clear text")
print("-" * 40)

# Create test image with text
img = Image.new('RGB', (800, 200), color='white')
draw = ImageDraw.Draw(img)

# Try to use a font, fallback to default if not found
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

draw.text((50, 80), "Breaking: Major flood hits New York", fill='black', font=font)
img.save('test_ocr_image.png')
print("✅ Created test image: test_ocr_image.png")

# Extract text
extracted = image_ocr.extract_text_from_image(img)
print(f"\nExtracted text: '{extracted}'")

if "flood" in extracted.lower() and "york" in extracted.lower():
    print("✅ OCR working correctly!")
else:
    print("⚠️ OCR not extracting text properly")

# Test 2: Analyze with matching claim
print("\nTest 2: Matching claim")
print("-" * 40)

result = image_ocr.analyze_image_text(img, "Breaking news about flood in New York")
print(f"Score: {result['score']:.2f}")
print(f"Match ratio: {result.get('match_ratio', 0):.2%}")
print(f"Info: {result['info']}")

if result['score'] > 0.7:
    print("✅ High match detected!")
else:
    print("⚠️ Match not detected")

# Test 3: Analyze with non-matching claim
print("\nTest 3: Non-matching claim")
print("-" * 40)

result2 = image_ocr.analyze_image_text(img, "Stock market crashes today")
print(f"Score: {result2['score']:.2f}")
print(f"Match ratio: {result2.get('match_ratio', 0):.2%}")
print(f"Info: {result2['info']}")

if result2['score'] < 0.5:
    print("✅ Mismatch correctly detected!")
else:
    print("⚠️ Should detect mismatch")

# Test 4: Image without text (photo)
print("\nTest 4: Image without text")
print("-" * 40)

blank_img = Image.new('RGB', (400, 300), color='blue')
result3 = image_ocr.analyze_image_text(blank_img, "Some claim")
print(f"Has text: {result3['has_text']}")
print(f"Score: {result3['score']:.2f}")
print(f"Info: {result3['info']}")

if not result3['has_text']:
    print("✅ Correctly identified no text!")
else:
    print("⚠️ False positive - detected text where none exists")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)