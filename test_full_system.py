# test_full_system.py

from PIL import Image
from utils import detection
import time

print("=" * 60)
print("FULL MULTIMODAL SYSTEM TEST")
print("=" * 60)

# Create test image
test_img = Image.new('RGB', (400, 300), color='blue')

# Test claim
claim = "Breaking: Major flooding hits coastal city"

print("\n🔍 Testing complete image analysis pipeline...")
print(f"Claim: {claim}")
print("-" * 60)

start_time = time.time()

# Run full detection
label, score, details = detection.fuse_predictions(
    claim_text=claim,
    article_text=claim,
    url="https://example.com",
    image=test_img,
    hf_result=None
)

elapsed = time.time() - start_time

print("\n📊 RESULTS:")
print("-" * 60)
print(f"Final Verdict: {label}")
print(f"Final Score: {score:.3f}")
print(f"Processing Time: {elapsed:.2f} seconds")

print("\n🔬 Component Scores:")
print("-" * 60)
print(f"├─ HF Text Score: {details.get('hf_score', 0):.3f}")
print(f"├─ Image Score (Combined): {details.get('image_score', 0):.3f}")

if details.get('ocr_result'):
    ocr = details['ocr_result']
    print(f"│  ├─ OCR Score: {ocr.get('score', 0):.3f}")
    print(f"│  └─ Has Text: {ocr.get('has_text', False)}")

if details.get('ai_detection_result'):
    ai = details['ai_detection_result']
    print(f"│  ├─ AI Detection Score: {ai.get('score', 0):.3f}")
    print(f"│  ├─ Is AI: {ai.get('is_ai', False)}")
    print(f"│  └─ Confidence: {ai.get('confidence', 0):.1%}")

print("\n" + "=" * 60)

# Check if all components ran
components_ran = {
    'HF': details.get('hf_score') is not None,
    'Image': details.get('image_score') is not None,
    'OCR': details.get('ocr_result') is not None,
    'AI Detection': details.get('ai_detection_result') is not None
}

print("\n✅ Components Status:")
for component, status in components_ran.items():
    symbol = "✅" if status else "❌"
    print(f"{symbol} {component}: {'Running' if status else 'Missing'}")

print("\n" + "=" * 60)

if all(components_ran.values()):
    print("🎉 ALL COMPONENTS WORKING!")
else:
    print("⚠️ Some components missing - check errors above")