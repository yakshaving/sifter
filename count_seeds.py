#!/usr/bin/env python3
"""
One-command seed counting: Capture from camera and analyze with Moondream AI.

Usage:
    python count_seeds.py           # Shows live preview, press SPACEBAR to capture & analyze
    python count_seeds.py --auto    # Auto-capture after 3 seconds, no interaction needed
"""

import os
import cv2
import time
import sys

# CRITICAL: Must disable MPS before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_default_device("cpu")
torch.backends.mps.is_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import re

CAPTURES_DIR = "captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)


def capture_image(auto_mode=False):
    """Capture an image from webcam."""
    print("🎥 Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera!")
        return None

    time.sleep(0.5)  # Let camera warm up

    print("✅ Camera ready!")

    if auto_mode:
        print("⏱️  Auto-capturing in 3 seconds...")
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            print(f"   {i}...")
            time.sleep(1)

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            cap.release()
            return None

        print("📸 Captured!")
    else:
        print("\n" + "="*60)
        print("LIVE PREVIEW - Press SPACEBAR to capture, Q to quit")
        print("="*60 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                cap.release()
                return None

            # Show preview
            cv2.imshow("Seed Counter - Press SPACEBAR to capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACEBAR
                print("📸 Captured!")
                break
            elif key == ord('q'):
                print("👋 Cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return None

    # Save capture
    timestamp = int(time.time())
    filename = os.path.join(CAPTURES_DIR, f"capture_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"💾 Saved: {filename}\n")

    cap.release()
    cv2.destroyAllWindows()

    return filename


def analyze_with_moondream(image_path):
    """Analyze image with Moondream AI."""
    print("🌙 Loading Moondream model...")
    print("   (First time: downloads ~4GB, then cached)")
    print("   This may take 1-2 minutes on first run...\n")

    try:
        model_id = "vikhyatk/moondream2"
        revision = "2025-01-09"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        print("✅ Model loaded!\n")

        # Load and analyze image
        img = Image.open(image_path)
        enc_image = model.encode_image(img)

        question = """Look carefully at this image and count EVERY individual seed, including overlapping ones.
There are two types:
1. Green/olive colored pumpkin seeds (larger, oval shaped)
2. Tan/beige/light brown sunflower seeds (smaller, teardrop shaped)

Count each seed separately, even if they are touching or overlapping. Be thorough and count all visible seeds.
Provide ONLY the counts in the format: 'Pumpkin: X, Sunflower: Y'"""

        print("🔍 Counting seeds with AI (detailed analysis)...")
        response = model.answer_question(enc_image, question, tokenizer)

        print("\n" + "="*60)
        print("🌙 MOONDREAM ANALYSIS")
        print("="*60)
        print(f"Response: {response}")
        print("="*60 + "\n")

        # Parse counts
        pumpkin_match = re.search(r'[Pp]umpkin.*?(\d+)', response)
        sunflower_match = re.search(r'[Ss]unflower.*?(\d+)', response)

        if pumpkin_match and sunflower_match:
            pumpkin = int(pumpkin_match.group(1))
            sunflower = int(sunflower_match.group(1))
            total = pumpkin + sunflower

            print("=" * 60)
            print("📊 FINAL SEED COUNT")
            print("=" * 60)
            print(f"  🎃 Pumpkin seeds:   {pumpkin}")
            print(f"  🌻 Sunflower seeds: {sunflower}")
            print(f"  📝 Total seeds:     {total}")
            if pumpkin > 0 and sunflower > 0:
                print(f"  📐 Ratio:           {pumpkin}:{sunflower}")
            print("=" * 60 + "\n")
        else:
            print("⚠️  Could not parse counts from Moondream response\n")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🌱 Seed Counter - Capture + AI Analysis")
    print("="*60 + "\n")

    auto_mode = "--auto" in sys.argv

    # Step 1: Capture
    image_path = capture_image(auto_mode=auto_mode)
    if not image_path:
        return

    # Step 2: Analyze
    analyze_with_moondream(image_path)


if __name__ == "__main__":
    main()
