#!/usr/bin/env python3
"""
Analyze a saved capture image using Moondream AI for accurate seed counting.

Usage:
    python analyze_capture.py captures/capture_1234567890.jpg
    python analyze_capture.py  # Analyzes most recent capture
"""

import sys
import os
import glob

# CRITICAL: Must disable MPS before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
# Force PyTorch to use CPU only
torch.set_default_device("cpu")
# Disable MPS backend completely
torch.backends.mps.is_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import re


def get_latest_capture():
    """Find the most recent capture in the captures directory."""
    captures = glob.glob("captures/capture_*.jpg")
    if not captures:
        return None
    return max(captures, key=os.path.getctime)


def analyze_with_moondream(image_path):
    """Use Moondream to count seeds in the image."""
    print(f"📸 Analyzing: {image_path}")
    print("🌙 Loading Moondream model (first time: downloads ~4GB)...")
    print("   This may take 1-2 minutes on first run...\n")

    try:
        # Load Moondream
        model_id = "vikhyatk/moondream2"
        revision = "2025-01-09"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        print("✅ Model loaded!\n")

        # Load image
        img = Image.open(image_path)
        enc_image = model.encode_image(img)

        # Ask Moondream to count seeds
        question = """Look carefully at this image and count EVERY individual seed, including overlapping ones.
There are two types:
1. Green/olive colored pumpkin seeds (larger, oval shaped)
2. Tan/beige/light brown sunflower seeds (smaller, teardrop shaped)

Count each seed separately, even if they are touching or overlapping. Be thorough and count all visible seeds.
Provide ONLY the counts in the format: 'Pumpkin: X, Sunflower: Y'"""

        print("🔍 Asking Moondream to count seeds (detailed analysis)...")
        response = model.answer_question(
            enc_image,
            question,
            tokenizer
        )

        print("\n" + "="*60)
        print("MOONDREAM ANALYSIS:")
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

            print("📊 FINAL COUNTS:")
            print(f"   🎃 Pumpkin seeds: {pumpkin}")
            print(f"   🌻 Sunflower seeds: {sunflower}")
            print(f"   📝 Total: {total}")
            if pumpkin > 0 and sunflower > 0:
                print(f"   📐 Ratio: {pumpkin}:{sunflower}")
            print()
        else:
            print("⚠️  Could not parse counts from Moondream response")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use most recent capture
        image_path = get_latest_capture()
        if not image_path:
            print("❌ No captures found in captures/ directory")
            print("   Run ./run.sh and press SPACEBAR to capture an image first")
            return

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    analyze_with_moondream(image_path)


if __name__ == "__main__":
    main()
