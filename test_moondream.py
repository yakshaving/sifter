#!/usr/bin/env python3
"""
Test script to verify Moondream works OFFLINE.

Before running:
1. Make sure you've downloaded the model: moondream download moondream-2b
2. Disconnect Wi-Fi to verify offline mode

Usage:
    python test_moondream.py
    python test_moondream.py path/to/your/image.jpg
"""

import sys
from moondream import VisionModel
from PIL import Image
import os

def test_moondream(image_path=None):
    print("🌙 Testing Moondream (offline mode)...")

    # Check if model exists locally
    model_path = "moondream-2b"
    if not os.path.exists(model_path):
        print("❌ ERROR: Moondream model not found!")
        print("   Please run: moondream download moondream-2b")
        return

    print("✅ Found Moondream model")
    print("🔄 Loading model (this may take 10-20 seconds)...")

    try:
        # Load the model locally
        model = VisionModel(local_model_path=model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    # Use provided image or sample
    if image_path and os.path.exists(image_path):
        print(f"📸 Analyzing image: {image_path}")
    elif os.path.exists("sample_seeds.jpg"):
        image_path = "sample_seeds.jpg"
        print(f"📸 Analyzing sample image: {image_path}")
    else:
        print("❌ ERROR: No image found!")
        print("   Usage: python test_moondream.py path/to/image.jpg")
        print("   Or place a 'sample_seeds.jpg' in this directory")
        return

    try:
        img = Image.open(image_path)
        print("✅ Image loaded")
        print("🔍 Running Moondream analysis...")

        # Run inference
        description = model.describe(img)

        print("\n" + "="*60)
        print("MOONDREAM ANALYSIS:")
        print("="*60)
        print(description)
        print("="*60)
        print("\n✅ Offline test successful!")
        print("   (If you see this, Moondream is working without internet)")

    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_moondream(image_path)
