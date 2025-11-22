#!/usr/bin/env python3
"""
Test script to verify Moondream works OFFLINE.

On first run, this will download the model from Hugging Face (~4GB).
After that, it works completely offline.

Usage:
    python test_moondream.py
    python test_moondream.py path/to/your/image.jpg
"""

import sys
import os

# CRITICAL: Must disable MPS before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
# Force PyTorch to use CPU only
torch.set_default_device("cpu")
# Disable MPS backend completely
torch.backends.mps.is_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def test_moondream(image_path=None):
    print("🌙 Testing Moondream (offline mode)...")
    print("🔄 Loading model (first time: downloads ~4GB, then cached)...")
    print("   This may take 1-2 minutes on first run...")

    try:
        # Load Moondream from Hugging Face (auto-downloads first time, then cached)
        model_id = "vikhyatk/moondream2"
        revision = "2025-01-09"

        # Load on CPU to avoid device conflicts
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print("\n   Troubleshooting:")
        print("   - Make sure you have internet on FIRST run (to download)")
        print("   - After first run, disconnect Wi-Fi to test offline mode")
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

        # Encode image
        enc_image = model.encode_image(img)

        # Generate description
        description = model.answer_question(
            enc_image,
            "Describe this image in detail.",
            tokenizer
        )

        print("\n" + "="*60)
        print("MOONDREAM ANALYSIS:")
        print("="*60)
        print(description)
        print("="*60)
        print("\n✅ Test successful!")
        print("   (Model is now cached locally - works offline!)")

    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_moondream(image_path)
