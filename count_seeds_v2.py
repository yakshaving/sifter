#!/usr/bin/env python3
"""
Enhanced seed counting with multiple AI passes for better accuracy.
"""

import os
import sys

# CRITICAL: Must disable MPS before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_default_device("cpu")
torch.backends.mps.is_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import re
import glob


def get_latest_capture():
    """Find the most recent capture in the captures directory."""
    captures = glob.glob("captures/capture_*.jpg")
    if not captures:
        return None
    return max(captures, key=os.path.getctime)


def analyze_with_chain_of_thought(image_path):
    """Use multi-pass analysis for more accurate counting."""
    print(f"📸 Analyzing: {image_path}")
    print("🌙 Loading Moondream model...")

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

        # Load image
        img = Image.open(image_path)
        enc_image = model.encode_image(img)

        # Pass 1: Describe what's in the image
        print("🔍 Pass 1: Analyzing image content...")
        q1 = "Describe what you see in this image. What objects are visible and how are they arranged?"
        description = model.answer_question(enc_image, q1, tokenizer)
        print(f"Description: {description}\n")

        # Pass 2: Identify seed types
        print("🔍 Pass 2: Identifying seed types...")
        q2 = "What types of seeds are visible in this image? Describe their colors and shapes."
        seed_types = model.answer_question(enc_image, q2, tokenizer)
        print(f"Seed types: {seed_types}\n")

        # Pass 3: Count pumpkin seeds specifically
        print("🔍 Pass 3: Counting pumpkin seeds...")
        q3 = "Count only the green/olive colored pumpkin seeds. Look carefully at each one, including overlapping seeds. How many pumpkin seeds do you see?"
        pumpkin_response = model.answer_question(enc_image, q3, tokenizer)
        print(f"Pumpkin count response: {pumpkin_response}\n")

        # Pass 4: Count sunflower seeds specifically
        print("🔍 Pass 4: Counting sunflower seeds...")
        q4 = "Count only the tan/beige/light colored sunflower seeds. Look carefully at each one, including overlapping seeds. How many sunflower seeds do you see?"
        sunflower_response = model.answer_question(enc_image, q4, tokenizer)
        print(f"Sunflower count response: {sunflower_response}\n")

        # Parse counts from responses
        pumpkin_numbers = re.findall(r'\d+', pumpkin_response)
        sunflower_numbers = re.findall(r'\d+', sunflower_response)

        print("=" * 60)
        print("📊 CHAIN-OF-THOUGHT ANALYSIS RESULTS")
        print("=" * 60)

        if pumpkin_numbers and sunflower_numbers:
            # Take the last number mentioned (often the final count)
            pumpkin = int(pumpkin_numbers[-1])
            sunflower = int(sunflower_numbers[-1])
            total = pumpkin + sunflower

            print(f"  🎃 Pumpkin seeds:   {pumpkin} (from: {pumpkin_response})")
            print(f"  🌻 Sunflower seeds: {sunflower} (from: {sunflower_response})")
            print(f"  📝 Total seeds:     {total}")
            if pumpkin > 0 and sunflower > 0:
                print(f"  📐 Ratio:           {pumpkin}:{sunflower}")
        else:
            print("⚠️  Could not parse counts from responses")
            print(f"Pumpkin response: {pumpkin_response}")
            print(f"Sunflower response: {sunflower_response}")

        print("=" * 60 + "\n")

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
            print("   Run: python count_seeds.py")
            return

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    analyze_with_chain_of_thought(image_path)


if __name__ == "__main__":
    main()
