#!/usr/bin/env python3
"""
Batch analyze all images in captures directory without displaying windows
Now with improved watershed separation for overlapping seeds
"""

import cv2
import glob
import os
from pathlib import Path

# Import detection function from sifter (now includes watershed separation)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sifter import detect_seeds, draw_detections


def batch_analyze():
    """Analyze all images in captures directory."""
    image_files = sorted(glob.glob("captures/capture_*.jpg"))

    if not image_files:
        print("❌ No images found in captures/")
        return

    print(f"\n📁 Found {len(image_files)} images")
    print("="*70)

    results = []

    for img_path in image_files:
        filename = Path(img_path).name
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ {filename}: Could not load")
            continue

        detections = detect_seeds(image)
        annotated, pumpkin, sunflower = draw_detections(image, detections)

        # Save annotated version
        output_path = img_path.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(output_path, annotated)

        total = pumpkin + sunflower
        results.append({
            'file': filename,
            'pumpkin': pumpkin,
            'sunflower': sunflower,
            'total': total
        })

        print(f"✓ {filename:30s}  🎃 {pumpkin:2d}  🌻 {sunflower:2d}  📝 {total:2d}")

    print("="*70)
    print(f"\n📊 SUMMARY OF {len(results)} IMAGES:")
    print("="*70)

    if results:
        avg_pumpkin = sum(r['pumpkin'] for r in results) / len(results)
        avg_sunflower = sum(r['sunflower'] for r in results) / len(results)
        avg_total = sum(r['total'] for r in results) / len(results)

        print(f"Average per image:")
        print(f"  🎃 Pumpkin:   {avg_pumpkin:.1f}")
        print(f"  🌻 Sunflower: {avg_sunflower:.1f}")
        print(f"  📝 Total:     {avg_total:.1f}")

        print(f"\nTotals across all images:")
        print(f"  🎃 Pumpkin:   {sum(r['pumpkin'] for r in results)}")
        print(f"  🌻 Sunflower: {sum(r['sunflower'] for r in results)}")
        print(f"  📝 Total:     {sum(r['total'] for r in results)}")

        print(f"\n✅ All annotated images saved to captures/ folder")


if __name__ == "__main__":
    batch_analyze()
