#!/usr/bin/env python3
"""
SAM-based Seed Sifter - Uses Segment Anything Model for accurate instance segmentation
Much faster and more accurate than watershed for separating overlapping seeds.

Approach:
1. Grid/mesh division of image for parallel processing
2. Color-based filtering to identify seed regions
3. SAM automatic mask generation for instance segmentation
4. Classification based on color and shape features
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not installed. Install with: pip install segment-anything")

CAPTURES_DIR = "captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)


class SAMSeedSifter:
    """Seed detection using SAM for instance segmentation."""

    def __init__(self, sam_checkpoint=None, model_type="vit_b"):
        """Initialize SAM model."""
        if not SAM_AVAILABLE:
            raise ImportError("SAM is not installed. Run: pip install segment-anything")

        # Try to find SAM checkpoint
        if sam_checkpoint is None:
            # Common locations for SAM checkpoint (check ViT-B first, then ViT-H)
            possible_paths = [
                "sam_vit_b_01ec64.pth",
                "sam_vit_h_4b8939.pth",
                "models/sam_vit_b_01ec64.pth",
                "models/sam_vit_h_4b8939.pth",
                os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth"),
                os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    sam_checkpoint = path
                    # Auto-detect model type from checkpoint name
                    if "vit_h" in path:
                        model_type = "vit_h"
                    elif "vit_b" in path:
                        model_type = "vit_b"
                    break

        if sam_checkpoint is None or not os.path.exists(sam_checkpoint):
            print("\n⚠️  SAM checkpoint not found!")
            print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            print("Or use smaller model: sam_vit_b_01ec64.pth")
            raise FileNotFoundError("SAM checkpoint not found")

        print(f"🤖 Loading SAM model from {sam_checkpoint}...")
        device = "cpu"  # Use CPU for now, can switch to "cuda" if GPU available
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        # Configure mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,  # Grid points for segmentation
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=400,  # Minimum seed size
        )
        print("✅ SAM model loaded!")

    def create_grid_regions(self, image, grid_size=3):
        """Divide image into grid regions for parallel processing."""
        height, width = image.shape[:2]
        regions = []

        h_step = height // grid_size
        w_step = width // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * h_step
                y2 = (i + 1) * h_step if i < grid_size - 1 else height
                x1 = j * w_step
                x2 = (j + 1) * w_step if j < grid_size - 1 else width

                regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'crop': image[y1:y2, x1:x2]
                })

        return regions

    def get_seed_mask(self, image):
        """Create binary mask of potential seed regions using color filtering."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Pumpkin seeds (green)
        mask_pumpkin = cv2.inRange(hsv, (22, 45, 45), (88, 255, 255))

        # Sunflower seeds (tan/beige)
        mask_sunflower = cv2.inRange(hsv, (5, 35, 85), (24, 110, 190))

        # Combine masks
        seed_mask = cv2.bitwise_or(mask_pumpkin, mask_sunflower)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel)

        return seed_mask, mask_pumpkin, mask_sunflower

    def classify_seed(self, mask, image, mask_pumpkin, mask_sunflower):
        """Classify seed as pumpkin or sunflower based on color."""
        # Get the region of the seed
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None

        # Check overlap with color masks
        pumpkin_overlap = np.sum(mask & mask_pumpkin)
        sunflower_overlap = np.sum(mask & mask_sunflower)

        # Classify based on dominant color
        if pumpkin_overlap > sunflower_overlap:
            return 'pumpkin'
        elif sunflower_overlap > 0:
            return 'sunflower'
        return None

    def detect_seeds(self, image):
        """Detect seeds using SAM segmentation."""
        print("   🔍 Running SAM segmentation...")
        start_time = time.time()

        # Get seed regions using color filtering
        seed_mask, mask_pumpkin, mask_sunflower = self.get_seed_mask(image)

        # Run SAM on the image
        masks = self.mask_generator.generate(image)
        print(f"   📊 SAM found {len(masks)} segments in {time.time() - start_time:.2f}s")

        # Filter and classify masks
        detections = []
        pumpkin_count = 0
        sunflower_count = 0

        for mask_data in masks:
            mask = mask_data['segmentation'].astype(np.uint8) * 255

            # Check if this segment overlaps with seed regions
            overlap = np.sum(mask & seed_mask)
            mask_area = np.sum(mask > 0)

            if overlap < mask_area * 0.3:  # At least 30% overlap with seed colors
                continue

            # Check size constraints
            if mask_area < 400 or mask_area > 40000:
                continue

            # Classify the seed
            seed_type = self.classify_seed(mask, image, mask_pumpkin, mask_sunflower)
            if seed_type is None:
                continue

            # Get bounding box
            y_indices, x_indices = np.where(mask > 0)
            x, y = x_indices.min(), y_indices.min()
            w, h = x_indices.max() - x + 1, y_indices.max() - y + 1

            # Check aspect ratio
            ar = w / h if h > 0 else 0
            if ar < 0.3 or ar > 3.5:
                continue

            # Proximity filter for sunflower seeds
            if seed_type == 'sunflower':
                cx, cy = x + w//2, y + h//2

                # Check for overlap with pumpkin
                is_overlap = False
                for d in detections:
                    if d['type'] == 'pumpkin':
                        if (d['x'] < cx < d['x']+d['w'] and d['y'] < cy < d['y']+d['h']):
                            is_overlap = True
                            break

                if is_overlap:
                    continue

                # Check proximity to pumpkin seeds
                min_distance = float('inf')
                for d in detections:
                    if d['type'] == 'pumpkin':
                        d_cx = d['x'] + d['w'] // 2
                        d_cy = d['y'] + d['h'] // 2
                        dist = ((cx - d_cx) ** 2 + (cy - d_cy) ** 2) ** 0.5
                        min_distance = min(min_distance, dist)

                if min_distance > 200:  # Too far from pumpkin seeds
                    continue

            # Add detection
            color = (0, 255, 0) if seed_type == 'pumpkin' else (0, 165, 255)
            detections.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'type': seed_type,
                'color': color,
                'confidence': mask_data['stability_score']
            })

            if seed_type == 'pumpkin':
                pumpkin_count += 1
            else:
                sunflower_count += 1

        print(f"   ✅ Detected {pumpkin_count} pumpkin, {sunflower_count} sunflower seeds")
        return detections


def draw_detections(image, detections):
    """Draw numbered boxes on detected seeds."""
    annotated = image.copy()
    pumpkin_idx = 0
    sunflower_idx = 0

    for det in detections:
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        color = det['color']
        seed_type = det['type']

        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)
        cv2.circle(annotated, (x+w//2, y+h//2), 5, color, -1)

        # Number label
        if seed_type == 'pumpkin':
            pumpkin_idx += 1
            label = f"P{pumpkin_idx}"
        else:
            sunflower_idx += 1
            label = f"S{sunflower_idx}"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y - 5, 20)
        cv2.rectangle(annotated, (x, label_y - 20), (x + label_size[0] + 10, label_y), color, -1)
        cv2.putText(annotated, label, (x + 5, label_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Count summary
    pumpkin_count = sum(1 for d in detections if d['type'] == 'pumpkin')
    sunflower_count = sum(1 for d in detections if d['type'] == 'sunflower')
    count_text = f"Pumpkin: {pumpkin_count} | Sunflower: {sunflower_count} | Total: {len(detections)}"

    cv2.rectangle(annotated, (0, 0), (len(count_text) * 15, 40), (0, 0, 0), -1)
    cv2.putText(annotated, count_text, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated, pumpkin_count, sunflower_count


def analyze_image(image_path, sam_sifter):
    """Analyze a single image with SAM."""
    print(f"\n📸 Analyzing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return

    detections = sam_sifter.detect_seeds(image)
    annotated, pumpkin, sunflower = draw_detections(image, detections)

    print(f"\n📊 Results:")
    print(f"  🎃 Pumpkin seeds:   {pumpkin}")
    print(f"  🌻 Sunflower seeds: {sunflower}")
    print(f"  📝 Total:           {pumpkin + sunflower}")

    # Save annotated image
    output_path = str(Path(image_path).with_stem(Path(image_path).stem + '_sam'))
    cv2.imwrite(output_path, annotated)
    print(f"\n💾 Saved: {output_path}")

    # Show image
    cv2.imshow(f"SAM Detection - {Path(image_path).name}", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    if not SAM_AVAILABLE:
        print("❌ SAM is not installed!")
        print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return

    print("\n" + "="*70)
    print("🤖 SAM-BASED SEED SIFTER")
    print("="*70)

    # Initialize SAM
    try:
        sam_sifter = SAMSeedSifter()
    except Exception as e:
        print(f"\n❌ Error initializing SAM: {e}")
        print("\nTo use SAM:")
        print("1. Install: pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("2. Download checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("   Or smaller model: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return

    # Get image path
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("\nEnter image path (or press Enter for test image): ").strip()
        if not image_path:
            image_path = "captures/capture_1764557172.jpg"

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    analyze_image(image_path, sam_sifter)


if __name__ == "__main__":
    main()
