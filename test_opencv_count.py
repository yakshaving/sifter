#!/usr/bin/env python3
"""
Test OpenCV seed detection on a saved image.
Draws numbered boxes on each detected seed for verification.
"""

import cv2
import numpy as np
import sys
import glob
import os


def get_latest_capture():
    """Find the most recent capture."""
    captures = glob.glob("captures/capture_*.jpg")
    if not captures:
        return None
    return max(captures, key=os.path.getctime)


def detect_seeds_opencv(image):
    """Detect seeds using color-based OpenCV approach."""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Try different parameter combinations - iterate quickly on static image
    # Attempt: Relax sunflower constraints to catch more real seeds
    mask_pumpkin = cv2.inRange(hsv, (22, 45, 45), (88, 255, 255))
    mask_sunflower = cv2.inRange(hsv, (7, 38, 90), (22, 105, 180))  # Wider range for sunflower

    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_pumpkin = cv2.erode(mask_pumpkin, kernel, iterations=1)
    mask_pumpkin = cv2.dilate(mask_pumpkin, kernel, iterations=2)
    mask_sunflower = cv2.erode(mask_sunflower, kernel, iterations=1)
    mask_sunflower = cv2.dilate(mask_sunflower, kernel, iterations=2)

    # Find contours
    contours_p, _ = cv2.findContours(mask_pumpkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_s, _ = cv2.findContours(mask_sunflower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and collect detections
    pumpkin_seeds = []
    sunflower_seeds = []
    min_area_pumpkin = 400
    min_area_sunflower = 500  # Slightly relaxed from 600
    max_area = 40000

    for contour in contours_p:
        area = cv2.contourArea(contour)
        if area < min_area_pumpkin or area > max_area:
            continue

        # Calculate solidity to filter irregular shapes
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue
        if solidity < 0.65:
            continue

        pumpkin_seeds.append({'bbox': (x, y, w, h), 'area': area, 'contour': contour})

    for contour in contours_s:
        area = cv2.contourArea(contour)
        if area < min_area_sunflower or area > max_area:
            continue

        # Calculate solidity - wood grain tends to have lower solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 3.0:
            continue
        if solidity < 0.72:  # Stricter to filter wood grain
            continue

        # Check for overlap with pumpkin seeds
        cx, cy = x + w//2, y + h//2
        is_overlap = False
        for p in pumpkin_seeds:
            px, py, pw, ph = p['bbox']
            if px < cx < px+pw and py < cy < py+ph:
                is_overlap = True
                break

        if not is_overlap:
            # Additional filter: sunflower seeds should be near pumpkin seeds (in the cluster)
            # Find nearest pumpkin seed
            min_distance = float('inf')
            for p in pumpkin_seeds:
                px, py, pw, ph = p['bbox']
                p_center_x = px + pw // 2
                p_center_y = py + ph // 2
                distance = ((cx - p_center_x) ** 2 + (cy - p_center_y) ** 2) ** 0.5
                min_distance = min(min_distance, distance)

            # Tighter proximity: only accept sunflower seeds within 150 pixels of a pumpkin seed
            if min_distance < 150:
                sunflower_seeds.append({'bbox': (x, y, w, h), 'area': area, 'contour': contour})

    return pumpkin_seeds, sunflower_seeds


def draw_detections(image, pumpkin_seeds, sunflower_seeds):
    """Draw numbered boxes on detected seeds."""
    annotated = image.copy()

    # Draw pumpkin seeds in green
    for idx, seed in enumerate(pumpkin_seeds):
        x, y, w, h = seed['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw number label
        label = f"P{idx+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw sunflower seeds in orange
    for idx, seed in enumerate(sunflower_seeds):
        x, y, w, h = seed['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # Draw number label
        label = f"S{idx+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), (0, 165, 255), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Add count summary
    count_text = f"Pumpkin: {len(pumpkin_seeds)} | Sunflower: {len(sunflower_seeds)} | Total: {len(pumpkin_seeds) + len(sunflower_seeds)}"
    cv2.rectangle(annotated, (0, 0), (len(count_text) * 15, 40), (0, 0, 0), -1)
    cv2.putText(annotated, count_text, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = get_latest_capture()
        if not image_path:
            print("❌ No captures found")
            return

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    print(f"📸 Analyzing: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image")
        return

    # Detect seeds
    print("🔍 Running OpenCV detection...")
    pumpkin_seeds, sunflower_seeds = detect_seeds_opencv(image)

    print(f"\n📊 Detection Results:")
    print(f"  🎃 Pumpkin seeds: {len(pumpkin_seeds)}")
    print(f"  🌻 Sunflower seeds: {len(sunflower_seeds)}")
    print(f"  📝 Total: {len(pumpkin_seeds) + len(sunflower_seeds)}")

    # Draw annotations
    annotated = draw_detections(image, pumpkin_seeds, sunflower_seeds)

    # Save annotated image
    output_path = image_path.replace('.jpg', '_annotated.jpg')
    cv2.imwrite(output_path, annotated)
    print(f"\n💾 Saved annotated image: {output_path}")
    print(f"   Green boxes (P1, P2, ...) = Pumpkin seeds")
    print(f"   Orange boxes (S1, S2, ...) = Sunflower seeds")
    print(f"\n✅ Open the annotated image to verify detections")


if __name__ == "__main__":
    main()
