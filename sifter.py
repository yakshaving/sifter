#!/usr/bin/env python3
"""
Unified Seed Sifter - Multiple detection modes with camera or file input
Usage:
    python sifter.py                    # Interactive menu
    python sifter.py --camera           # Live camera mode
    python sifter.py --image <path>     # Analyze image file
    python sifter.py --folder <path>    # Analyze folder of images
"""

import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path

CAPTURES_DIR = "captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)


def show_menu():
    """Interactive menu for mode selection."""
    print("\n" + "="*60)
    print("🌱 SEED SIFTER - Educational Seed Counter")
    print("="*60)
    print("\nSelect mode:")
    print("  1. Live Camera (OpenCV detection)")
    print("  2. Analyze Image File")
    print("  3. Analyze Folder of Images")
    print("  4. AI Analysis (Moondream)")
    print("  5. Exit")
    print("="*60)

    choice = input("\nEnter choice (1-5): ").strip()
    return choice


def separate_overlapping_seeds(contour, mask, min_area, max_single_seed_area=3000):
    """
    Separate overlapping seeds using watershed algorithm.
    Returns list of individual seed bounding boxes.
    """
    # If contour is small enough to be a single seed, return as-is
    area = cv2.contourArea(contour)
    if area < max_single_seed_area:
        x, y, w, h = cv2.boundingRect(contour)
        return [(x, y, w, h)]

    # Create a mask for this specific contour
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    # Distance transform to find seed centers
    dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)

    # Find local maxima (seed centers)
    # Use adaptive threshold based on the max distance
    max_dist = dist_transform.max()
    threshold = max_dist * 0.4  # Seeds centers should be at least 40% of max distance
    _, sure_fg = cv2.threshold(dist_transform, threshold, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Find individual seed markers
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # If we only found one component, return original contour
    if num_labels <= 2:  # 1 background + 1 seed
        x, y, w, h = cv2.boundingRect(contour)
        return [(x, y, w, h)]

    # Apply watershed
    markers = markers + 1  # Add 1 so background is not 0
    markers[contour_mask == 0] = 0  # Mark background as 0

    # Create a 3-channel image for watershed
    contour_region = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(contour_region, markers)

    # Extract bounding boxes for each separated seed
    separated_boxes = []
    for label in range(2, num_labels + 1):  # Skip background (1) and border (-1)
        seed_mask = np.uint8(markers == label) * 255
        seed_contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for sc in seed_contours:
            seed_area = cv2.contourArea(sc)
            if seed_area >= min_area:
                x, y, w, h = cv2.boundingRect(sc)
                separated_boxes.append((x, y, w, h))

    # If separation failed, return original
    if not separated_boxes:
        x, y, w, h = cv2.boundingRect(contour)
        return [(x, y, w, h)]

    return separated_boxes


def detect_seeds(image):
    """Detect seeds using tuned OpenCV parameters with overlapping seed separation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]

    detections = []
    min_area_pumpkin = 400
    min_area_sunflower = 500
    max_area = 40000
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Pumpkin seeds (green)
    mask_pumpkin = cv2.inRange(hsv, (22, 45, 45), (88, 255, 255))
    mask_pumpkin = cv2.erode(mask_pumpkin, kernel, iterations=1)
    mask_pumpkin = cv2.dilate(mask_pumpkin, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_pumpkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_pumpkin < area < max_area:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Use watershed to separate overlapping seeds
            separated_boxes = separate_overlapping_seeds(contour, mask_pumpkin, min_area_pumpkin)

            for (x, y, w, h) in separated_boxes:
                ar = w / h if h > 0 else 0
                if 0.3 < ar < 3.5 and solidity > 0.65:
                    detections.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'type': 'pumpkin', 'color': (0, 255, 0)
                    })

    # Sunflower seeds (tan) with proximity filter
    # Relaxed HSV range to catch more sunflower seeds
    mask_sunflower = cv2.inRange(hsv, (5, 35, 85), (24, 110, 190))
    mask_sunflower = cv2.erode(mask_sunflower, kernel, iterations=1)
    mask_sunflower = cv2.dilate(mask_sunflower, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_sunflower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_sunflower < area < max_area:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Use watershed to separate overlapping seeds
            separated_boxes = separate_overlapping_seeds(contour, mask_sunflower, min_area_sunflower)

            for (x, y, w, h) in separated_boxes:
                ar = w / h if h > 0 else 0

                # Relaxed solidity threshold from 0.72 to 0.68
                if 0.4 < ar < 3.0 and solidity > 0.68:
                    cx, cy = x + w//2, y + h//2
                    is_overlap = False
                    for d in detections:
                        if (d['x'] < cx < d['x']+d['w'] and d['y'] < cy < d['y']+d['h']):
                            is_overlap = True
                            break

                    if not is_overlap:
                        # Proximity filter - relaxed from 150 to 200 pixels
                        min_distance = float('inf')
                        for d in detections:
                            if d['type'] == 'pumpkin':
                                d_cx = d['x'] + d['w'] // 2
                                d_cy = d['y'] + d['h'] // 2
                                dist = ((cx - d_cx) ** 2 + (cy - d_cy) ** 2) ** 0.5
                                min_distance = min(min_distance, dist)

                        if min_distance < 200:
                            detections.append({
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'type': 'sunflower', 'color': (0, 165, 255)
                            })

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


def analyze_image_file(image_path):
    """Analyze a single image file."""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    print(f"\n📸 Analyzing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return

    detections = detect_seeds(image)
    annotated, pumpkin, sunflower = draw_detections(image, detections)

    print(f"\n📊 Results:")
    print(f"  🎃 Pumpkin seeds:   {pumpkin}")
    print(f"  🌻 Sunflower seeds: {sunflower}")
    print(f"  📝 Total:           {pumpkin + sunflower}")

    # Save annotated image
    output_path = str(Path(image_path).with_stem(Path(image_path).stem + '_annotated'))
    cv2.imwrite(output_path, annotated)
    print(f"\n💾 Saved: {output_path}")

    # Show image
    cv2.imshow(f"Seed Detection - {Path(image_path).name} (Press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def analyze_folder(folder_path):
    """Analyze all images in a folder."""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in Path(folder_path).iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return

    print(f"\n📁 Found {len(image_files)} images")
    print("="*60)

    for img_path in sorted(image_files):
        analyze_image_file(str(img_path))
        print("="*60)


def camera_mode():
    """Live camera detection mode."""
    print("\n🎥 Opening camera...")
    print("Controls:")
    print("  SPACEBAR: Analyze current frame")
    print("  q: Quit")
    print("="*60 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Seed Sifter - Press SPACEBAR to analyze, Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACEBAR
            print("\n🔍 Analyzing...")
            detections = detect_seeds(frame)
            annotated, pumpkin, sunflower = draw_detections(frame, detections)

            print(f"📊 Pumpkin: {pumpkin} | Sunflower: {sunflower} | Total: {pumpkin + sunflower}")

            cv2.imshow("Detection Results - Press any key to continue", annotated)
            cv2.waitKey(0)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Seed Sifter - Educational Seed Counter')
    parser.add_argument('--camera', action='store_true', help='Live camera mode')
    parser.add_argument('--image', type=str, help='Analyze image file')
    parser.add_argument('--folder', type=str, help='Analyze folder of images')
    parser.add_argument('--ai', action='store_true', help='Use AI analysis (Moondream)')

    args = parser.parse_args()

    if args.camera:
        camera_mode()
    elif args.image:
        analyze_image_file(args.image)
    elif args.folder:
        analyze_folder(args.folder)
    elif args.ai:
        # Launch AI analysis
        import subprocess
        subprocess.run([sys.executable, 'count_seeds.py'])
    else:
        # Interactive menu
        while True:
            choice = show_menu()

            if choice == '1':
                camera_mode()
            elif choice == '2':
                path = input("\nEnter image path: ").strip()
                analyze_image_file(path)
            elif choice == '3':
                path = input("\nEnter folder path: ").strip()
                analyze_folder(path)
            elif choice == '4':
                import subprocess
                subprocess.run([sys.executable, 'count_seeds.py'])
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice")


if __name__ == "__main__":
    main()
