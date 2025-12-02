#!/usr/bin/env python3
"""
Seed Sifter - OpenCV-based seed detection and counting

Hardware: Mac webcam only
Controls:
    SPACEBAR - Capture and analyze current frame
    c - Switch camera
    q - Quit

Detects pumpkin seeds (green) and sunflower seeds (black/striped).
"""

import os
import cv2
import time
import threading
import numpy as np

# Configuration
WINDOW_NAME = "Seed Sifter"
CAPTURES_DIR = "captures"
CAMERA_INDEX = 0  # Built-in webcam. Set to 1 for external camera.

# Create captures directory if it doesn't exist
os.makedirs(CAPTURES_DIR, exist_ok=True)

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


class SeedSifter:
    def __init__(self):
        print("🌱 Initializing Seed Sifter...")

        # Initialize camera with retry logic
        camera_index = CAMERA_INDEX if CAMERA_INDEX is not None else 0
        print(f"🎥 Opening camera {camera_index}...")

        max_retries = 5
        test_frame = None
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"   Retry {attempt}/{max_retries-1}... waiting 2 seconds")
                time.sleep(2)

            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Could not open camera {camera_index} after {max_retries} attempts!")
                continue

            time.sleep(0.5)
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"✅ Camera {camera_index} connected on attempt {attempt + 1}")
                break
            else:
                self.cap.release()
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Camera {camera_index} opened but cannot grab frames!")

        if test_frame is None:
            raise RuntimeError(f"Failed to initialize camera {camera_index}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized! ({width}x{height})")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACEBAR - Capture and analyze seeds")
        print("  c - Switch camera (toggle between 0 and 1)")
        print("  q - Quit")
        print("="*60 + "\n")

        # State
        self.last_analysis = "Use DARK background! Press SPACEBAR to analyze..."
        self.last_detections = []
        self.analyzing = False
        self.current_camera = camera_index

    def detect_seeds_opencv(self, frame):
        """
        Detect pumpkin seeds (green) and sunflower seeds (tan/beige).
        Note: Sunflower seeds need a dark background to be detected reliably.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]

        detections = []
        min_area_pumpkin = 400  # Moderate threshold for pumpkin
        min_area_sunflower = 500  # Tuned to filter wood grain
        max_area = 40000
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # --- PUMPKIN SEEDS: Green/yellow, high saturation ---
        mask_pumpkin = cv2.inRange(hsv, (22, 45, 45), (88, 255, 255))
        mask_pumpkin = cv2.erode(mask_pumpkin, kernel, iterations=1)
        mask_pumpkin = cv2.dilate(mask_pumpkin, kernel, iterations=2)

        contours, _ = cv2.findContours(mask_pumpkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area_pumpkin < area < max_area:
                # Calculate solidity to filter irregular shapes (wood grain)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Use watershed to separate overlapping seeds
                separated_boxes = separate_overlapping_seeds(contour, mask_pumpkin, min_area_pumpkin)

                for (x, y, w, h) in separated_boxes:
                    ar = w / h if h > 0 else 0

                    # More strict filtering
                    if 0.3 < ar < 3.5 and solidity > 0.65:
                        detections.append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'area': area, 'type': 'pumpkin', 'color': (0, 255, 0)
                        })

        # --- SUNFLOWER SEEDS: Tan/beige ---
        # Uses proximity filter to avoid wood grain false positives
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
                        # Check overlap with pumpkin
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
                                    d_center_x = d['x'] + d['w'] // 2
                                    d_center_y = d['y'] + d['h'] // 2
                                    distance = ((cx - d_center_x) ** 2 + (cy - d_center_y) ** 2) ** 0.5
                                    min_distance = min(min_distance, distance)

                            # Only accept if within 200 pixels of a pumpkin seed
                            if min_distance < 200:
                                detections.append({
                                    'x': x, 'y': y, 'w': w, 'h': h,
                                    'area': area, 'type': 'sunflower', 'color': (0, 165, 255)
                                })

        return detections

    def analyze_seeds(self, frame):
        """Analyze frame and count seeds by type."""
        print("   🔍 Detecting seeds...")
        start_time = time.time()

        detections = self.detect_seeds_opencv(frame)
        detection_time = time.time() - start_time
        print(f"   ⏱️  Detection time: {detection_time:.3f}s")

        # Count by type
        pumpkin_count = sum(1 for d in detections if d.get('type') == 'pumpkin')
        sunflower_count = sum(1 for d in detections if d.get('type') == 'sunflower')
        total = len(detections)

        print(f"   🎃 Pumpkin seeds: {pumpkin_count}")
        print(f"   🌻 Sunflower seeds: {sunflower_count}")

        # Store detections for drawing
        height, width = frame.shape[:2]
        self.last_detections = []
        for det in detections:
            self.last_detections.append({
                'x_min': det['x'] / width,
                'y_min': det['y'] / height,
                'x_max': (det['x'] + det['w']) / width,
                'y_max': (det['y'] + det['h']) / height,
                'type': det.get('type', 'unknown'),
                'color': det.get('color', (0, 255, 0))
            })

        if total == 0:
            return "No seeds detected - try adjusting lighting or background"

        # Calculate ratio
        if pumpkin_count > 0 and sunflower_count > 0:
            ratio = f" | Ratio {pumpkin_count}:{sunflower_count}"
        else:
            ratio = ""

        return f"Pumpkin: {pumpkin_count} | Sunflower: {sunflower_count} | Total: {total}{ratio}"

    def draw_ui(self, frame):
        """Draw UI overlay on frame with bounding boxes"""
        height, width = frame.shape[:2]

        # Draw bounding boxes with numbered labels
        pumpkin_idx = 0
        sunflower_idx = 0

        for detection in self.last_detections:
            try:
                if not isinstance(detection, dict):
                    continue

                x_min = detection.get('x_min', 0)
                y_min = detection.get('y_min', 0)
                x_max = detection.get('x_max', 0)
                y_max = detection.get('y_max', 0)
                color = detection.get('color', (0, 255, 0))
                seed_type = detection.get('type', 'unknown')

                x1 = int(x_min * width)
                y1 = int(y_min * height)
                x2 = int(x_max * width)
                y2 = int(y_max * height)

                # Draw rectangle and center dot
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

                # Add numbered label
                if seed_type == 'pumpkin':
                    pumpkin_idx += 1
                    label = f"P{pumpkin_idx}"
                elif seed_type == 'sunflower':
                    sunflower_idx += 1
                    label = f"S{sunflower_idx}"
                else:
                    label = "?"

                # Draw label background and text
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1 - 5, 20)  # Don't go off-screen
                cv2.rectangle(frame, (x1, label_y - 20), (x1 + label_size[0] + 10, label_y), color, -1)
                cv2.putText(frame, label, (x1 + 5, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            except Exception as e:
                print(f"Warning: Could not draw detection: {e}")
                continue

        # Status bar at top
        overlay = frame.copy()
        status_color = (0, 165, 255) if self.analyzing else (0, 255, 0)
        status_text = "ANALYZING..." if self.analyzing else "READY"
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(overlay, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Results at bottom
        if self.last_analysis:
            words = self.last_analysis.split(' ')
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + " " + word if current_line else word
                (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                if w < width - 40:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            text_height = len(lines) * 30 + 20
            cv2.rectangle(overlay, (0, height - text_height), (width, height), (0, 0, 0), -1)

            for i, line in enumerate(lines):
                y_pos = height - text_height + 25 + (i * 30)
                cv2.putText(overlay, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame

    def run(self):
        """Main application loop"""
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            display_frame = self.draw_ui(frame.copy())
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n👋 Shutting down...")
                break

            if key == ord('c'):
                print("\n🔄 Switching camera...")
                self.cap.release()
                new_camera = 1 if self.current_camera == 0 else 0
                self.cap = cv2.VideoCapture(new_camera)
                time.sleep(0.5)
                ret, test = self.cap.read()
                if ret and test is not None:
                    self.current_camera = new_camera
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ Switched to camera {new_camera} ({width}x{height})")
                    self.last_analysis = f"Camera {new_camera} ({width}x{height})"
                else:
                    print(f"Camera {new_camera} not available")
                    self.cap = cv2.VideoCapture(self.current_camera)

            # Capture on SPACEBAR
            if key == 32 and not self.analyzing:
                print("\n📸 Capturing frame...")
                self.analyzing = True

                timestamp = int(time.time())
                filename = os.path.join(CAPTURES_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")

                def analyze_thread():
                    analysis = self.analyze_seeds(frame)
                    self.last_analysis = analysis
                    self.analyzing = False

                    print("\n" + "="*60)
                    print("RESULT:", analysis)
                    print("="*60 + "\n")

                thread = threading.Thread(target=analyze_thread)
                thread.daemon = True
                thread.start()

        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Done!")

def main():
    try:
        sifter = SeedSifter()
        sifter.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Make sure camera permissions are enabled")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
