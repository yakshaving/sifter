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
        min_area = 300
        max_area = 40000
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # --- PUMPKIN SEEDS: Green/yellow, high saturation ---
        # This works on any background
        mask_pumpkin = cv2.inRange(hsv, (20, 40, 40), (90, 255, 255))
        mask_pumpkin = cv2.erode(mask_pumpkin, kernel, iterations=2)
        mask_pumpkin = cv2.dilate(mask_pumpkin, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_pumpkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                ar = w / h if h > 0 else 0
                if 0.25 < ar < 4.0:
                    detections.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area, 'type': 'pumpkin', 'color': (0, 255, 0)
                    })

        # --- SUNFLOWER SEEDS: Tan/beige ---
        # Only works well on dark background; on white counter they blend in
        # Detect by: low-medium saturation, medium value, warm hue
        mask_sunflower = cv2.inRange(hsv, (5, 30, 80), (25, 120, 200))
        mask_sunflower = cv2.erode(mask_sunflower, kernel, iterations=1)
        mask_sunflower = cv2.dilate(mask_sunflower, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_sunflower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                ar = w / h if h > 0 else 0
                if 0.25 < ar < 4.0:
                    # Make sure we're not double-counting (check overlap with pumpkin)
                    cx, cy = x + w//2, y + h//2
                    is_overlap = False
                    for d in detections:
                        if (d['x'] < cx < d['x']+d['w'] and d['y'] < cy < d['y']+d['h']):
                            is_overlap = True
                            break
                    if not is_overlap:
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

        # Draw bounding boxes
        for detection in self.last_detections:
            try:
                if not isinstance(detection, dict):
                    continue

                x_min = detection.get('x_min', 0)
                y_min = detection.get('y_min', 0)
                x_max = detection.get('x_max', 0)
                y_max = detection.get('y_max', 0)
                color = detection.get('color', (0, 255, 0))

                x1 = int(x_min * width)
                y1 = int(y_min * height)
                x2 = int(x_max * width)
                y2 = int(y_max * height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
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
