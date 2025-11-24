#!/usr/bin/env python3
"""
Seed Sifter - Phase 1: Simple Description Mode

Hardware: Mac webcam only
Controls:
    SPACEBAR - Capture and analyze current frame
    q - Quit

This version uses Moondream to describe what it sees.
Focus: Differentiating pumpkin seeds vs sunflower seeds.
"""

import os
# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
# Try to use MPS (Mac GPU) if available, otherwise CPU
if torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Using MPS (Mac GPU) for acceleration")
else:
    device = "cpu"
    print("💻 Using CPU (MPS not available)")

import cv2
import time
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading

# Configuration
WINDOW_NAME = "Seed Sifter - Phase 1 (Simple Mode)"
MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2025-01-09"
CAPTURES_DIR = "captures"
CAMERA_INDEX = 1  # 0 = built-in, 1 = external USB camera (Obsbot 4K)

# Create captures directory if it doesn't exist
os.makedirs(CAPTURES_DIR, exist_ok=True)

class SeedSifter:
    def __init__(self):
        print("🌙 Initializing Seed Sifter...")
        print("🔄 Loading Moondream model...")
        print("   (First run: downloads ~4GB, then cached for offline use)")

        # Load Moondream model (use GPU if available)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=MODEL_REVISION
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        print("✅ Moondream loaded!")

        # Initialize camera with validation
        print(f"🎥 Attempting to open camera {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera {CAMERA_INDEX}!")

        # Test if we can actually grab frames
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(f"❌ Camera {CAMERA_INDEX} opened but cannot grab frames! Try changing CAMERA_INDEX in sifter_simple.py")

        # Get camera resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera {CAMERA_INDEX} initialized! ({width}x{height})")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACEBAR - Capture and analyze seeds")
        print("  q - Quit")
        print("="*60 + "\n")

        # State
        self.last_analysis = "Press SPACEBAR to analyze seeds..."
        self.last_detections = []  # Store bounding boxes
        self.analyzing = False

    def analyze_seeds(self, frame):
        """
        Analyze a frame using Moondream with bounding box detection.
        Detects seeds and returns both boxes and counts.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Run Moondream inference with bounding box detection
        try:
            # Detect seeds in the image
            result = self.model.detect(pil_image, "seeds")
            detections = result.get("objects", [])

            # Store detections for drawing
            self.last_detections = detections

            # Count seeds
            count = len(detections)

            # Create summary message
            if count == 0:
                return "No seeds detected"
            elif count == 1:
                return "1 seed detected"
            else:
                return f"{count} seeds detected"

        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def draw_ui(self, frame):
        """Draw UI overlay on frame with bounding boxes"""
        height, width = frame.shape[:2]

        # Draw bounding boxes for detected seeds
        for detection in self.last_detections:
            # Moondream returns normalized coordinates (0-1)
            # Format: [x_center, y_center, width, height]
            if len(detection) >= 4:
                x_norm, y_norm, w_norm, h_norm = detection[:4]

                # Convert to pixel coordinates
                box_w = int(w_norm * width)
                box_h = int(h_norm * height)
                x1 = int(x_norm * width - box_w / 2)
                y1 = int(y_norm * height - box_h / 2)
                x2 = x1 + box_w
                y2 = y1 + box_h

                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw small center point
                center_x = int(x_norm * width)
                center_y = int(y_norm * height)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Create semi-transparent overlay for text background
        overlay = frame.copy()

        # Status bar at top
        status_color = (0, 165, 255) if self.analyzing else (0, 255, 0)
        status_text = "ANALYZING..." if self.analyzing else "READY"
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(overlay, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Analysis result at bottom
        if self.last_analysis:
            # Word wrap the analysis text
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

            # Draw background box for text
            text_height = len(lines) * 30 + 20
            cv2.rectangle(overlay, (0, height - text_height), (width, height), (0, 0, 0), -1)

            # Draw text lines
            for i, line in enumerate(lines):
                y_pos = height - text_height + 25 + (i * 30)
                cv2.putText(overlay, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        return frame

    def run(self):
        """Main application loop"""
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("❌ Failed to grab frame")
                break

            # Draw UI
            display_frame = self.draw_ui(frame.copy())

            # Show frame
            cv2.imshow(WINDOW_NAME, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Quit on 'q'
            if key == ord('q'):
                print("\n👋 Shutting down...")
                break

            # Capture and analyze on SPACEBAR
            if key == 32 and not self.analyzing:  # SPACEBAR (only if not already analyzing)
                print("\n📸 Capturing frame...")
                self.analyzing = True

                # Save captured image
                timestamp = int(time.time())
                filename = os.path.join(CAPTURES_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")

                # Terminal output
                print("🔍 Analyzing with Moondream...")

                # Analyze in background thread to keep UI responsive
                def analyze_thread():
                    analysis = self.analyze_seeds(frame)
                    self.last_analysis = analysis
                    self.analyzing = False

                    # Print to terminal
                    print("\n" + "="*60)
                    print("ANALYSIS RESULT:")
                    print("="*60)
                    print(analysis)
                    print("="*60 + "\n")

                thread = threading.Thread(target=analyze_thread)
                thread.daemon = True
                thread.start()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Sifter closed successfully!")

def main():
    try:
        sifter = SeedSifter()
        sifter.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Make sure:")
        print("   1. Internet connection on FIRST run (to download model)")
        print("   2. Camera permissions are enabled")
        print("   3. All dependencies are installed: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
