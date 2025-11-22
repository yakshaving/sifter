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

import cv2
import time
import os
from PIL import Image
from moondream import VisionModel

# Configuration
WINDOW_NAME = "Seed Sifter - Phase 1 (Simple Mode)"
MODEL_PATH = "moondream-2b"
CAPTURES_DIR = "captures"

# Create captures directory if it doesn't exist
os.makedirs(CAPTURES_DIR, exist_ok=True)

class SeedSifter:
    def __init__(self):
        print("🌙 Initializing Seed Sifter...")
        print("🔄 Loading Moondream model (this may take 10-20 seconds)...")

        # Load Moondream model
        self.model = VisionModel(local_model_path=MODEL_PATH)
        print("✅ Moondream loaded!")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open camera!")

        print("✅ Camera initialized!")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACEBAR - Capture and analyze seeds")
        print("  q - Quit")
        print("="*60 + "\n")

        # State
        self.last_analysis = "Press SPACEBAR to analyze seeds..."
        self.analyzing = False

    def analyze_seeds(self, frame):
        """
        Analyze a frame using Moondream.
        Optimized prompt for seed counting and differentiation.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Craft a specific prompt for seed analysis
        prompt = (
            "Describe the seeds in this image. "
            "Specifically identify if you see pumpkin seeds (large, white, oval) "
            "or sunflower seeds (smaller, striped black and white). "
            "Estimate the count of each type if possible."
        )

        # Run Moondream inference
        try:
            description = self.model.describe(pil_image)
            return description
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def draw_ui(self, frame):
        """Draw UI overlay on frame"""
        height, width = frame.shape[:2]

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
            if key == 32:  # SPACEBAR
                print("\n📸 Capturing frame...")
                self.analyzing = True

                # Save captured image
                timestamp = int(time.time())
                filename = os.path.join(CAPTURES_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")

                # Terminal output
                print("🔍 Analyzing with Moondream...")

                # Analyze
                analysis = self.analyze_seeds(frame)
                self.last_analysis = analysis
                self.analyzing = False

                # Print to terminal
                print("\n" + "="*60)
                print("ANALYSIS RESULT:")
                print("="*60)
                print(analysis)
                print("="*60 + "\n")

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
        print("   1. Moondream model is downloaded: moondream download moondream-2b")
        print("   2. Camera permissions are enabled")
        print("   3. All dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
