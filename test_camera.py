#!/usr/bin/env python3
"""
Test script to verify your Mac webcam is working.
Press 'q' to quit.
"""

import cv2

def test_camera():
    print("🎥 Testing webcam...")
    print("Press 'q' to quit")

    # Open default camera (usually 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        print("   Make sure:")
        print("   1. Camera permissions are enabled for Terminal")
        print("   2. No other app is using the camera")
        return

    print("✅ Camera opened successfully!")
    print("   You should see a video window...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame")
            break

        # Add text overlay
        cv2.putText(frame, "Camera Test - Press 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow('Camera Test', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test complete!")

if __name__ == "__main__":
    test_camera()
