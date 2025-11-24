#!/usr/bin/env python3
"""
=======================================================
SEED SIFTER - AI-Powered Seed Detector for Kids!
=======================================================

What does this program do?
- Uses a camera to look at seeds
- Uses AI to find and count seeds automatically
- Draws green boxes around each seed it finds
- Works completely offline (no internet needed after setup)

How it works:
1. Opens your camera and shows live video
2. When you press SPACEBAR, it takes a picture
3. The AI looks at the picture and finds all the seeds
4. It draws boxes around the seeds and counts them

Controls:
- SPACEBAR: Take a picture and analyze it
- C: Switch between cameras
- Q: Quit the program

Educational Purpose:
This teaches kids about:
- Computer vision (how computers "see")
- Artificial intelligence
- Object detection
- Archaeology (sorting artifacts)
"""

# ============================================
# STEP 1: Import the Tools We Need
# ============================================

import os  # For working with files and folders
import time  # For tracking time and adding delays
import threading  # For running multiple tasks at once

# Tell the computer to use the Mac's GPU if it can (makes AI faster)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # PyTorch - the AI engine
import cv2  # OpenCV - for camera and drawing on images
from PIL import Image  # For working with images
from transformers import AutoModelForCausalLM, AutoTokenizer  # For loading the AI brain

# ============================================
# STEP 2: Check if We Can Use the GPU
# ============================================
# The GPU (Graphics Processing Unit) is like a super-fast brain for AI
# If we have one, use it! If not, use the regular CPU (slower but still works)

if torch.backends.mps.is_available():
    device = "mps"  # MPS = Metal Performance Shaders (Mac's GPU)
    print("🚀 Using Mac GPU for super-fast AI!")
else:
    device = "cpu"
    print("💻 Using CPU (no GPU found, will be slower)")

# ============================================
# STEP 3: Settings and Configuration
# ============================================

WINDOW_NAME = "Seed Sifter - AI Seed Detector"
MODEL_ID = "vikhyatk/moondream2"  # The AI model we're using
MODEL_REVISION = "2025-01-09"  # Version of the AI
CAPTURES_DIR = "captures"  # Folder where we save pictures
CAMERA_INDEX = 1  # Which camera to use (0=built-in, 1=external)

# Create the captures folder if it doesn't exist yet
os.makedirs(CAPTURES_DIR, exist_ok=True)


# ============================================
# STEP 4: The Main Seed Sifter Class
# ============================================
# A "class" is like a blueprint for building something
# This class contains all the code for our Seed Sifter

class SeedSifter:
    """
    The SeedSifter class is our AI seed detector!

    It has three main jobs:
    1. Load the AI "brain" (Moondream model)
    2. Open the camera and show live video
    3. Analyze pictures to find and count seeds
    """

    def __init__(self):
        """
        __init__ means "initialize" - this runs when we first start the program.
        It loads the AI brain and opens the camera.
        """

        print("🌙 Starting up Seed Sifter...")
        print("🔄 Loading the AI brain (Moondream)...")
        print("   (First time: downloads 4GB file, then saved forever)")

        # ============================================
        # Load the AI Brain
        # ============================================
        # Moondream is an AI that can "see" and understand images
        # It was trained on millions of pictures to recognize objects

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,  # Which AI model to use
            trust_remote_code=True,  # Allow the AI to run its code
            revision=MODEL_REVISION  # Which version
        ).to(device)  # Put it on the GPU or CPU

        # Load the "tokenizer" - helps the AI understand words
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        print("✅ AI brain loaded and ready!")

        # ============================================
        # Open the Camera with Retry Logic
        # ============================================
        # Sometimes cameras take a moment to wake up
        # So we try multiple times if it doesn't work the first time

        camera_index = CAMERA_INDEX
        print(f"🎥 Opening camera {camera_index}...")

        max_retries = 5  # Try up to 5 times
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"   Retry {attempt}/{max_retries-1}... waiting 2 seconds")
                time.sleep(2)  # Wait 2 seconds between tries

            # Try to open the camera
            self.cap = cv2.VideoCapture(camera_index)

            # Check if it opened successfully
            if not self.cap.isOpened():
                if attempt == max_retries - 1:
                    raise RuntimeError(f"❌ Could not open camera after {max_retries} tries!")
                continue

            # Wait half a second for camera to fully wake up
            time.sleep(0.5)

            # Try to grab a test picture from the camera
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"✅ Camera {camera_index} connected successfully!")
                break
            else:
                self.cap.release()  # Close the camera and try again
                if attempt == max_retries - 1:
                    raise RuntimeError(f"❌ Camera won't give us pictures!")

        if test_frame is None:
            raise RuntimeError(f"❌ Failed to initialize camera")

        # Get the camera's width and height (resolution)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized! ({width}x{height} pixels)")

        # Show the controls to the user
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACEBAR - Take picture and find seeds")
        print("  C - Switch camera (toggle between cameras)")
        print("  Q - Quit the program")
        print("="*60 + "\n")

        # ============================================
        # Set Up Variables to Track State
        # ============================================
        # These variables remember what's happening

        self.last_analysis = "Press SPACEBAR to analyze seeds... Press 'c' to switch camera"
        self.last_detections = []  # List of seed locations we found
        self.analyzing = False  # Are we currently analyzing? (True/False)
        self.current_camera = camera_index  # Which camera are we using?


    def analyze_seeds(self, frame):
        """
        This is the magic function that finds seeds!

        How it works:
        1. Takes a picture (frame) from the camera
        2. Asks the AI: "Where are the seeds in this picture?"
        3. The AI returns a list of locations (x, y coordinates)
        4. We count how many seeds it found

        Kids learn: How AI can "see" objects in images
        """

        # ============================================
        # Convert the Image Format
        # ============================================
        # Cameras use BGR color format (Blue-Green-Red)
        # But the AI understands RGB format (Red-Green-Blue)
        # So we need to convert it

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)  # Convert to PIL Image format

        # ============================================
        # Ask the AI to Find Seeds
        # ============================================
        try:
            # Call the AI's "detect" function
            # We're asking it: "Find all the seeds in this image"
            result = self.model.detect(pil_image, "seeds")

            # The AI returns a dictionary with "objects" containing seed locations
            detections = result.get("objects", [])

            # Save the detections so we can draw boxes later
            self.last_detections = detections

            # Count how many seeds we found
            count = len(detections)

            # Create a message to show the user
            if count == 0:
                return "No seeds detected 😕"
            elif count == 1:
                return "Found 1 seed! ✨"
            else:
                return f"Found {count} seeds! ✨"

        except Exception as e:
            # If something goes wrong, show the error
            return f"Error during analysis: {str(e)}"


    def draw_ui(self, frame):
        """
        This function draws the user interface on top of the video.

        What it draws:
        1. Green boxes around each seed (bounding boxes)
        2. Small green dots at the center of each seed
        3. Status bar at top (READY or ANALYZING)
        4. Text at bottom showing the results

        Kids learn: How computers draw on images
        """

        height, width = frame.shape[:2]  # Get image size

        # ============================================
        # Draw Bounding Boxes Around Seeds
        # ============================================
        # For each seed the AI found, draw a green rectangle around it

        for detection in self.last_detections:
            # The AI gives us normalized coordinates (numbers from 0 to 1)
            # Format: [x_center, y_center, width, height]

            if len(detection) >= 4:
                x_norm, y_norm, w_norm, h_norm = detection[:4]

                # Convert from 0-1 range to actual pixel coordinates
                box_w = int(w_norm * width)  # Box width in pixels
                box_h = int(h_norm * height)  # Box height in pixels
                x1 = int(x_norm * width - box_w / 2)  # Left edge
                y1 = int(y_norm * height - box_h / 2)  # Top edge
                x2 = x1 + box_w  # Right edge
                y2 = y1 + box_h  # Bottom edge

                # Draw a green rectangle (box) around the seed
                # (0, 255, 0) = Green color in BGR format
                # 3 = thickness of the line
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw a small green circle at the center of the seed
                center_x = int(x_norm * width)
                center_y = int(y_norm * height)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # ============================================
        # Draw Status Bar at Top
        # ============================================
        # Create a semi-transparent black bar at the top

        overlay = frame.copy()

        # Choose color based on whether we're analyzing
        if self.analyzing:
            status_color = (0, 165, 255)  # Orange for analyzing
            status_text = "ANALYZING..."
        else:
            status_color = (0, 255, 0)  # Green for ready
            status_text = "READY"

        # Draw black rectangle at top
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)

        # Write the status text
        cv2.putText(overlay, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # ============================================
        # Draw Results at Bottom
        # ============================================
        # Show the analysis results at the bottom of the screen

        if self.last_analysis:
            # Split text into words for word wrapping
            words = self.last_analysis.split(' ')
            lines = []
            current_line = ""

            # Word wrap: fit text to screen width
            for word in words:
                test_line = current_line + " " + word if current_line else word
                (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                if w < width - 40:  # Does it fit?
                    current_line = test_line
                else:  # Too wide, start new line
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            # Draw background box for text
            text_height = len(lines) * 30 + 20
            cv2.rectangle(overlay, (0, height - text_height), (width, height), (0, 0, 0), -1)

            # Draw each line of text
            for i, line in enumerate(lines):
                y_pos = height - text_height + 25 + (i * 30)
                cv2.putText(overlay, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Blend the overlay with the original frame (makes it semi-transparent)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        return frame


    def run(self):
        """
        This is the main loop that runs the program!

        What it does:
        1. Get a picture from the camera (30 times per second)
        2. Draw the user interface on it
        3. Show it on the screen
        4. Listen for keyboard presses
        5. Repeat forever until you quit

        Kids learn: How programs loop and respond to user input
        """

        print("📹 Camera feed is now live! Look for the window...")

        # ============================================
        # The Main Loop - Runs Forever
        # ============================================

        while True:  # Loop forever (until we break out)

            # Get one frame (picture) from the camera
            ret, frame = self.cap.read()

            if not ret:  # If we couldn't get a picture
                print("❌ Camera stopped working!")
                break

            # Draw the user interface on top of the picture
            display_frame = self.draw_ui(frame.copy())

            # Show the picture in a window
            cv2.imshow(WINDOW_NAME, display_frame)

            # ============================================
            # Listen for Keyboard Presses
            # ============================================
            # cv2.waitKey(1) waits 1 millisecond for a key press

            key = cv2.waitKey(1) & 0xFF

            # ========== Q Key: Quit ==========
            if key == ord('q'):
                print("\n👋 Shutting down...")
                break

            # ========== C Key: Switch Camera ==========
            if key == ord('c'):
                print("\n🔄 Switching camera...")
                self.cap.release()  # Close current camera

                # Switch: if using camera 0, switch to 1, and vice versa
                new_camera = 1 if self.current_camera == 0 else 0
                self.cap = cv2.VideoCapture(new_camera)
                time.sleep(0.5)  # Wait for camera to wake up

                # Test if the new camera works
                ret, test = self.cap.read()
                if ret and test is not None:
                    self.current_camera = new_camera
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✅ Switched to camera {new_camera} ({width}x{height})")
                    self.last_analysis = f"Now using camera {new_camera} ({width}x{height})"
                else:
                    print(f"❌ Camera {new_camera} not available")
                    self.cap = cv2.VideoCapture(self.current_camera)  # Go back

            # ========== SPACEBAR: Analyze Seeds ==========
            if key == 32 and not self.analyzing:  # 32 = SPACEBAR code
                print("\n📸 Taking picture...")
                self.analyzing = True  # Mark that we're analyzing

                # Save the picture to the captures folder
                timestamp = int(time.time())  # Use current time as filename
                filename = os.path.join(CAPTURES_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"💾 Saved picture: {filename}")

                print("🔍 Asking AI to find seeds...")

                # ============================================
                # Run Analysis in Background Thread
                # ============================================
                # We use threading so the video doesn't freeze
                # The AI analysis takes 3-5 seconds, so we do it in the background

                def analyze_thread():
                    """
                    This function runs in the background while the video keeps playing
                    """
                    # Ask the AI to analyze the picture
                    analysis = self.analyze_seeds(frame)

                    # Update the results
                    self.last_analysis = analysis
                    self.analyzing = False  # We're done!

                    # Print results to the terminal
                    print("\n" + "="*60)
                    print("🎯 AI RESULTS:")
                    print("="*60)
                    print(analysis)
                    print(f"Total detections: {len(self.last_detections)}")
                    print("="*60 + "\n")

                # Start the background thread
                thread = threading.Thread(target=analyze_thread)
                thread.daemon = True  # Thread will stop when program stops
                thread.start()

        # ============================================
        # Clean Up When Program Ends
        # ============================================

        self.cap.release()  # Close the camera
        cv2.destroyAllWindows()  # Close all windows
        print("✅ Seed Sifter closed successfully!")


# ============================================
# STEP 5: The Main Function
# ============================================
# This is where the program starts when you run it

def main():
    """
    The main function creates a SeedSifter and runs it.
    It also catches errors if something goes wrong.
    """
    try:
        # Create a new SeedSifter object
        sifter = SeedSifter()

        # Run it!
        sifter.run()

    except KeyboardInterrupt:
        # If user presses Ctrl+C, quit gracefully
        print("\n\n👋 Stopped by user (Ctrl+C)")

    except Exception as e:
        # If something goes wrong, show what happened
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure internet is ON for first run (downloads AI)")
        print("2. Make sure camera permissions are enabled")
        print("3. Make sure all files are installed (run: pip install -r requirements.txt)")

        # Show detailed error for debugging
        import traceback
        traceback.print_exc()


# ============================================
# STEP 6: Start the Program!
# ============================================
# This special code runs only when you run this file directly
# (not when you import it into another program)

if __name__ == "__main__":
    main()
