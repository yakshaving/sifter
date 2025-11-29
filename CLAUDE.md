# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Seed Sifter is an educational computer vision seed identification system that uses Mac webcam + OpenCV color detection for real-time classification of seeds (pumpkin vs sunflower). Built with Python and OpenCV for STEM education and archaeology simulations. Fast, lightweight, and works completely offline.

## Development Commands

### Setup & Installation
```bash
# Create and activate virtual environment (ALWAYS required before running)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (lightweight - only OpenCV and NumPy)
pip install -r requirements.txt

# No model downloads needed - uses OpenCV color detection!
```

### Running the Application
```bash
# Option 1: Use the launcher script (recommended)
./run.sh

# Option 2: Manual run
source venv/bin/activate
python sifter_simple.py

# Test camera access (if having issues)
python test_camera.py
```

### Controls
- **SPACEBAR**: Capture and analyze current frame
- **c**: Switch camera (toggle between 0 and 1)
- **q**: Quit application

### Testing & Validation
```bash
# Verify camera access
python test_camera.py

# Manual testing tips:
# 1. Use DARK background for best results
# 2. Good lighting is important
# 3. Spread seeds out (avoid overlapping)
# 4. Press SPACEBAR to capture and analyze
```

## Architecture & Code Structure

### Core Components
- **sifter_simple.py**: Main application - live camera feed with OpenCV color detection
  - SeedSifter class handles camera, detection algorithm, UI rendering, and capture logic
  - Uses HSV color space for seed type differentiation
  - Real-time performance (~instant detection, no AI model latency)
  - Includes retry logic for camera initialization
  - Threaded analysis to keep UI responsive

### Key Technical Decisions
1. **Color-Based Detection**: Uses HSV thresholds instead of AI for speed and simplicity
2. **Real-Time Processing**: Instant feedback (< 100ms) vs 3-5 sec with AI models
3. **Background Dependency**: Works best with dark backgrounds for sunflower seeds
4. **No External Models**: Self-contained, no downloads or internet required
5. **UI Pattern**: OpenCV for both camera capture and UI overlay (no separate GUI framework)

### Detection Algorithm
- **Pumpkin Seeds**: HSV range `(20, 40, 40)` to `(90, 255, 255)` - detects green/yellow high-saturation
- **Sunflower Seeds**: HSV range `(5, 30, 80)` to `(25, 120, 200)` - detects tan/beige medium-saturation
- **Filtering**: Morphological operations (erosion/dilation) to reduce noise
- **Validation**: Area constraints (300-40000 pixels) and aspect ratio checks (0.25-4.0)
- **Performance**: Processes full frames in real-time

### Development Phases
- **Phase 1** (Current): OpenCV color detection with counting and bounding boxes
- **Phase 2** (Optional): AI-enhanced classification with Moondream (test_moondream.py exists)
- **Phase 3** (Planned): Advanced segmentation with SAM2 or similar

## Important Implementation Notes

### Camera Setup
- Terminal needs camera access in System Settings → Privacy & Security → Camera
- Camera conflicts can occur with Zoom/FaceTime - close other camera apps
- Uses retry logic (5 attempts with 2-second delays) for robust initialization
- Supports camera switching with 'c' key (toggles between index 0 and 1)

### Detection Best Practices
- **Background**: Use DARK background for best sunflower seed detection
- **Lighting**: Good, even lighting improves accuracy
- **Seed Placement**: Spread seeds out to avoid overlapping
- **Surface**: Contrasting surface helps (dark fabric, black paper)

### Performance Characteristics
- **Detection Speed**: Near-instant (< 100ms per frame)
- **Memory Usage**: Minimal (~100MB vs 6GB for AI models)
- **Accuracy**: Color-based (good for distinct seed types)
- **Threading**: Analysis runs in background thread to keep UI responsive

### Error Handling Focus
- Camera permission errors → Guide to System Settings
- Camera initialization failures → Automatic retry with feedback
- Empty detections → Suggest background/lighting adjustments

## File Purposes
- **sifter_simple.py**: Main OpenCV-based detection application (active)
- **sifter_simple_commented.py**: Heavily commented version for learning
- **test_camera.py**: Camera validation utility
- **test_moondream.py**: AI model testing (optional alternative approach)
- **run.sh**: Launcher script with environment setup
- **requirements.txt**: Minimal dependencies (opencv-python, numpy)
- **captures/**: Auto-created directory for saved analysis images
- **sample_seeds.jpg**: Test image for validation
- **CLAUDE.md**: This file - guidance for Claude Code
- **codebase_analysis.md**: Comprehensive project analysis

## Alternative Implementations
- **Moondream AI** (test_moondream.py): Slower but more flexible, uses natural language
  - Requires 4GB model download
  - 3-5 second inference time
  - More accurate descriptions but overkill for simple counting
- **Current OpenCV** (sifter_simple.py): Fast, lightweight, real-time
  - No downloads required
  - Instant results
  - Best for educational demonstrations