# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Seed Sifter is an educational computer vision seed identification system that uses Mac webcam + AI vision model (Moondream) for accurate classification of seeds (pumpkin vs sunflower). Built with Python, OpenCV for camera capture, and Moondream AI for semantic understanding. Works completely offline after initial model download (~4GB).

**Key Innovation**: Separates camera capture (OpenCV) from AI analysis (Moondream/PyTorch) to avoid threading conflicts while providing accurate counts that ignore wood grain false positives.

## Development Commands

### Setup & Installation
```bash
# Create and activate virtual environment (ALWAYS required before running)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes transformers, torch, opencv)
pip install -r requirements.txt

# Download Moondream model (one-time, ~4GB, then cached)
moondream download moondream-2b
```

### Running the Application

**Unified Interface**: Interactive menu with all modes:
```bash
python sifter.py                    # Interactive menu
python sifter.py --camera           # Live camera mode
python sifter.py --image <path>     # Analyze image file
python sifter.py --folder <path>    # Analyze folder of images
python sifter.py --ai               # AI analysis with Moondream
```

**RECOMMENDED**: Use count_seeds.py for accurate AI-based counting:
```bash
# Interactive mode - shows live preview, press SPACEBAR to capture & analyze
source venv/bin/activate
python count_seeds.py

# Auto mode - captures after 3 seconds, no interaction needed
python count_seeds.py --auto
```

**Alternative**: Use sifter_simple.py for real-time OpenCV color detection (fast but less accurate on wood backgrounds):
```bash
# Option 1: Use the launcher script
./run.sh

# Option 2: Manual run
source venv/bin/activate
python sifter_simple.py
```

**Post-Analysis**: Analyze previously captured images:
```bash
# Analyze most recent capture
python analyze_capture.py

# Analyze specific capture
python analyze_capture.py captures/capture_1234567890.jpg

# Batch analyze all captures without GUI
python batch_analyze.py
```

### Controls (count_seeds.py)
- **SPACEBAR**: Capture image and start AI analysis
- **q**: Quit without capturing

### Controls (sifter_simple.py)
- **SPACEBAR**: Capture and analyze current frame
- **m**: Toggle detection mode (Watershed / OpenCV)
- **c**: Switch camera (toggle between 0 and 1)
- **q**: Quit application

### Testing & Validation
```bash
# Verify camera access
python test_camera.py

# Test Moondream model installation
python test_moondream.py

# Test OpenCV counting with sample image
python test_opencv_count.py

# Manual testing tips:
# 1. Use DARK background for best results
# 2. Good lighting is important
# 3. Spread seeds out (avoid overlapping)
# 4. Press SPACEBAR to capture and analyze
```

## Architecture & Code Structure

### Three-Tier Architecture
1. **Unified Entry Point** (`sifter.py`): Interactive menu system for all detection modes
2. **Real-Time Detection** (`sifter_simple.py`): OpenCV/Watershed for live camera feed
3. **AI Analysis** (`count_seeds.py`, `analyze_capture.py`): Moondream for accurate counting

### Core Components
- **sifter.py**: Unified entry point with interactive menu and command-line arguments
  - Provides access to all detection modes through single interface
  - Supports camera, image file, folder, and AI analysis modes

- **sifter_simple.py**: Main application - live camera feed with OpenCV color detection
  - SeedSifter class handles camera, detection algorithm, UI rendering, and capture logic
  - Uses HSV color space for seed type differentiation
  - Real-time performance (~instant detection, no AI model latency)
  - Includes retry logic for camera initialization
  - Threaded analysis to keep UI responsive

- **count_seeds.py**: AI-powered capture and analysis workflow
  - Captures image from webcam, saves to captures/, then analyzes with Moondream
  - Separates GUI operations from AI analysis to avoid threading conflicts
  - Supports both interactive (SPACEBAR) and auto-capture (--auto) modes

- **batch_analyze.py**: Headless batch processing for multiple images
  - Processes all images in captures/ directory without GUI
  - Generates annotated versions with bounding boxes
  - Outputs summary statistics

### Key Technical Decisions
1. **Color-Based Detection**: Uses HSV thresholds instead of AI for speed and simplicity
2. **Real-Time Processing**: Instant feedback (< 100ms) vs 3-5 sec with AI models
3. **Background Dependency**: Works best with dark backgrounds for sunflower seeds
4. **No External Models**: Self-contained, no downloads or internet required
5. **UI Pattern**: OpenCV for both camera capture and UI overlay (no separate GUI framework)

### Detection Algorithms

The application supports three detection modes (cycle with 'M' key):

#### 1. OpenCV Color Detection Mode (Default)
**Traditional approach: Direct color-based detection**
- **Pumpkin Seeds**: HSV range `(20, 40, 40)` to `(90, 255, 255)` - detects green/yellow high-saturation
- **Sunflower Seeds**: HSV range `(6, 35, 85)` to `(22, 110, 190)` - detects tan/beige
- **Filtering**: Morphological operations (erosion/dilation) to reduce noise
- **Validation**:
  - Area constraints: 400-40000 pixels
  - Aspect ratio: 0.25-4.0
  - Solidity filter: > 0.65 (area/convex_hull_area) to reject irregular shapes
- **Advantages**: Fastest, real-time feedback, shows bounding boxes
- **Best for**: Well-separated seeds, quick preview
- **Limitations**: Struggles with overlapping seeds, wood grain false positives
- **Performance**: < 100ms per frame

#### 2. Watershed Segmentation Mode
**Hybrid approach: Instance segmentation + color classification**
- Uses distance transform to locate seed centers
- Applies watershed algorithm to separate touching/overlapping seeds
- Classifies each separated segment by dominant color (pumpkin vs sunflower)
- **Advantages**: Better at separating clustered seeds
- **Best for**: Dense seed arrangements, overlapping seeds
- **Limitations**: May still have false positives from wood grain
- **Performance**: ~100-200ms per frame

#### 3. Moondream AI Mode (Most Accurate - Separate Script)
**AI vision model: Semantic understanding via count_seeds.py**
- Uses Moondream vision-language model for seed counting
- Asks AI to count and identify seeds in natural language
- Model auto-downloads on first use (~4GB, then cached locally)
- Works completely offline after initial download
- **Advantages**: Most accurate, semantic understanding, no false positives from wood grain
- **Best for**: Final accurate counts, complex seed arrangements, wood backgrounds
- **Limitations**: Slower (3-5 seconds), requires separate capture step, requires model download
- **Performance**: 3-5 seconds for analysis (after capture)
- **Note**: First use requires internet for model download; subsequent uses work offline
- **Architecture**: Separate from sifter_simple.py to avoid PyTorch/OpenCV threading conflicts

### Development Phases
- **Phase 1** (Complete): OpenCV color detection with counting and bounding boxes
- **Phase 2** (Complete): Watershed segmentation for overlapping seeds
- **Phase 3** (Current): Moondream AI for accurate semantic counting via count_seeds.py
- **Future**: Advanced segmentation with SAM2 or similar (if needed)

## Important Implementation Notes

### Recent Detection Improvements (2025-11-29)

#### Phase 1: Color Detection Tuning
Fixed over-counting issue where wood grain and background textures caused 70+ false positives:

**The Problem**: Initial testing showed 106 detections when only ~15-20 seeds were present. Wood grain texture was being detected as sunflower seeds due to similar color ranges.

**Tuning Process**:
- **Attempt 1** (too strict): min_area=1000, HSV `(8, 50, 100)-(20, 100, 180)`, solidity > 0.7
  - Result: Under-detection, no sunflower seeds found
- **Attempt 2** (still under-counting): min_area=600, HSV `(6, 35, 85)-(22, 110, 190)`, solidity > 0.6
  - Result: Only 15 pumpkin seeds detected when 40-60 visible, missing most individual seeds
- **Attempt 3** (balanced but limited): min_area=400, solidity > 0.65
  - Result: Better detection but still struggled with overlapping seeds and wood grain

**Core Challenge**: Pure color-based detection cannot reliably separate overlapping seeds or distinguish wood grain from tan seeds.

#### Phase 2: Watershed Hybrid Approach
**Solution**: Added watershed segmentation mode to solve overlapping seed and false positive issues:

**How It Works**:
1. Creates combined mask of all seed-colored regions (both pumpkin and sunflower)
2. Uses distance transform to find individual seed centers
3. Applies watershed algorithm to separate touching/overlapping seeds into distinct segments
4. Classifies each segment by dominant color (pumpkin vs sunflower)

**Result**: Improved separation of touching seeds but still had wood grain issues.

#### Phase 3: Moondream AI Solution (Current Recommended Approach)
**Solution**: Created separate capture-then-analyze workflow to avoid PyTorch/OpenCV threading conflicts:

**Why Moondream**:
- Color-based methods (OpenCV and Watershed) cannot distinguish wood grain from tan seeds
- AI can semantically understand what a seed looks like vs wood texture
- Natural language approach: asks AI to count seeds directly
- Works offline after initial model download (~4GB)

**Architecture Decision - Separation of Concerns**:
Initial attempt to integrate Moondream into sifter_simple.py caused segmentation faults (exit code 139). Root cause: PyTorch creates background threads that conflict with OpenCV's `cv2.waitKey()` GUI event loop.

**Solution**: Separate scripts:
1. **count_seeds.py** (recommended): Captures with OpenCV, closes GUI, then analyzes with Moondream
2. **analyze_capture.py**: Analyzes previously saved captures with Moondream
3. **sifter_simple.py**: Real-time OpenCV/Watershed detection (fast but less accurate)

**How count_seeds.py Works**:
```
1. Opens camera with OpenCV
2. Shows live preview, waits for SPACEBAR
3. Saves image to captures/
4. Closes OpenCV completely (frees GUI thread)
5. Loads Moondream model on CPU (no MPS/GPU conflicts)
6. Analyzes saved image with AI
7. Parses and displays accurate counts
```

**Result**: Accurate AI-based counting (20-30 seeds) vs color detection (100+ false positives from wood grain).

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
- **sifter.py**: Unified entry point with interactive menu
- **sifter_simple.py**: Main OpenCV-based detection application (active)
- **sifter_simple_commented.py**: Heavily commented version for learning
- **count_seeds.py**: AI-powered capture and analysis workflow
- **analyze_capture.py**: Analyze previously saved captures with Moondream
- **batch_analyze.py**: Batch process all captures without GUI
- **test_camera.py**: Camera validation utility
- **test_moondream.py**: AI model testing
- **test_opencv_count.py**: OpenCV counting validation
- **run.sh**: Launcher script with environment setup
- **requirements.txt**: Python dependencies
- **captures/**: Auto-created directory for saved analysis images
- **sample_seeds.jpg**: Test image for validation
- **CLAUDE.md**: This file - guidance for Claude Code
- **README.md**: User-facing documentation
- **USAGE.md**: Quick start guide with examples

## Alternative Implementations
- **Moondream AI** (test_moondream.py): Slower but more flexible, uses natural language
  - Requires 4GB model download
  - 3-5 second inference time
  - More accurate descriptions but overkill for simple counting
- **Current OpenCV** (sifter_simple.py): Fast, lightweight, real-time
  - No downloads required
  - Instant results
  - Best for educational demonstrations