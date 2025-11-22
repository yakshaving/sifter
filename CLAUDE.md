# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Seed Sifter is an educational AI-powered seed identification system that uses Mac webcam + Moondream vision AI for offline classification of seeds (pumpkin vs sunflower). Built with Python, OpenCV, and Hugging Face Transformers for STEM education and archaeology simulations.

## Development Commands

### Setup & Installation
```bash
# Create and activate virtual environment (ALWAYS required before running)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Moondream model (4GB, one-time, then works offline)
# Uses Hugging Face transformers - no separate moondream CLI tool needed
```

### Running the Application
```bash
# IMPORTANT: Always activate venv first
source venv/bin/activate

# Test camera access
python test_camera.py

# Test Moondream model (works offline after initial download)
python test_moondream.py

# Run main application (Phase 1 - Simple Mode)
python sifter_simple.py
```

### Testing & Validation
```bash
# No formal test suite yet - use these for validation:
python test_camera.py      # Verify camera permissions and access
python test_moondream.py   # Test AI model inference (offline)

# Manual testing: Run main app and press SPACEBAR to capture/analyze
```

## Architecture & Code Structure

### Core Components
- **sifter_simple.py**: Main application (Phase 1) - live camera feed with Moondream analysis
  - SeedSifter class handles camera, model loading, UI rendering, and capture logic
  - Uses Hugging Face transformers directly (no separate moondream package)
  - Forces CPU-only inference to avoid MPS device conflicts on Mac

### Key Technical Decisions
1. **Offline-First**: Downloads Moondream2 model from Hugging Face on first run, then works completely offline
2. **CPU-Only Inference**: Disabled MPS/GPU to avoid PyTorch device conflicts on Mac
3. **Model Loading**: Uses AutoModelForCausalLM from transformers with trust_remote_code=True
4. **UI Pattern**: OpenCV for both camera capture and UI overlay (no separate GUI framework)

### Model Configuration
- Model ID: "vikhyatk/moondream2"
- Revision: "2025-01-09"
- Device: CPU only (MPS disabled via environment variables)
- Inference time: 3-5 seconds per image on typical Mac

### Development Phases
- **Phase 1** (Current): Simple description mode - natural language output
- **Phase 2** (Stub): Counting mode - parse counts from descriptions
- **Phase 3** (Planned): Bounding boxes - visual detection overlays

## Important Implementation Notes

### Environment Setup Requirements
- MUST set `PYTORCH_ENABLE_MPS_FALLBACK="1"` before importing torch
- MUST disable MPS backend: `torch.backends.mps.is_available = lambda: False`
- These are critical to prevent device allocation errors on Mac

### Camera Permissions
- Terminal needs camera access in System Settings → Privacy & Security → Camera
- Camera conflicts can occur with Zoom/FaceTime - close other camera apps

### Model Loading Pattern
```python
# Critical order of operations:
1. Set environment variables BEFORE any imports
2. Import torch and disable MPS
3. Load model with .to("cpu") explicitly
4. Use trust_remote_code=True for Moondream
```

### Error Handling Focus
- Camera permission errors → Guide to System Settings
- Model download failures → Check internet on first run
- Slow inference → Expected (3-5 sec), suggest M1/M2/M3 for better performance

## File Purposes
- **requirements.txt**: Core dependencies (transformers, opencv-python, torch, etc.)
- **captures/**: Auto-created directory for saved analysis images
- **sample_seeds.jpg**: Test image for model validation without camera
- **test_*.py**: Validation scripts for camera and model components