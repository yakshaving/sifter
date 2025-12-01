# 🌻 Seed Sifter

**An offline AI-powered seed identification and counting system for educational archaeology simulations.**

Uses your Mac's webcam + Moondream vision AI to differentiate between seed types (pumpkin vs sunflower), count them, and eventually draw bounding boxes around each one.

Perfect for:
- Teaching kids about archaeology and classification
- Hands-on STEM activities
- Computer vision learning projects
- Future integration with Arduino physical controls

---

## 🎯 Current Status: Phase 3 (AI-Powered Counting)

**What works now:**
- ✅ Live webcam feed with preview
- ✅ Press spacebar to capture and analyze
- ✅ Moondream AI counts seeds accurately (no wood grain false positives!)
- ✅ Works 100% offline (after initial model download)
- ✅ Differentiates pumpkin seeds (green) vs sunflower seeds (tan/beige)
- ✅ Single-command workflow: capture + analyze
- ✅ Real-time OpenCV detection (fast but less accurate)
- ✅ Watershed segmentation for overlapping seeds

**Implementation Highlights:**
- Separate capture/analysis scripts to avoid PyTorch/OpenCV threading conflicts
- Three detection modes: OpenCV (fast), Watershed (separates touching seeds), Moondream (most accurate)
- Automatic count parsing and ratio calculations
- Saves all captures for later re-analysis

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- **Mac** (M1/M2/M3 or Intel)
- **Python 3.8+**
- **Webcam** (built-in Mac camera works great)

### 2️⃣ Setup (5-10 minutes)

See [setup_instructions.md](setup_instructions.md) for detailed walkthrough.

**TL;DR:**
```bash
# Clone and setup
git clone git@github.com:yakshaving/sifter.git
cd sifter
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Moondream model (one-time, ~4GB)
moondream download moondream-2b

# Test everything works
python test_camera.py
python test_moondream.py

# Launch the sifter!
python sifter_simple.py
```

### 3️⃣ Usage

1. Run `python sifter_simple.py`
2. Position seeds in front of camera (or show image on phone)
3. Press **SPACEBAR** to capture and analyze
4. See results in **both**:
   - Video window overlay
   - Terminal output
5. Press **q** to quit

---

## 📸 What to Test With

**Ideal subjects:**
- Pumpkin seeds (large, white, oval)
- Sunflower seeds (small, striped black/white)
- Mixed groups for differentiation testing

**Tips for best results:**
- Plain background (white paper or light surface)
- Good lighting
- Seeds spread out (not overlapping)
- Camera positioned directly above

---

## 🗂️ Project Structure

```
sifter/
├── sifter_simple.py         # Phase 1: Main app (use this!)
├── sifter_counter.py         # Phase 2: Counting mode (stub)
├── sifter_bbox.py            # Phase 3: Bounding boxes (stub)
├── test_camera.py            # Test webcam
├── test_moondream.py         # Test Moondream offline
├── requirements.txt          # Python dependencies
├── setup_instructions.md     # Detailed setup guide
└── captures/                 # Auto-created for saved images
```

---

## 🔧 Troubleshooting

### Camera not opening?
- Grant Terminal camera permissions: **System Settings → Privacy & Security → Camera**
- Close other apps using the camera (Zoom, etc.)

### Moondream errors?
- Make sure you downloaded the model: `moondream download moondream-2b`
- Check the `moondream-2b/` directory exists
- Try disconnecting Wi-Fi to verify offline mode

### Slow analysis?
- The 2B model takes 3-5 seconds per image (normal on Mac)
- For faster inference, consider switching to 1.8B model
- M1/M2/M3 Macs perform significantly better than Intel

---

## 🎓 Educational Use

This project teaches:
- **Computer vision basics** - How AI "sees" objects
- **Classification** - Differentiating similar objects
- **Data collection** - Systematic capture and analysis
- **Scientific method** - Hypothesis → Test → Results

**Classroom activity ideas:**
1. "Seed archaeologist" - Find and classify mixed seeds
2. "Accuracy challenge" - Compare AI count vs human count
3. "Pattern recognition" - What features help AI identify seeds?

---

## 🔮 Future Enhancements

**Hardware integration:**
- Arduino button trigger (replace spacebar)
- LEDs for feedback
- Buzzer for rare finds
- Motion sensor for automatic capture

**Software upgrades:**
- Real-time object detection (YOLO)
- Custom seed dataset training
- Multi-class support (add more seed types)
- Leaderboard and scoring system

---

## 📄 License

MIT License - Feel free to use for educational purposes!

---

## 🙋 Questions?

Open an issue or check [setup_instructions.md](setup_instructions.md) for detailed help.

**Happy sifting!** 🌻
