# 🛠️ Seed Sifter - Detailed Setup Instructions

**Follow these steps exactly to get the Seed Sifter running on your Mac.**

Time required: **10-15 minutes** (most of it is downloading the 4GB model)

---

## ✅ Step 1: Verify Prerequisites

Open **Terminal** and check:

```bash
# Check Python version (need 3.8 or higher)
python3 --version
```

Should show something like `Python 3.11.x` or `Python 3.10.x`.

If not installed, download from [python.org](https://www.python.org/downloads/)

---

## ✅ Step 2: Clone the Repository

```bash
# Navigate to where you want the project
cd ~/Documents  # or wherever you prefer

# Clone the repo
git clone git@github.com:yakshaving/sifter.git

# Enter the directory
cd sifter
```

---

## ✅ Step 3: Create Virtual Environment

**Why?** Keeps dependencies isolated from your system Python.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (you'll do this every time you work on the project)
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` at the beginning.

**Important:** Always activate the venv before running the sifter!

---

## ✅ Step 4: Install Python Dependencies

```bash
# Make sure venv is activated (see Step 3)
pip install --upgrade pip  # get latest pip first
pip install -r requirements.txt
```

This installs:
- `moondream` - The vision AI model
- `opencv-python` - Camera and image processing
- `pillow` - Image format handling
- `torch` - Machine learning backend

**Troubleshooting PyTorch on Apple Silicon:**

If you get errors installing `torch`, use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## ✅ Step 5: Download Moondream Model Weights

**This is the one-time 4GB download.**

```bash
# Download the 2B model (better accuracy for seed differentiation)
moondream download moondream-2b
```

You'll see progress like:
```
Downloading moondream-2b...
███████████████░░░  85%
```

**After download**, verify the model exists:

```bash
ls moondream-2b/
```

Should show a `.gguf` file (the model "brain").

**💡 Offline Mode Test:**

Once downloaded, disconnect Wi-Fi completely. The model will still work!

---

## ✅ Step 6: Test Your Camera

```bash
python test_camera.py
```

**What should happen:**
- A window opens showing your webcam feed
- Text overlay says "Camera Test - Press 'q' to quit"
- Press `q` to close

**If it doesn't work:**
- **Grant camera permissions:** System Settings → Privacy & Security → Camera → Enable for Terminal
- **Close other camera apps** (Zoom, FaceTime, etc.)
- Try unplugging/replugging external webcams

---

## ✅ Step 7: Test Moondream (Offline)

**First, disconnect Wi-Fi** (to prove it works offline).

```bash
python test_moondream.py
```

**What should happen:**
- Loads the model (~10-20 seconds first time)
- If you have `sample_seeds.jpg`, analyzes it
- Prints a description of what Moondream sees

**Example output:**
```
🌙 Testing Moondream (offline mode)...
✅ Found Moondream model
🔄 Loading model (this may take 10-20 seconds)...
✅ Model loaded successfully!
📸 Analyzing sample image...
✅ Image loaded
🔍 Running Moondream analysis...

============================================================
MOONDREAM ANALYSIS:
============================================================
I see several pumpkin seeds (large, white, oval-shaped)
and some sunflower seeds (smaller, with black and white
striped shells) arranged on a light surface.
============================================================

✅ Offline test successful!
```

**If this works, you're ready to go!**

---

## ✅ Step 8: Launch the Sifter!

**Reconnect Wi-Fi** (you just needed to test offline mode).

```bash
python sifter_simple.py
```

**What should happen:**
1. Terminal shows: "🌙 Initializing Seed Sifter..."
2. Model loads (~10-20 seconds)
3. Camera window opens with live feed
4. Status bar at top says "READY"
5. Bottom text says "Press SPACEBAR to analyze seeds..."

**Now:**
- Place seeds in front of camera (or hold up a phone showing a seed image)
- Press **SPACEBAR**
- Wait 3-5 seconds
- See analysis in **both**:
  - Video window overlay (bottom of screen)
  - Terminal output (detailed)

**To quit:** Press `q`

---

## 🎉 Success Checklist

Confirm all these work:

- [ ] `python test_camera.py` - Camera opens
- [ ] `python test_moondream.py` - Model analyzes image **with Wi-Fi OFF**
- [ ] `python sifter_simple.py` - Full app launches
- [ ] Press SPACEBAR - Capture and analysis works
- [ ] See results in video overlay **and** terminal

If all checked, **you're done!** 🎊

---

## 🧪 What to Test Next

### Test Case 1: Single Seed Type
- Place 5-10 pumpkin seeds on white paper
- Press SPACEBAR
- Does Moondream correctly identify them as pumpkin seeds?

### Test Case 2: Mixed Seeds
- Mix pumpkin and sunflower seeds
- Press SPACEBAR
- Does Moondream differentiate between the two types?

### Test Case 3: Phone Image
- Find a seed image online
- Show it on your phone to the camera
- Press SPACEBAR
- Does it still work?

---

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'moondream'"
**Fix:** Activate virtual environment first!
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Camera permission denied"
**Fix:** System Settings → Privacy & Security → Camera → Enable Terminal

### "Model not found" error
**Fix:** Download the model:
```bash
moondream download moondream-2b
```

### Analysis is slow (10+ seconds)
**Cause:** Intel Macs are slower than M1/M2/M3
**Options:**
- Switch to smaller 1.8B model: `moondream download moondream-1.8b`
- Edit `sifter_simple.py` and change `MODEL_PATH = "moondream-1.8b"`

### Video window freezes
**Fix:** Press `q` to quit cleanly, then restart

---

## 🔄 Daily Usage

Every time you want to use the sifter:

```bash
cd ~/Documents/sifter  # or wherever you cloned it
source venv/bin/activate
python sifter_simple.py
```

---

## 📁 Where Are Captured Images Saved?

All captures go to: `sifter/captures/`

Filenames: `capture_1234567890.jpg` (timestamp)

You can review these later or use for training data.

---

## 🚀 Next Steps

Once comfortable with Phase 1:

1. **Phase 2 (Counting)** - Extract counts from Moondream output
2. **Phase 3 (Bounding Boxes)** - Draw boxes around individual seeds
3. **Arduino Integration** - Replace spacebar with physical button

See [README.md](README.md) for roadmap.

---

## 🆘 Still Having Issues?

1. Check all steps above carefully
2. Make sure virtual environment is activated
3. Try restarting Terminal
4. Open an issue with:
   - Your Mac model (M1/M2/Intel?)
   - Python version (`python3 --version`)
   - Full error message

---

**Happy sifting!** 🌻
