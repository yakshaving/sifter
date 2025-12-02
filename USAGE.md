# 🌱 Seed Sifter - Quick Start Guide

## Easy Launch (Double-Click)

Just double-click one of these files in Finder:

- **🌱 Seed Sifter.command** - Interactive menu (choose mode)
- **🎥 Camera Mode.command** - Live camera detection
- **🤖 AI Analysis.command** - AI-powered counting (most accurate)

## Command Line Usage

### Interactive Menu
```bash
python sifter.py
```
Shows a menu where you can choose:
1. Live Camera
2. Analyze Image File
3. Analyze Folder of Images
4. AI Analysis (Moondream)

### Direct Commands

**Live Camera:**
```bash
python sifter.py --camera
```

**Analyze a Single Image:**
```bash
python sifter.py --image path/to/image.jpg
```

**Analyze All Images in a Folder:**
```bash
python sifter.py --folder path/to/folder
```

**AI Analysis (Moondream):**
```bash
python sifter.py --ai
```

## Examples

### Analyze a saved photo:
```bash
python sifter.py --image captures/capture_1764557172.jpg
```

### Analyze all photos in captures folder:
```bash
python sifter.py --folder captures
```

### Quick test with an image:
```bash
# Interactive - will ask for file path
python sifter.py
# Choose option 2, then enter: captures/capture_1764557172.jpg
```

## Controls

### Camera Mode:
- **SPACEBAR**: Analyze current frame
- **Q**: Quit

### Image/Folder Mode:
- Automatically shows results
- Press any key to close result window

## Adding Custom Icons (Optional)

To add custom icons to the .command files:

1. Find or create a seed icon (PNG or ICNS)
2. Right-click the .command file → Get Info
3. Drag your icon onto the small icon in top-left
4. The file will now show your custom icon in Finder!

Suggested icon sources:
- https://www.flaticon.com (search "seed")
- Create your own with Preview or any image editor
- Use emoji as icon (🌱 🌻 🎃)

## Tips

- **Best results**: Use good lighting and spread seeds out
- **Wood backgrounds**: May have some false positives (use AI mode for accuracy)
- **Overlapping seeds**: AI mode works best
- **Speed vs Accuracy**: Camera mode is fast, AI mode is accurate
