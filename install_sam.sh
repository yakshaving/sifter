#!/bin/bash
# Install SAM (Segment Anything Model) for improved seed detection

echo "🤖 Installing SAM (Segment Anything Model)..."
echo "=============================================="

# Activate virtual environment
source venv/bin/activate

# Install SAM
echo "📦 Installing segment-anything package..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Check if we need to download the model
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo ""
    echo "📥 Downloading SAM model checkpoint..."
    echo "   Using smaller ViT-B model for faster performance..."

    # Download smaller model (ViT-B) - faster and sufficient for our use case
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

    echo ""
    echo "✅ Model downloaded: sam_vit_b_01ec64.pth"
    echo ""
    echo "Optional: For best accuracy, download the larger model:"
    echo "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
    echo "✅ SAM checkpoint already exists"
fi

echo ""
echo "=============================================="
echo "✅ SAM installation complete!"
echo ""
echo "Usage:"
echo "  python sifter_sam.py captures/capture_1764557172.jpg"
echo ""
