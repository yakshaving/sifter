#!/bin/bash
# Seed Sifter Run Script
# Makes it easy to run the application with all required settings

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set library path for libvips
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Run the application
python sifter_simple.py
