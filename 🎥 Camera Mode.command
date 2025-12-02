#!/bin/bash
# Seed Sifter - Camera Mode
cd "$(dirname "$0")"
source venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
python sifter.py --camera
