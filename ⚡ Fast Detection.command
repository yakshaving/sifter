#!/bin/bash
# Seed Sifter - Fast Grid-Based Detection
cd "$(dirname "$0")"
source venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
python sifter_fast.py
