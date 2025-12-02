#!/bin/bash
# Seed Sifter - Main Launcher
cd "$(dirname "$0")"
source venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
python sifter.py
