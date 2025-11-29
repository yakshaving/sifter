#!/bin/bash
# Launcher script for Seed Sifter
# Sets up required environment variables and runs the app

cd "$(dirname "$0")"
source venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
python -u sifter_simple.py "$@"
