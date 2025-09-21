#!/bin/bash

# Make this script executable and double-clickable on macOS

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "================================================="
echo "   PDF to Speech Converter - macOS Launcher"
echo "================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python from https://python.org"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Python found:"
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Check for ffmpeg (needed for MP3 conversion)
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Checking for Homebrew..."
    if command -v brew &> /dev/null; then
        echo "Installing ffmpeg via Homebrew..."
        brew install ffmpeg
    else
        echo "⚠️  Warning: ffmpeg not found and Homebrew not installed."
        echo "Audio will be saved as WAV instead of MP3."
        echo "To get MP3 output, install ffmpeg:"
        echo "  - Install Homebrew: https://brew.sh"
        echo "  - Then run: brew install ffmpeg"
        echo ""
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/bin/flask" ]; then
    echo "Installing dependencies..."
    echo "This may take a few minutes on first run..."

    # Install requirements
    pip install flask pypdf soundfile pydub torch kokoro spacy

    # Install spaCy language model
    echo "Installing spaCy language model..."
    python -m spacy download en_core_web_sm

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "Dependencies installed successfully!"
fi

# Start the application
echo ""
echo "Starting PDF to Speech Converter..."
echo "Your browser should open automatically."
echo ""
echo "Hardware optimization will be detected automatically:"
echo "• Apple Silicon (M1/M2/M3/M4): GPU acceleration enabled"
echo "• NVIDIA GPU: CUDA acceleration enabled"
echo "• Intel Mac/CPU: Optimized for efficiency"
echo ""
echo "To stop the server, close this window or press Ctrl+C"
echo ""

python app.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Application failed to start"
    read -p "Press Enter to exit..."
fi