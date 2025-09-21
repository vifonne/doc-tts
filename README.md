# 🎵 PDF to Speech Converter

Convert your PDF documents to natural-sounding speech using AI-powered text-to-speech technology.

## ✨ Features

- **Drag & Drop Interface**: Simply drag PDF files into the browser
- **High-Quality Speech**: Powered by Kokoro TTS model
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Hardware Optimized**: Automatically uses GPU acceleration when available
- **No Installation Required**: Just double-click to start
- **Beautiful Web Interface**: Modern, responsive design
- **Real-Time Progress**: See conversion progress in real-time

## 🚀 Quick Start

### Windows
1. Double-click `start_app.bat`
2. Wait for your browser to open automatically
3. Drag your PDF file and click "Convert to Speech"

### macOS
1. Double-click `start_app.command`
2. Wait for your browser to open automatically
3. Drag your PDF file and click "Convert to Speech"

### Linux
1. Open terminal and run `./start_app.sh`
2. Wait for your browser to open automatically
3. Drag your PDF file and click "Convert to Speech"

## 📋 Requirements

- Python 3.8 or higher
- Internet connection (for first-time setup)
- FFmpeg (optional, for MP3 output - otherwise WAV will be generated)

## 🚀 Hardware Acceleration

The app automatically detects and optimizes for your hardware:

- **Apple Silicon (M1/M2/M3/M4)**: Uses Metal Performance Shaders for GPU acceleration
- **NVIDIA GPUs**: Uses CUDA acceleration (RTX 3050Ti, RTX 4090, etc.)
- **Intel Macs & CPUs**: Optimized for efficiency with memory management
- **Automatic Detection**: No configuration needed - just run and go!

## 🔧 Manual Installation

If you prefer to install manually:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Run the app
python app.py
```

## 📱 How to Use

1. **Start the App**: Use one of the launcher scripts above
2. **Upload PDF**: Drag and drop your PDF file or click to browse
3. **Convert**: Click "Convert to Speech" button
4. **Download**: Once complete, download your MP3 file

## 🎯 Supported Formats

- **Input**: PDF files (up to 100MB)
- **Output**: MP3 audio files (with ffmpeg) or WAV audio files (fallback)

## 🛠️ Troubleshooting

### App won't start
- Ensure Python 3.8+ is installed
- Check your internet connection for first-time setup
- Try running manually: `python app.py`

### Conversion fails
- Ensure the PDF contains readable text (not just images)
- Check that the PDF isn't password protected
- Verify the file size is under 100MB

### Audio quality issues
- The app uses high-quality Kokoro TTS model
- Audio is generated at 24kHz sample rate
- Different voice options may be added in future updates

### Getting WAV instead of MP3
- This happens when ffmpeg is not installed
- **Windows**: Install via `winget install ffmpeg` or download from ffmpeg.org
- **macOS**: Install via `brew install ffmpeg` (requires Homebrew)
- **Linux**: Install via `sudo apt install ffmpeg` (Ubuntu/Debian)

## 📁 Project Structure

```
pdf-to-speech/
├── app.py                 # Main Flask application
├── start_app.bat         # Windows launcher
├── start_app.command     # macOS launcher
├── start_app.sh          # Linux launcher
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface
├── uploads/             # Temporary PDF storage
└── outputs/             # Generated audio files
```

## 🤝 Contributing

This is a simple, self-contained application. Feel free to:
- Report bugs or issues
- Suggest new features
- Submit improvements

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: The first run may take a few minutes as it downloads and installs dependencies. Subsequent runs will be much faster!