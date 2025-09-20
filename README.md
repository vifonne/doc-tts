# PDF to Speech Converter

A simple Python application that converts PDF documents to speech using the Kokoro TTS model. Available as both a command-line tool and a web application.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install kokoro soundfile pypdf pydub flask
```

## Usage

### Web Application (Recommended)

1. Start the web server:
```bash
./run_webapp.sh
# or manually: source venv/bin/activate && python app.py
```

2. Open your browser and go to: `http://localhost:5000`

3. Upload a PDF file and click "Convert to Speech"

4. Download the generated MP3 file

### Command Line

Convert a PDF to MP3:
```bash
python pdf_to_speech.py input.pdf -o output.mp3
```

## Features

### Web App Features
- Drag & drop PDF upload interface
- Real-time conversion progress tracking
- Automatic file download
- Clean, responsive web interface
- File cleanup (removes old files after 1 hour)
- 16MB maximum file size limit

### CLI Features
- Automatic text chunking for large documents
- Progress tracking during conversion
- MP3 output format
- Simple command-line interface

## Technical Details

- Uses Kokoro TTS model (82M parameters) with 'af_heart' voice
- Extracts text from PDF using pypdf
- Chunks large texts for better processing
- Converts to MP3 format via WAV intermediate
- Flask web framework for the web interface

## Requirements

- Python 3.10+
- Virtual environment (recommended)
- About 20MB output file for a typical document
- Web browser (for web interface)

## File Structure

```
├── app.py                 # Flask web application
├── pdf_to_speech.py      # Command-line script
├── run_webapp.sh         # Web app startup script
├── templates/
│   └── index.html        # Web interface template
├── static/
│   ├── uploads/          # Temporary PDF uploads
│   └── downloads/        # Generated audio files
└── venv/                 # Python virtual environment
```