# PDF to Speech Converter

A simple Python application that converts PDF documents to speech using the Kokoro TTS model.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install kokoro soundfile pypdf pydub
```

## Usage

Convert a PDF to MP3:
```bash
python pdf_to_speech.py input.pdf -o output.mp3
```

The script will:
- Extract text from the PDF
- Convert it to speech using Kokoro TTS with the 'af_heart' voice
- Save the result as an MP3 file

## Example

```bash
python pdf_to_speech.py example.pdf -o example_audio.mp3
```

## Features

- Automatic text chunking for large documents
- Progress tracking during conversion
- MP3 output format
- Simple command-line interface

## Requirements

- Python 3.10+
- Virtual environment (recommended)
- About 20MB output file for a typical document