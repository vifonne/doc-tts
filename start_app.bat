@echo off
title PDF to Speech Converter - Starting...

echo.
echo =================================================
echo    PDF to Speech Converter - Windows Launcher
echo =================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from https://python.org
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Check for ffmpeg (for MP3 conversion)
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg not found
    echo Audio will be saved as WAV instead of MP3
    echo.
    echo To get MP3 output, install ffmpeg:
    echo 1. Download from: https://ffmpeg.org/download.html
    echo 2. Or use chocolatey: choco install ffmpeg
    echo 3. Or use winget: winget install ffmpeg
    echo.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if requirements are installed
if not exist "venv\Scripts\flask.exe" (
    echo Installing dependencies...
    echo This may take a few minutes on first run...

    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo Installing other dependencies...
    pip install flask pypdf soundfile pydub kokoro spacy

    echo Installing spaCy language model...
    python -m spacy download en_core_web_sm

    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Dependencies installed successfully!
)

:: Start the application
echo.
echo Starting PDF to Speech Converter...
echo Your browser should open automatically.
echo.
echo To stop the server, close this window or press Ctrl+C
echo.

python app.py

:: Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    pause
)