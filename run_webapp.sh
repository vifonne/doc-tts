#!/bin/bash

# Activate virtual environment and run the Flask web app
source venv/bin/activate
echo "Starting PDF to Speech Web App..."
echo "Access the app at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
python app.py