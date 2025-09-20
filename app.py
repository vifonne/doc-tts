from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
import uuid
import pypdf
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DOWNLOAD_FOLDER'] = 'static/downloads'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# Global variable to track conversion progress
conversion_status = {}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def text_to_speech_web(text, output_path, job_id):
    """Convert text to speech using Kokoro TTS with progress tracking"""
    try:
        conversion_status[job_id] = {'status': 'initializing', 'progress': 0}

        pipeline = KPipeline(lang_code='a')
        conversion_status[job_id] = {'status': 'processing', 'progress': 10}

        # Split text into chunks if it's too long
        max_chunk_size = 1000  # characters
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        all_audio = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                conversion_status[job_id] = {
                    'status': f'processing chunk {i+1}/{total_chunks}',
                    'progress': 10 + (i / total_chunks) * 70
                }

                generator = pipeline(chunk, voice='af_heart')
                for _, _, audio in generator:
                    all_audio.append(audio)

        conversion_status[job_id] = {'status': 'finalizing audio', 'progress': 85}

        # Concatenate all audio chunks
        if all_audio:
            final_audio = np.concatenate(all_audio)

            # Save as WAV first
            wav_path = output_path.replace('.mp3', '.wav')
            sf.write(wav_path, final_audio, 24000)

            conversion_status[job_id] = {'status': 'converting to MP3', 'progress': 95}

            # Convert to MP3
            audio = AudioSegment.from_wav(wav_path)
            audio.export(output_path, format="mp3")

            # Remove temporary WAV file
            os.remove(wav_path)

            conversion_status[job_id] = {'status': 'completed', 'progress': 100}
            return True
        else:
            conversion_status[job_id] = {'status': 'error', 'error': 'No audio generated'}
            return False

    except Exception as e:
        conversion_status[job_id] = {'status': 'error', 'error': str(e)}
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

        # Save uploaded file
        file.save(pdf_path)

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text or not text.strip():
            os.remove(pdf_path)  # Clean up
            return jsonify({'error': 'No text could be extracted from the PDF'}), 400

        # Generate output filename
        audio_filename = f"{unique_id}_{os.path.splitext(filename)[0]}.mp3"
        output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], audio_filename)

        # Start conversion in background thread
        job_id = unique_id
        conversion_thread = threading.Thread(
            target=text_to_speech_web,
            args=(text, output_path, job_id)
        )
        conversion_thread.start()

        # Wait for completion (for simplicity - in production you'd use proper job queuing)
        conversion_thread.join()

        # Clean up uploaded PDF
        os.remove(pdf_path)

        # Check conversion result
        if job_id in conversion_status and conversion_status[job_id]['status'] == 'completed':
            return jsonify({
                'message': 'Conversion completed successfully',
                'filename': audio_filename,
                'download_url': url_for('download_file', filename=audio_filename)
            })
        else:
            error_msg = conversion_status.get(job_id, {}).get('error', 'Unknown error occurred')
            return jsonify({'error': f'Conversion failed: {error_msg}'}), 500

    except Exception as e:
        # Clean up files on error
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Get conversion progress for a job"""
    status = conversion_status.get(job_id, {'status': 'not_found', 'progress': 0})
    return jsonify(status)

@app.route('/download/<filename>')
def download_file(filename):
    """Download converted audio file"""
    try:
        file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup_files():
    """Clean up old files (you might want to run this periodically)"""
    try:
        # Remove files older than 1 hour
        current_time = time.time()
        for folder in [app.config['UPLOAD_FOLDER'], app.config['DOWNLOAD_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
        return jsonify({'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting PDF to Speech Web App...")
    print("Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)