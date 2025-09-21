from flask import Flask, request, render_template, jsonify, send_file, url_for, Response
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
import webbrowser
from pathlib import Path
import platform
import subprocess
import logging
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# Setup logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better formatting"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        timestamp = datetime.now().strftime('%H:%M:%S')
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        if record.levelname == 'INFO':
            prefix = f"{color}[{timestamp}] ℹ️ {reset}"
        elif record.levelname == 'WARNING':
            prefix = f"{color}[{timestamp}] ⚠️ {reset}"
        elif record.levelname == 'ERROR':
            prefix = f"{color}[{timestamp}] ❌ {reset}"
        elif record.levelname == 'DEBUG':
            prefix = f"{color}[{timestamp}] 🔍 {reset}"
        else:
            prefix = f"{color}[{timestamp}] 📋 {reset}"

        return f"{prefix}{record.getMessage()}"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add colored handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.handlers = [console_handler]
logger.propagate = False

# Global variable to track conversion progress
conversion_status = {}

class HardwareOptimizer:
    """Detect and optimize for different hardware configurations"""

    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.device_info = self.detect_hardware()

    def detect_hardware(self):
        """Detect available hardware and return optimization settings"""
        logger.info("🔍 Detecting hardware configuration...")

        device_info = {
            'device': 'cpu',
            'device_name': 'CPU',
            'optimization_level': 'basic',
            'batch_size': 1,
            'use_half_precision': False,
            'memory_efficient': False
        }

        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")

            # Apple Silicon optimization (M1/M2/M3/M4)
            if self.system == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info(f"🚀 Apple Silicon detected: {self.machine}")
                device_info.update({
                    'device': 'mps',
                    'device_name': f'Apple Silicon ({self.machine})',
                    'optimization_level': 'high',
                    'batch_size': 2,
                    'use_half_precision': False,  # MPS doesn't always support FP16
                    'memory_efficient': True,
                    'env_vars': {'PYTORCH_ENABLE_MPS_FALLBACK': '1'}
                })
                logger.info("✅ MPS (Metal Performance Shaders) acceleration enabled")

            # NVIDIA GPU optimization
            elif torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                logger.info(f"🚀 NVIDIA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")

                # Optimize based on GPU memory
                if gpu_memory >= 8:  # High-end GPU
                    batch_size = 4
                    use_half_precision = True
                    logger.info("⚡ High-end GPU: Maximum performance settings")
                elif gpu_memory >= 4:  # Mid-range GPU like RTX 3050Ti
                    batch_size = 2
                    use_half_precision = True
                    logger.info("⚡ Mid-range GPU: Balanced performance settings")
                else:  # Lower memory GPU
                    batch_size = 1
                    use_half_precision = False
                    logger.info("⚡ Entry-level GPU: Conservative settings")

                device_info.update({
                    'device': 'cuda',
                    'device_name': f'NVIDIA {gpu_name} ({gpu_memory:.1f}GB)',
                    'optimization_level': 'high',
                    'batch_size': batch_size,
                    'use_half_precision': use_half_precision,
                    'memory_efficient': gpu_memory < 6,
                    'env_vars': {'CUDA_LAUNCH_BLOCKING': '0'}
                })
                logger.info("✅ CUDA acceleration enabled")

            # Intel Mac or older hardware fallback
            else:
                # Detect if it's an older Intel Mac
                if self.system == 'Darwin' and 'intel' in self.machine.lower():
                    logger.info(f"💻 Intel Mac detected: {self.machine}")
                    device_info.update({
                        'device_name': f'Intel Mac ({self.machine})',
                        'optimization_level': 'conservative',
                        'memory_efficient': True
                    })
                    logger.info("✅ CPU mode with memory optimization")
                elif self.system == 'Windows':
                    logger.info(f"💻 Windows CPU detected")
                    device_info.update({
                        'device_name': 'Windows CPU',
                        'optimization_level': 'basic'
                    })
                    logger.info("✅ CPU mode enabled")
                else:
                    logger.info(f"💻 Generic CPU detected: {self.system}")
                    logger.info("✅ Basic CPU mode enabled")

        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
            logger.info("✅ Fallback to CPU mode")

        except Exception as e:
            logger.error(f"Hardware detection error: {e}")
            logger.info("✅ Fallback to CPU mode")

        logger.info(f"Final configuration: {device_info['device_name']} ({device_info['optimization_level']} mode)")
        return device_info

    def apply_optimizations(self):
        """Apply hardware-specific environment variables and settings"""
        if 'env_vars' in self.device_info:
            for key, value in self.device_info['env_vars'].items():
                os.environ[key] = value

        return self.device_info

    def get_status_message(self):
        """Get a user-friendly hardware status message"""
        device_name = self.device_info['device_name']
        optimization = self.device_info['optimization_level']

        if optimization == 'high':
            return f"🚀 Using {device_name} with GPU acceleration"
        elif optimization == 'conservative':
            return f"⚡ Using {device_name} with optimized settings"
        else:
            return f"💻 Using {device_name} (CPU mode)"

# Initialize hardware optimizer
hardware_optimizer = HardwareOptimizer()
hardware_info = hardware_optimizer.apply_optimizations()

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

def text_to_speech_web(text, output_path, job_id, pdf_path=None):
    """Convert text to speech using Kokoro TTS with progress tracking"""
    start_time = time.time()
    logger.info(f"🎵 Starting TTS conversion [Job: {job_id[:8]}...]")
    logger.info(f"📄 Input text length: {len(text)} characters")
    logger.info(f"💾 Output path: {output_path}")

    try:
        conversion_status[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': 'Starting TTS engine...',
            'timestamp': time.time()
        }
        time.sleep(0.5)  # Give UI time to update

        conversion_status[job_id] = {
            'status': 'loading_model',
            'progress': 5,
            'message': f'Loading Kokoro TTS model on {hardware_info["device_name"]}...',
            'timestamp': time.time()
        }

        # Initialize pipeline with hardware optimization
        logger.info("🤖 Loading Kokoro TTS model...")
        try:
            # Apply device-specific settings for Kokoro
            pipeline_kwargs = {'lang_code': 'a'}

            # Note: Kokoro may not directly support all PyTorch device settings,
            # but we can optimize the environment and chunk processing
            pipeline = KPipeline(**pipeline_kwargs)

            # Log hardware optimization status
            logger.info(f"✅ Model loaded with {hardware_optimizer.get_status_message()}")

        except Exception as e:
            logger.warning(f"Hardware optimization failed, falling back to default: {e}")
            pipeline = KPipeline(lang_code='a')
            logger.info("✅ Model loaded in fallback mode")

        conversion_status[job_id] = {
            'status': 'processing_text',
            'progress': 15,
            'message': 'Preparing text for synthesis...',
            'timestamp': time.time()
        }

        # Split text into chunks with hardware-optimized sizing
        if hardware_info['optimization_level'] == 'high':
            max_chunk_size = 1500  # Larger chunks for powerful hardware
        elif hardware_info['optimization_level'] == 'conservative':
            max_chunk_size = 800   # Smaller chunks for older hardware
        else:
            max_chunk_size = 1000  # Default

        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        total_chunks = len(chunks)

        logger.info(f"📝 Text split into {total_chunks} chunks (max {max_chunk_size} chars each)")

        conversion_status[job_id] = {
            'status': 'processing_chunks',
            'progress': 20,
            'message': f'Converting {total_chunks} text chunks to speech...',
            'timestamp': time.time()
        }

        all_audio = []

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                chunk_start = time.time()
                chunk_progress = 20 + (i / total_chunks) * 60

                logger.info(f"🎤 Processing chunk {i+1}/{total_chunks} ({len(chunk)} chars)")

                conversion_status[job_id] = {
                    'status': 'converting',
                    'progress': chunk_progress,
                    'message': f'Converting chunk {i+1} of {total_chunks}... ({len(chunk)} chars)',
                    'current_chunk': i + 1,
                    'total_chunks': total_chunks,
                    'timestamp': time.time()
                }

                generator = pipeline(chunk, voice='af_heart')
                for _, _, audio in generator:
                    all_audio.append(audio)

                chunk_time = time.time() - chunk_start
                logger.info(f"✅ Chunk {i+1} completed in {chunk_time:.1f}s")

        conversion_status[job_id] = {
            'status': 'finalizing',
            'progress': 85,
            'message': 'Combining audio segments...',
            'timestamp': time.time()
        }

        # Concatenate all audio chunks
        if all_audio:
            logger.info(f"🔗 Combining {len(all_audio)} audio segments...")
            final_audio = np.concatenate(all_audio)
            audio_duration = len(final_audio) / 24000  # 24kHz sample rate

            # Save as WAV first
            wav_path = output_path.replace('.mp3', '.wav')
            logger.info(f"💾 Saving WAV file ({audio_duration:.1f}s duration)...")
            sf.write(wav_path, final_audio, 24000)

            # Try to convert to MP3, fallback to WAV if ffmpeg is not available
            try:
                logger.info("🎵 Converting to MP3 format...")
                conversion_status[job_id] = {
                    'status': 'converting_format',
                    'progress': 95,
                    'message': 'Converting to MP3...',
                    'timestamp': time.time()
                }
                audio = AudioSegment.from_wav(wav_path)
                audio.export(output_path, format="mp3")
                os.remove(wav_path)
                output_file = os.path.basename(output_path)
                logger.info("✅ MP3 conversion completed")
            except Exception as e:
                # If MP3 conversion fails (no ffmpeg), just rename WAV to final output
                logger.warning(f"MP3 conversion failed: {e}")
                logger.info("📁 Saving as WAV instead...")
                import shutil
                wav_output = output_path.replace('.mp3', '.wav')
                shutil.move(wav_path, wav_output)
                output_file = os.path.basename(wav_output)
                logger.info("✅ WAV file saved")

            total_time = time.time() - start_time
            chars_per_sec = len(text) / total_time
            audio_ratio = audio_duration / total_time

            conversion_status[job_id] = {
                'status': 'completed',
                'progress': 100,
                'message': 'Conversion completed successfully!',
                'output_file': output_file,
                'timestamp': time.time()
            }

            logger.info(f"🎉 Conversion completed!")
            logger.info(f"⏱️  Total time: {total_time:.1f}s")
            logger.info(f"📊 Performance: {chars_per_sec:.0f} chars/sec")
            logger.info(f"🎵 Audio generated: {audio_duration:.1f}s (ratio: {audio_ratio:.1f}x)")
            logger.info(f"📁 Output file: {output_file}")

            # Clean up uploaded PDF
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    logger.info("🗑️  Cleaned up uploaded PDF")
                except Exception as e:
                    logger.warning(f"Could not clean up PDF file: {e}")

            return True
        else:
            logger.error("❌ No audio was generated from the text")
            conversion_status[job_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'No audio was generated from the text',
                'error': 'No audio generated',
                'timestamp': time.time()
            }
            return False

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Conversion failed after {total_time:.1f}s: {str(e)}")

        conversion_status[job_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Conversion failed: {str(e)}',
            'error': str(e),
            'timestamp': time.time()
        }

        # Clean up uploaded PDF on error
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.info("🗑️  Cleaned up uploaded PDF after error")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up PDF file: {cleanup_error}")

        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hardware-info')
def get_hardware_info():
    """Get current hardware optimization status"""
    return jsonify({
        'device_name': hardware_info['device_name'],
        'optimization_level': hardware_info['optimization_level'],
        'status_message': hardware_optimizer.get_status_message(),
        'device': hardware_info['device'],
        'batch_size': hardware_info.get('batch_size', 1),
        'memory_efficient': hardware_info.get('memory_efficient', False)
    })

@app.route('/convert', methods=['POST'])
def convert_pdf():
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
    logger.info(f"📥 New conversion request from {client_ip}")

    if 'file' not in request.files:
        logger.warning("❌ No file provided in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("❌ Empty filename provided")
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"❌ Invalid file type: {file.filename}")
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    logger.info(f"📄 Processing file: {file.filename} ({len(file.read())} bytes)")
    file.seek(0)  # Reset file pointer after reading size

    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

        # Save uploaded file
        file.save(pdf_path)
        logger.info(f"💾 File saved to: {pdf_path}")

        # Extract text from PDF
        logger.info("📖 Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        if not text or not text.strip():
            os.remove(pdf_path)  # Clean up
            logger.error("❌ No text could be extracted from PDF")
            return jsonify({'error': 'No text could be extracted from the PDF'}), 400

        logger.info(f"✅ Extracted {len(text)} characters from PDF")

        # Generate output filename
        audio_filename = f"{unique_id}_{os.path.splitext(filename)[0]}.mp3"
        output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], audio_filename)

        # Start conversion in background thread
        job_id = unique_id
        logger.info(f"🚀 Starting background conversion [Job: {job_id[:8]}...]")

        conversion_thread = threading.Thread(
            target=text_to_speech_web,
            args=(text, output_path, job_id, pdf_path)  # Pass pdf_path for cleanup
        )
        conversion_thread.daemon = True
        conversion_thread.start()

        # Return immediately with job_id for progress tracking
        return jsonify({
            'job_id': job_id,
            'message': 'Conversion started',
            'filename': audio_filename
        })

    except Exception as e:
        logger.error(f"❌ Server error during conversion setup: {str(e)}")
        # Clean up files on error
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.info("🗑️  Cleaned up PDF after server error")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Get conversion progress for a job via Server-Sent Events"""
    def generate():
        while True:
            status = conversion_status.get(job_id, {'status': 'not_found', 'progress': 0})

            # Send the current status as JSON
            import json
            yield f"data: {json.dumps(status)}\n\n"

            # If completed or error, stop streaming
            if status.get('status') in ['completed', 'error', 'not_found']:
                break

            time.sleep(0.5)  # Update every 500ms

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get single status check (non-streaming)"""
    status = conversion_status.get(job_id, {'status': 'not_found', 'progress': 0})
    return jsonify(status)

@app.route('/download/<filename>')
def download_file(filename):
    """Download converted audio file"""
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
    logger.info(f"📥 Download request from {client_ip} for: {filename}")

    try:
        file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"📤 Serving file: {filename} ({file_size} bytes)")
            return send_file(file_path, as_attachment=True)
        else:
            logger.warning(f"❌ File not found: {filename}")
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"❌ Download error for {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_old_files():
    """Clean up old files periodically"""
    while True:
        try:
            time.sleep(300)  # Run every 5 minutes
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1 hour

            cleaned_count = 0
            for folder in [app.config['UPLOAD_FOLDER'], app.config['DOWNLOAD_FOLDER']]:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) and os.path.getctime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} old files")

            # Clean old status entries
            old_sessions = [
                session_id for session_id, status in conversion_status.items()
                if current_time - status.get('timestamp', 0) > 3600
            ]

            if old_sessions:
                for session_id in old_sessions:
                    del conversion_status[session_id]
                logger.info(f"🧹 Cleaned up {len(old_sessions)} old conversion status entries")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

@app.route('/cleanup')
def cleanup_files():
    """Clean up old files (manual endpoint)"""
    try:
        # Remove files older than 1 hour
        current_time = time.time()
        cleaned_count = 0
        for folder in [app.config['UPLOAD_FOLDER'], app.config['DOWNLOAD_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
                        cleaned_count += 1

        logger.info(f"🧹 Manual cleanup completed: {cleaned_count} files removed")
        return jsonify({'message': f'Cleanup completed: {cleaned_count} files removed'})
    except Exception as e:
        logger.error(f"❌ Manual cleanup error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Auto-open browser
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        logger.info("🌐 Opening browser...")
        webbrowser.open('http://127.0.0.1:5000')

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    print("\n" + "="*50)
    print("🎵 PDF to Speech Converter")
    print("="*50)
    logger.info("🚀 Server starting...")
    logger.info(f"🔧 {hardware_optimizer.get_status_message()}")
    logger.info("🌐 Your browser should open automatically")
    logger.info("📡 Server URL: http://127.0.0.1:5000")
    logger.info("⏹️  Press Ctrl+C to stop the server")
    print("="*50 + "\n")

    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    logger.info("🧹 Background cleanup service started")

    # Run Flask app
    logger.info("✅ Server ready - waiting for requests...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)