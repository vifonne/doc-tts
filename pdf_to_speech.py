#!/usr/bin/env python3
import sys
import pypdf
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import os
import argparse

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

def text_to_speech(text, output_path):
    """Convert text to speech using Kokoro TTS"""
    try:
        pipeline = KPipeline(lang_code='a')

        # Split text into chunks if it's too long
        max_chunk_size = 1000  # characters
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        all_audio = []

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                generator = pipeline(chunk, voice='af_heart')

                for _, _, audio in generator:
                    all_audio.append(audio)

        # Concatenate all audio chunks
        if all_audio:
            final_audio = np.concatenate(all_audio)

            # Save as WAV first
            wav_path = output_path.replace('.mp3', '.wav')
            sf.write(wav_path, final_audio, 24000)

            # Convert to MP3
            audio = AudioSegment.from_wav(wav_path)
            audio.export(output_path, format="mp3")

            # Remove temporary WAV file
            os.remove(wav_path)

            print(f"Audio saved as: {output_path}")
        else:
            print("No audio generated")

    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to speech using Kokoro TTS')
    parser.add_argument('pdf_file', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output MP3 file path', default='output.mp3')

    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found")
        sys.exit(1)

    print(f"Reading PDF: {args.pdf_file}")
    text = extract_text_from_pdf(args.pdf_file)

    if not text or not text.strip():
        print("Error: No text extracted from PDF")
        sys.exit(1)

    print(f"Extracted {len(text)} characters of text")
    print("Converting to speech...")

    text_to_speech(text, args.output)
    print("Done!")

if __name__ == "__main__":
    main()