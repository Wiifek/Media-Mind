import hashlib

import yt_dlp 
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url:str) -> str:
    """
    Downloads audio from a YouTube URL and saves it as an MP3 file.
    
    Args:
        url (str): The YouTube video URL.
        
    Returns:
        str: The path to the downloaded MP3 file.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        #'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp3"
    return audio_file

def convert_to_wav(input_file:str) -> str:
    """
    Converts any audio/video file to WAV format using pydub.
    
    Args:
        input_file (str): The path to the input audio file.
        
    Returns:
        str: The path to the converted WAV file.
    """
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
    audio.export(output_file, format="wav")
    return output_file

def chunk_audio(input_file:str, chunk_minutes:int=10) -> list:
    """
    Splits a WAV audio file into smaller chunks.
    
    Args:
        input_file (str): The path to the input WAV file.
        chunk_minutes (int): The length of each chunk in minutes. Default is 10 minutes.
        
    Returns:
        list: A list of paths to the chunked audio files.
    """
    audio = AudioSegment.from_file(input_file).set_channels(1).set_frame_rate(16000)
    chunk_length_ms = chunk_minutes * 60 * 1000  # Convert minutes to milliseconds
    chunks = []

    safe_id = hashlib.md5(input_file.encode("utf-8")).hexdigest()[:8]

    for i,start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_filename = os.path.join(DOWNLOAD_DIR, f"chunk_{safe_id}_{i}.mp3")
        chunk.export(chunk_filename, format="mp3", bitrate="48k")
        chunks.append(chunk_filename)

    
    return chunks

def process_input(source:str) -> list:
    """
    Processes an input audio file by converting it to WAV format and chunking it.
    
    Args:
        source (str): The path to the input audio file or a YouTube URL.
        
    Returns:
        list: A list of paths to the processed audio files.
    """
    if source.startswith("http://") or source.startswith("https://"):
        print("Detecting YouTube URL. Downloading audio...")
        audio_path = download_youtube_audio(source)
    else:
        print("Detected local file.")
        audio_path = source
 
    print("Chunking audio...")
    chunked_files = chunk_audio(audio_path)
    print(f"Audio processing complete. Generated {len(chunked_files)} chunks.")
    return chunked_files
