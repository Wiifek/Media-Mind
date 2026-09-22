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
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        #'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav').replace('.mp4', '.wav')

    return audio_file

data = download_youtube_audio("https://www.youtube.com/watch?v=PmpFmclctuM")

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
    audio = AudioSegment.from_wav(input_file)
    chunk_length_ms = chunk_minutes * 60 * 1000  # Convert minutes to milliseconds
    chunks = []
    
    for i,start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_filename = f"{input_file}_chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
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
    if(source.startswith("http://") or source.startswith("https://")):
        print("Detecting YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunked_files = chunk_audio(wav_path)
    print(f"Audio processing complete. Generated {len(chunked_files)} chunks.")
    return chunked_files
