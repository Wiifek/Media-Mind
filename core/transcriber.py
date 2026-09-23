import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully.")
    return _model

def transcribe_chunk(chunk_path:str, translate: bool = False) -> str:
    """
    Transcribes a single audio chunk using the Whisper model.
    
    Args:
        chunk_path (str): The path to the audio chunk file.
        translate (bool): Whether to translate the transcription to English.
        
    Returns:
        str: The transcribed text from the audio chunk.
    """
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    return result['text']

def transcribe_all(chunk_paths:list, translate: bool = False) -> str:
    """
    Transcribes all audio chunks and combines the results.
    
    Args:
        chunk_paths (list): A list of paths to the audio chunk files.
        translate (bool): Whether to translate the transcriptions to English.
        
    Returns:
        str: The combined transcribed text from all audio chunks.
    """
    full_transcription = ""
    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i + 1}")
        transcription = transcribe_chunk(chunk_path, translate)
        full_transcription += transcription + " "
    print("All chunks transcribed successfully.")
    return full_transcription.strip()