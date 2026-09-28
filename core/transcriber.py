from groq import Groq
import os

from typing import Optional

# whisper-large-v3-turbo : multilingual, best price/performance (default)
# whisper-large-v3       : multilingual, most accurate
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
# Groq's translation endpoint (any language -> English) uses whisper-large-v3.
GROQ_TRANSLATE_MODEL = os.getenv("GROQ_TRANSLATE_MODEL", "whisper-large-v3")
 
# Groq free-tier upload limit is 25 MB per request (keep a small margin).
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

_model = None

def load_model() -> Groq:
    global _model
    if _model is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
            )
        _model = Groq(api_key=api_key, max_retries=4)
    return _model

def transcribe_chunk(chunk_path:str, translate: bool = False, language: Optional[str] = None) -> str:
    """
    Transcribes a single audio chunk using Groq's hosted Whisper API.
 
    Args:
        chunk_path (str): The path to the audio chunk file.
        translate (bool): Whether to translate the transcription to English.
        language (str | None): ISO-639-1 code of the spoken language (e.g. "fr").
            Setting it improves accuracy; None = auto-detect. Ignored when translating.
 
    Returns:
        str: The transcribed text from the audio chunk.
    """
    size = os.path.getsize(chunk_path)
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Chunk {chunk_path} is {size / 1024 / 1024:.1f} MB, above Groq's "
            "25 MB upload limit. Use shorter chunks or export them as compressed "
            "mono audio (e.g. mp3 at 48k)."
        )
    model = load_model()
    with open(chunk_path, "rb") as f:
        audio = (os.path.basename(chunk_path), f.read())
 
    if translate:
        result = model.audio.translations.create(
            file=audio,
            model=GROQ_TRANSLATE_MODEL,
            temperature=0.0,
        )
    else:
        kwargs = dict(file=audio, model=GROQ_WHISPER_MODEL, temperature=0.0)
        if language:
            kwargs["language"] = language
        result = model.audio.transcriptions.create(**kwargs)
 
    return result.text.strip()

def transcribe_all(chunk_paths:list, translate: bool = False, language: Optional[str] = None) -> str:
    """
    Transcribes all audio chunks and combines the results.
 
    Args:
        chunk_paths (list): A list of paths to the audio chunk files.
        translate (bool): Whether to translate the transcriptions to English.
        language (str | None): Spoken language code, or None to auto-detect.
 
    Returns:
        str: The combined transcribed text from all audio chunks.
    """
    parts = []
    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i + 1}/{len(chunk_paths)}")
        text = transcribe_chunk(chunk_path, translate, language)
        if text:
            parts.append(text)
    print("All chunks transcribed successfully.")
    return " ".join(parts)