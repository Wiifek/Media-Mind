from utils.audio_processor import process_input
from core.transcriber import transcribe_all

def main():
    source = input("Enter the path to the audio file or YouTube URL: ")
    translate = input("Do you want to translate the transcription to English? (yes/no): ").strip().lower() == 'yes'
    
    print("Processing input...")
    chunked_files = process_input(source)
    
    print("Transcribing audio...")
    full_transcription = transcribe_all(chunked_files, translate)
    
    print("\nFinal Transcription:\n")
    print(full_transcription)

if __name__ == "__main__":
    main()