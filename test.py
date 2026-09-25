from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize_transcript, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions

def main():
    source = input("Enter the path to the audio file or YouTube URL: ")
    translate = input("Do you want to translate the transcription to English? (yes/no): ").strip().lower() == 'yes'
    
    print("Processing input...")
    chunked_files = process_input(source)
    
    print("Transcribing audio...")
    full_transcription = transcribe_all(chunked_files, translate)
    
    print("📝 TRANSCRIPT")
    print("=" * 60)
    print(full_transcription[:500] + "..." if len(full_transcription) > 500 else full_transcription)


    title = generate_title(full_transcription)
    summary = summarize_transcript(full_transcription)

    print("\n" + "=" * 60)
    print(f"TITLE: {title}")
    print("=" * 60)
    print("\n📋 SUMMARY")
    print("-" * 60)
    print(summary)



    action_items = extract_action_items(full_transcription)
    decisions = extract_key_decisions(full_transcription)
    questions = extract_questions(full_transcription)

    print("\n" + "=" * 60)
    print("ACTION ITEMS")
    print("=" * 60)
    print(action_items)

    print("\n" + "=" * 60)
    print("KEY DECISIONS")
    print("=" * 60)
    print(decisions)

    print("\n" + "=" * 60)
    print("OPEN QUESTIONS")
    print("=" * 60)
    print(questions)
    
if __name__ == "__main__":
    main()