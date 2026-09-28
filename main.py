from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize_transcript, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from db.repository import RecordingDB
from db.models import RecordingRecord


load_dotenv()

db = RecordingDB()


def run_pipeline(source: str, translate: bool = False, language: str = None) -> dict:
    print("starting Media Mind pipeline...")

    dedup_key = RecordingDB.make_key(source=source, source_name=source)
    if translate:
        dedup_key += "_translated"

    cached = db.get(dedup_key)
    if cached:
        print(f"✅ Already processed — loading cached result for '{source}'")
        # The RAG chain itself isn't stored (it's an in-memory LangChain
        # object, not serializable data), so it's rebuilt from the cached
        # transcript. Cheap compared to re-transcribing + re-summarizing.
        rag_chain = build_rag_chain(cached.transcript)
        return {
            "title": cached.title,
            "transcript": cached.transcript,
            "summary": cached.summary,
            "action_items": cached.action_items,
            "key_decisions": cached.key_decisions,
            "open_questions": cached.open_questions,
            "rag_chain": rag_chain,
        }

    chunks = process_input(source)

    transcript = transcribe_all(chunks, translate, language)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize_transcript(transcript)

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    db.save(RecordingRecord(
        dedup_key=dedup_key,
        source_name=source,
        transcript=transcript,
        title=title,
        summary=summary,
        action_items=action_items,
        key_decisions=decisions,
        open_questions=questions,
    ))

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    translate = input("Do you want to translate the transcription to English? (yes/no): ").strip().lower() == 'yes'
    if not translate:
            language = input("Spoken language code (e.g. fr, en, ar) or press Enter to auto-detect: ").strip().lower() or None
    result = run_pipeline(source, translate, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")