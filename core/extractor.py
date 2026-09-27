# Actionable items, decisions, questions, and other notes from the meeting.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()


def get_llm():
    """
    Initializes and returns a ChatGroq instance with the specified model and API key.
    """
    model_name = os.getenv("GROQ_MODEL")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.3,
        max_retries=2,
    )


def split_transcript(transcript: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> list:
    """
    Splits a transcript into smaller chunks so requests stay under the
    provider's tokens-per-minute limit.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(transcript)


def build_chain(system_prompt: str):
    '''
    Builds a processing chain for analyzing meeting transcripts based on the provided system prompt.
    '''
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )


def run_extraction(transcript: str, map_prompt: str, reduce_prompt: str, empty_label: str) -> str:
    """
    Runs a map-reduce extraction over a transcript: extracts partial results
    per chunk, then merges and deduplicates them into one final list.
    """
    chunks = split_transcript(transcript)

    map_chain = build_chain(map_prompt)
    partial_results = [map_chain.invoke(chunk) for chunk in chunks]

    # Skip the reduce call entirely if every chunk came back empty
    if all(empty_label.lower() in r.lower() for r in partial_results):
        return empty_label

    reduce_chain = build_chain(
        f"{reduce_prompt} Merge and deduplicate the items below into one final "
        f"numbered list. If nothing remains, say '{empty_label}'."
    )
    combined = "\n\n".join(partial_results)
    return reduce_chain.invoke(combined)


def extract_action_items(transcript: str) -> str:
    '''
    Extracts action items from a meeting transcript.
    '''
    map_prompt = (
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return run_extraction(
        transcript,
        map_prompt,
        "You are an expert meeting analyst reviewing partial lists of action items.",
        "No action items found.",
    )


def extract_key_decisions(transcript: str) -> str:
    '''
    Extracts key decisions from a meeting transcript.
    '''
    map_prompt = (
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return run_extraction(
        transcript,
        map_prompt,
        "You are an expert meeting analyst reviewing partial lists of key decisions.",
        "No key decisions found.",
    )


def extract_questions(transcript: str) -> str:
    '''
    Extracts unresolved questions or topics needing follow-up from a meeting transcript.
    '''
    map_prompt = (
        "From this portion of a meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return run_extraction(
        transcript,
        map_prompt,
        "You are reviewing partial lists of unresolved questions from a meeting.",
        "No open questions found.",
    )