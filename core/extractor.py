# Actionable items, decisions, questions, and other notes from the meeting.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    """
    Initializes and returns a ChatGroq instance with the specified model and API key.
    Returns:
        ChatGroq: An instance of the ChatGroq class. 
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

def build_chain(system_prompt : str):
    '''
    Builds a processing chain for analyzing meeting transcripts based on the provided system prompt.
    '''
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    '''
    Extracts action items from a meeting transcript.

    Args:
        transcript (str): The meeting transcript.

    Returns:
        str: A list of extracted action items.
    '''
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    '''
    Extracts key decisions from a meeting transcript.

    Args:
        transcript (str): The meeting transcript.

    Returns:
        str: A list of extracted key decisions.
    '''
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    '''
    Extracts unresolved questions or topics needing follow-up from a meeting transcript.
    '''
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
