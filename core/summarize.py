from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()


def get_llm():
    """
    Initializes and returns a ChatMistralAI instance with the specified model and API key.

    Returns:
        ChatMistralAI: An instance of the ChatMistralAI class.
    """
    model_name = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in the environment variables.")

    return ChatMistralAI(model=model_name, api_key=api_key, temperature=0.2)


def split_transcript(transcript: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> list:
    """
    Splits a transcript into smaller chunks for processing.

    Args:
        transcript (str): The full transcript text to be split.
        chunk_size (int): The maximum size of each chunk. Default is 3000 characters.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 200 characters.

    Returns:
        list: A list of transcript chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(transcript)


def summarize_transcript(transcript: str) -> str:
    """
    Summarizes a given transcript using a map-reduce approach: each chunk is
    summarized individually, then the partial summaries are synthesized into
    one final professional summary.

    Args:
        transcript (str): The full transcript text to be summarized.

    Returns:
        str: The final summarized text, formatted in bullet points.
    """
    llm = get_llm()
    chunks = split_transcript(transcript)

    # Map step: summarize each chunk independently
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    # Reduce step: synthesize partial summaries into one final summary
    combined = "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    """
    Generates a short professional title for the given transcript using the Mistral LLM.

    Args:
        transcript (str): The full transcript text for which to generate a title.

    Returns:
        str: The generated title.
    """
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])