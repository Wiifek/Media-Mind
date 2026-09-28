import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever, load_vector_store
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

def format_docs(docs):
    """
    Formats a list of documents into a single string for processing.

    Args:
        docs (list): A list of Document objects.

    Returns:
        str: A formatted string containing the content of all documents.
    """
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript : str):
    """
    Builds a processing chain for analyzing meeting transcripts based on the provided system prompt.

    Args:
        transcript (str): The meeting transcript to analyze.
    Returns:
        Runnable: A runnable chain for processing the transcript.
    """
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k= 4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
            based ONLY on the meeting transcript context provided below.

            If the answer is not found in the context, say: 
            "I could not find this information in the meeting transcript."

            Always be concise and precise. If quoting someone, mention it clearly.

            Context from meeting transcript:
            {context}""",
        ),
        ("human", "{question}"),
    ]
    )

    #full LCEL Rag pipeline 

    rag_chain = (

        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
            based ONLY on the meeting transcript context provided below.

            If the answer is not found in the context, say: 
            "I could not find this information in the meeting transcript."

            Always be concise and precise. If quoting someone, mention it clearly.

            Context from meeting transcript:
            {context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriever| RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question:str) -> str:
    answer = rag_chain.invoke(question)
    return answer