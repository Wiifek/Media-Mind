import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings():
    """
    Initializes and returns a HuggingFaceEmbeddings instance with the specified model.

    Returns:
        HuggingFaceEmbeddings: An instance of the HuggingFaceEmbeddings class.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"} 
        )

def build_vector_store(transcript: str) -> Chroma:
    """
    Builds a Chroma vector store from a single transcript.

    Args:
        transcript (str): A transcript string.
    Returns:
        Chroma: An instance of the Chroma vector store containing the embedded transcript.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Split the transcript into chunks
    chunks = text_splitter.split_text(transcript)  

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def load_vector_store() ->Chroma:
    """
    Loads an existing Chroma vector store from the specified directory.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    """
    Returns a retriever for the given Chroma vector store.

    Args:
        vector_store (Chroma): An instance of the Chroma vector store.
        k (int): The number of similar documents to retrieve. Default is 4.

    Returns:
        Retriever: A retriever for the Chroma vector store.
    """
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )