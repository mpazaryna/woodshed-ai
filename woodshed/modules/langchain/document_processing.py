from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def load_documents(path):
    """
    Load text documents from a specified directory.

    This function uses the DirectoryLoader to load all text files
    from the given path. It expects the files to have a .txt extension.

    Args:
        path (str or Path): The path to the directory containing text files.

    Returns:
        list: A list of documents loaded from the text files.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If no text files are found in the directory.
    """
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)
    return loader.load()


def split_text(documents):
    """
    Split a list of documents into smaller chunks.

    This function uses the RecursiveCharacterTextSplitter to divide
    the documents into smaller segments based on specified chunk size
    and overlap.

    Args:
        documents (list): A list of documents to be split.

    Returns:
        list: A list of text chunks created from the input documents.

    Raises:
        ValueError: If the input documents list is empty.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_vectordb(texts, persist_directory):
    """
    Create a vector database from a list of text documents.

    This function initializes the OpenAIEmbeddings and creates a Chroma
    vector store from the provided documents. The vector store is set
    to automatically persist the documents.

    Args:
        texts (list): A list of text documents to be stored in the vector database.
        persist_directory (str or Path): The directory where the vector database will be persisted.

    Returns:
        Chroma: An instance of the Chroma vector store containing the embedded documents.

    Raises:
        ValueError: If the input texts list is empty or if the persist directory is invalid.
    """
    embedding = OpenAIEmbeddings()
    persist_directory_str = str(persist_directory)  # Convert PosixPath to string
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embedding, persist_directory=persist_directory_str
    )
    # vectordb.persist()  # This line is no longer needed as persistence is automatic
    return vectordb
