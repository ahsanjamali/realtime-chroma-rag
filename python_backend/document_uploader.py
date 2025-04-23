import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
from langchain.schema import Document
import logging
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./new_chroma_db")
COLLECTION_NAME = "document_collection"

# Create directory if it doesn't exist
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

def get_vector_store() -> Chroma:
    """
    Get or create the ChromaDB vector store.
    
    Returns:
        Chroma: The vector store instance
    """
    try:
        embeddings = OpenAIEmbeddings()
        
        # Initialize client
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Get the vector store
        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        
        logger.info(f"Successfully connected to ChromaDB at {CHROMA_PERSIST_DIRECTORY}")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

def get_document_loader(file_path: str):
    """
    Get appropriate document loader based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        A document loader instance
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension in ['.doc', '.docx']:
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    elif file_extension in ['.csv', '.tsv']:
        return CSVLoader(file_path)
    elif file_extension in ['.html', '.htm']:
        return UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into manageable chunks.
    
    Args:
        documents: List of documents to chunk
        
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    return text_splitter.split_documents(documents)

def upload_document(file_path: str) -> bool:
    """
    Process a document and add it to the vector store.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        logger.info(f"Processing document: {file_path}")
        
        # Load the document
        loader = get_document_loader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No content extracted from {file_path}")
            return False
            
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
            
        # Chunk the documents
        logger.info(f"Chunking document into smaller pieces")
        chunks = chunk_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Get vector store and add documents
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)
        
        logger.info(f"Successfully added document to vector store")
        return True
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False

def delete_document(file_name: str) -> bool:
    """
    Delete a document from the vector store by filename.
    
    Args:
        file_name: Name of the file to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        vector_store = get_vector_store()
        
        # Get the underlying chromadb collection
        collection = vector_store._collection
        
        # Query to find documents with matching source
        results = collection.get(
            where={"source": file_name}
        )
        
        if results and 'ids' in results and results['ids']:
            # Delete the matching documents
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted document {file_name} from vector store")
            return True
        else:
            logger.warning(f"No documents found with source: {file_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False


# Provide the full path to your document
file_path = "C:/Users/ahsan/OneDrive/Documents/CSS/Outlines_of_MOCKS.pdf"  
success = upload_document(file_path)

if success:
    print(f"Successfully uploaded and processed {file_path}")
else:
    print(f"Failed to process {file_path}") 