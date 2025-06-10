from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from data_ingestion.load_docs import load_and_chunk_documents
import os

def initialize_vector_db(data_dir: str, collection_name: str = "legal_docs"):
    """
    Initialize ChromaDB with legal documents.
    """
    # Load and chunk documents
    documents = load_and_chunk_documents(data_dir, taluk="Mysuru", pincode="570001")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create ChromaDB vector store
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    return vector_db

if __name__ == "__main__":
    data_dir = "./legal_documents"
    os.makedirs(data_dir, exist_ok=True)
    vector_db = initialize_vector_db(data_dir)
    print("Vector DB initialized with documents.")