import logging
import os
import sys
# Add parent directory to sys.path to find data_ingestion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_ingestion.load_docs import load_and_chunk_documents

# Configure logging
logging.basicConfig(filename="indexing.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def initialize_vector_db():
    logging.info("Starting indexing process")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {str(e)}")
        print(f"Error: Failed to initialize embeddings: {str(e)}")
        return

    docs_dir = "./legal_documents"
    try:
        # Use load_and_chunk_documents with default taluk and pincode
        all_docs = load_and_chunk_documents(docs_dir, taluk="Mysuru", pincode="570001")
        logging.info(f"Loaded {len(all_docs)} document chunks before filtering.")
        print(f"Loaded {len(all_docs)} document chunks before filtering.")
    except Exception as e:
        logging.error(f"Failed to load documents: {str(e)}")
        print(f"Error: Failed to load documents: {str(e)}")
        return

    # Filter out empty chunks to avoid empty embeddings
    all_docs = [doc for doc in all_docs if doc.page_content.strip()]
    logging.info(f"After filtering empty chunks, {len(all_docs)} documents remain.")
    print(f"After filtering empty chunks, {len(all_docs)} documents remain.")

    if all_docs:
        for doc in all_docs:
            logging.info(f"Indexed chunk from {doc.metadata['source']} (Taluk: {doc.metadata['taluk']}, Pincode: {doc.metadata['pincode']})")
            print(f"Indexed chunk from {doc.metadata['source']} (Taluk: {doc.metadata['taluk']}, Pincode: {doc.metadata['pincode']})")
        
        print(f"Total non-empty documents to embed: {len(all_docs)}")
        print("Preview content:", all_docs[0].page_content[:100])

        try:
            vector_db = Chroma.from_documents(
                documents=all_docs,
                embedding=embeddings,
                persist_directory="./chroma_db",
                collection_name="legal_docs"
            )
            # Removed vector_db.persist() as it's deprecated and automatic
            logging.info("Vector database initialized successfully")
            print("Vector database initialized successfully!")
        except Exception as e:
            logging.error(f"Failed to initialize vector database: {str(e)}")
            print(f"Error: Failed to initialize vector database: {str(e)}")
    else:
        logging.warning("No documents found to index")
        print("No documents found to index.")

if __name__ == "__main__":
    initialize_vector_db()