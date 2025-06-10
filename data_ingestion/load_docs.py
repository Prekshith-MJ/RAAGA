import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from pathlib import Path

def load_and_chunk_documents(data_dir: str, taluk: str = None, pincode: str = None) -> List[Dict]:
    """
    Load and chunk legal documents from a directory, adding metadata for taluk/pincode.
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Iterate through files in data_dir
    for file_path in Path(data_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update({
                "source": str(file_path),
                "taluk": taluk or "unknown",
                "pincode": pincode or "unknown",
                "language": "en" if "en" in file_path.name.lower() else "kn"
            })
        documents.extend(chunks)
    
    return documents

if __name__ == "__main__":
    data_dir = "./legal_documents"  # Directory containing Karnataka legal PDFs
    os.makedirs(data_dir, exist_ok=True)
    docs = load_and_chunk_documents(data_dir, taluk="Mysuru", pincode="570001")
    print(f"Loaded {len(docs)} document chunks.")