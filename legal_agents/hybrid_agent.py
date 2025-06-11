from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from llm.ollama_setup import initialize_llm
from tools.web_search_tool import get_web_search_tool
import os
import logging

# Configure logging
logging.basicConfig(filename="agent.log", level=logging.INFO, format="%(asctime)s - %(message)s")

class HybridAgent:
    def __init__(self):
        self.llm = initialize_llm()
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            logging.error(f"Failed to initialize embeddings: {str(e)}")
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

        try:
            self.vector_db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings,
                collection_name="legal_docs"
            )
        except Exception as e:
            logging.error(f"Failed to initialize vector database: {str(e)}")
            raise RuntimeError(f"Failed to initialize vector database: {str(e)}")

        self.web_search_tool = get_web_search_tool()
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "language"],
            template="You are a legal assistant for rural Karnataka. Answer in simple terms (ELI5) in {language}. If {language} is Kannada and generation fails, provide the answer in English. Context: {context}\nQuestion: {question}\nAnswer:"
        )

    def process_query(self, query, language="English", taluk=None):
        try:
            if taluk:
                # Filter documents by taluk metadata
                docs = self.vector_db._collection.get(where={"taluk": taluk})
                doc_ids = docs["ids"]
                if doc_ids:
                    docs = self.vector_db.similarity_search_with_score(query, k=3, filter={"_id": {"$in": doc_ids}})
                else:
                    docs = []
            else:
                docs = self.vector_db.similarity_search_with_score(query, k=3)

            logging.info(f"Query: {query}, Taluk: {taluk}, Retrieved docs with scores: {[(doc.metadata.get('source', 'Unknown'), score) for doc, score in docs]}")
            relevant_docs = [doc for doc, score in docs if score > 0.5] 
            logging.info(f"Relevant docs after threshold (0.5): {len(relevant_docs)}")
            
            context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
            sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs] if relevant_docs else []

            if context:
                logging.info("Using vector store documents for answer.")
                prompt = self.prompt_template.format(context=context, question=query, language=language)
                answer = self.llm.invoke(prompt)
            else:
                logging.info("No relevant documents found, falling back to web search.")
                web_results = self.web_search_tool.func(query)
                context = web_results
                prompt = self.prompt_template.format(context=context, question=query, language=language)
                answer = self.llm.invoke(prompt)
                sources = ["Web search result"]

            return {"answer": answer, "sources": sources}
        except Exception as e:
            logging.error(f"Error processing query '{query}': {str(e)}")
            return {"answer": f"Error: Unable to process query due to {str(e)}", "sources": []}

def initialize_hybrid_agent():
    """
    Initialize and return a HybridAgent instance.
    """
    return HybridAgent()